# Model lifecycle: predictions log, monitoring, retraining

The runtime loop that watches the deployed Stats model and retrains it
automatically when it degrades. See `docs/model_comparison.md` for how the
model was originally picked and tuned; this document covers what happens to
it *after* it's deployed.

```
/argue (service.py)  --logs-->  predictions_log
                                       |
weekly ingestion (event/{gw}/live)     | (actual results land in silver)
                                       v
                          monitor.py: rolling RMSE vs baseline
                                       |
                        sustained degradation? --yes--> retrain.py
                                       |                    |
                                      no                    v
                                (log & stop)        combined dataset -> tune.py's
                                                     RMSE objective + selection rule
                                                            |
                                                            v
                                              models/stats_model.pkl (+ baseline)
```

## Predictions log

Every real `/argue` call logs one row per recommended player to Postgres'
`predictions_log` table (schema owned by `ingestion/db.py`, per CLAUDE.md's
"Data store" convention):

| column | meaning |
|---|---|
| `player_id` | who the prediction is for |
| `gw` | which gameweek it's a prediction *for* |
| `predicted_points` | the model's number |
| `model_version` | which exact model made it (see below) |
| `logged_at` | when |

It's **append-only** (like `bronze_responses`) - a prediction is a
point-in-time record, not something a later call replaces. If the same
player/gameweek gets scored more than once (e.g. re-run mid-week as fresh
data comes in), monitoring uses the *latest* logged row for that pair, not
an average of all of them.

Logging is best-effort: a DB hiccup must never cost a live scoring response,
so it's wrapped in its own try/except in `service.py`, the same principle
that already governs the LLM call in `reasoning.py`. Set
`STATS_AGENT_DISABLE_PREDICTION_LOG=1` to turn it off entirely (the test
suite does, so it never writes synthetic player ids into the real table).

### Model versioning

`train.py::_model_version` hashes the model's name + its own
`get_params()` + the save timestamp into a short id (e.g. `1f43693b9eb1`),
stored both inside `stats_model.pkl` (`model_version` field, read by
`StatsModel.model_version` at load time) and in every logged prediction.

This is the whole reason monitoring can survive a retrain cleanly: the
rolling RMSE computation (`monitor.rolling_rmse`) filters predictions by
`model_version`, so a retrain doesn't silently blend the old model's
predictions and the new model's predictions into one rolling number - it
just starts a fresh window for the new version, by construction.

## The baseline: `models/model_baseline.json`

The "backtest baseline" the monitoring job compares live performance
against isn't hardcoded - it's written automatically by
`train.py::save_model()` every time a model is deployed (by `train.py`,
`tune.py`, or `retrain.py` - they all funnel through the same function):

```json
{
  "model_version": "1f43693b9eb1",
  "model_name": "XGBoost (tuned)",
  "created_utc": "2026-09-12T15:09:30Z",
  "holdout_metrics": {"mae": 0.976, "rmse": 1.930, "spearman": 0.712, "beats_naive": true},
  "split": {"warmup_last_gw": 5, "train_gws": "6-29", "valid_gws": "30-38", "method": "chronological"}
}
```

This means updating the baseline is "deploy a new model" - never a manual
edit to remember to make. `monitor.py` reads `holdout_metrics.rmse` as the
number rolling performance is judged against, and `model_version` to know
which predictions actually belong to the currently-deployed model.

## Monitoring job (`agents/stats/monitor.py`)

Run: `python -m agents.stats.monitor` - intended to run as its own step
right after the weekly ingestion job (`ingestion/weekly.py`) has pulled a
finished gameweek's real results into silver, so `player_gameweek_stats`
has the actuals to join against. It's a separate CLI/Job, not called
automatically from `ingestion/weekly.py`, matching the "one Job per
concern" shape ARCHITECTURE.md's Kubernetes topology already uses.

Each run (`evaluate_and_log`):

1. Reads the baseline (model_version + baseline RMSE).
2. `rolling_rmse(conn, model_version, window=5)`: joins `predictions_log`
   (latest prediction per player/gw, for this model_version only) against
   `player_gameweek_stats.total_points` over the trailing `window`
   gameweeks that actually have both. **RMSE, not MAE** - same reasoning as
   `tune.py`'s corrected objective: this system needs to spot big-haul/
   captaincy performances, which RMSE is sensitive to (it penalises large
   errors quadratically) and MAE is not. `window` is a parameter, default 5.
3. `is_degraded` = rolling RMSE > baseline RMSE x (1 + 10%) (both the 10%
   threshold and the window are keyword arguments to `evaluate_and_log`,
   not constants baked into the logic).
4. Looks up how many prior *consecutive* runs (for this same model_version)
   were also degraded, and adds one if this run is too - a single noisy
   gameweek doesn't trigger anything; **2 consecutive degraded runs** does
   (also configurable). A non-degraded run resets the streak to 0.
5. Logs everything to `monitoring_runs` (queryable - not just a log line):
   the rolling RMSE, the baseline, the threshold, whether this run is
   degraded, the current streak, and whether a retrain fired.
6. If the streak reaches the trigger, calls `retrain.run_retrain(...)`
   directly - this is a real, wired-up call, not a TODO.

Every decision is also logged loudly (`log.warning`/`log.info`) at run time.
That's deliberate: a retrain is an internal engineering action on the
model, not a decision about a real transfer, so it does **not** go through
the recommend-and-confirm human-approval flow ARCHITECTURE.md defines for
actual squad recommendations - but "doesn't need approval" still means
"must be visible", not silent. The `monitoring_runs`/`retrain_log` tables
plus the log lines are that visibility; no separate notification channel
(Slack/email) was built for this - out of scope here, and easy to add later
by having a CronJob wrapper forward these log lines/rows if that's ever
wanted.

## Retrain trigger (`agents/stats/retrain.py`)

Run manually: `python -m agents.stats.retrain`. Fired automatically by
`monitor.py` above.

1. **Builds the full accumulated dataset**: the frozen vaastav archive
   (`agents.stats.data.load_gameweeks()`) + the current season's silver
   data to date (`ingestion.silver.load_gameweek_frame()`), concatenated.
   Both already produce the shared "gameweek frame" shape
   `features/engineering.py` expects - see `retrain.py`'s module docstring
   for why this, rather than `data/reconcile_archive.py`'s DB-schema
   mapping, is the actual reconciliation mechanism used here (short version:
   the join keys feature engineering needs - team/opponent NAME strings -
   don't have the cross-season ID-numbering problem that adapter solves).
2. **Season-aware split**: both seasons number their own gameweeks 1..38,
   so `features/engineering.py` now takes an optional `season` column and
   groups every rolling computation on `(season, name)` / `(season, team,
   GW)` instead of just `name`/`(team, GW)` - a player's GW1 this season is
   never treated as following last season's GW38. The archive sits
   entirely in train (it's all in the past); the validation holdout is the
   current season's **last 5 gameweeks** (a fixed *count*, not a fixed
   absolute gameweek number like `train.py`'s `VALID_FROM_GW=30` - a retrain
   can fire at any point in the season, so the holdout has to scale with
   however much current-season data actually exists). Below 8 usable
   current-season gameweeks, there's too little data for a meaningful
   holdout and the retrain is skipped (logged, not silently dropped).
3. **Refits XGBoost** on the combined training pool with the same
   hand-picked defaults `train.py` uses (`XGB_UNTUNED_PARAMS`), then
   **re-tunes it by calling `tune.py`'s actual `tune_xgboost()` function
   unmodified** - same RMSE objective, same forward-chaining chronological
   folds, same Spearman tracking. (A small internal detail: `tune_xgboost`
   assumes one globally-monotonic `GW` column for its fold-building: since
   the archive and the current season both number gameweeks 1..38,
   `retrain.py` offsets the current season's GW values by the archive's max
   GW - for this fold-building input only - so the two seasons form one
   clean chronological sequence without touching `tune.py` at all.)
4. **Applies the exact same selection rule as `tune.py`**
   (`tune._passes_selection_rule`): the tuned refit only stays deployed if
   it doesn't regress RMSE or Spearman (within the same small stated
   tolerances) versus the freshly-refit untuned baseline.
5. Per ARCHITECTURE.md's model-lifecycle note, a routine retrain does
   **not** repeat the full five-model comparison (naive/Poisson/RF/
   XGBoost/MLP) - only XGBoost is refit and re-tuned. The full comparison
   is for season-boundary checkpoints, not every retrain.

**A retrain always deploys *something* new** when it actually runs (skips
aside): re-fitting on more/fresher data is the point of retraining in the
first place. The selection rule decides only whether the *re-tuned*
hyperparameters are kept on top of that refresh, or whether the plain
untuned refit is deployed instead - not whether to retrain at all. Refusing
to update the model just because tuning didn't help would leave the exact
staleness problem the trigger fired to fix still in place.

Every retrain attempt - deployed, or skipped for lack of data - gets a row
in `retrain_log`: when, why (the specific threshold/streak that fired), the
old model's version + metrics, the new model's version + metrics (or
`NULL` if skipped), and whether it was actually deployed.

## Configuration reference

| knob | where | default |
|---|---|---|
| rolling window (N gameweeks) | `monitor.evaluate_and_log(window=...)` | 5 |
| degrade threshold | `monitor.evaluate_and_log(degrade_threshold_pct=...)` | 10% over baseline |
| consecutive-degraded trigger | `monitor.evaluate_and_log(retrain_consecutive=...)` | 2 |
| retrain holdout size | `retrain.CURRENT_SEASON_HOLDOUT_GWS` | 5 GWs |
| minimum current-season data for a retrain | `retrain.MIN_CURRENT_SEASON_GWS_FOR_HOLDOUT` | 8 GWs |
| tuned-vs-untuned selection tolerance | `tune.RMSE_TOLERANCE_PCT` / `tune.SPEARMAN_TOLERANCE` | 0.5% / 0.005 |
