# Stats agent

Two parts:

1. **The data-science pipeline** (`eda.py`, `features.py`, `train.py`) - offline
   model building: EDA, feature engineering, model comparison, SHAP. Produces
   `models/stats_model.pkl`.
2. **The FastAPI service** (`service.py`, `model_runtime.py`, `reasoning.py`) -
   loads that model and answers `POST /argue` in the gameweek debate.

---

## The service

```bash
pip install -r agents/stats/requirements.txt
uvicorn agents.stats.service:app --port 8001      # from the repo root
```

### `POST /argue`

Request:

```json
{
  "gameweek": 34,
  "top_k": 5,
  "players": [
    {"player_id": 351, "name": "Salah", "position": "MID",
     "features": {"pts_roll5": 6.2, "minutes_roll3": 88.0, "fixture_adj_xp": 5.1, "...": 0.0}}
  ]
}
```

`features` is the engineered feature vector for that player as of the gameweek
deadline (the 38 columns in `bundle["features"]`; missing ones default to 0.0).
Computing them from raw FPL data is the ingestion service's job - this service
stays thin and just scores what it's handed.

Response is the shared specialist-agent contract (see `ARCHITECTURE.md` §2):

```json
{
  "agent": "stats",
  "recommendations": [{"player_id": 351, "conviction": 1.0, "predicted_points": 6.4}],
  "reasoning": "Salah is the strongest pick: rising minutes and a higher season points-per-game ..."
}
```

### Why the model call and the LLM call are separate

- `model_runtime.py` owns **every number** - which players, their predicted
  points, the ranking, the conviction. Deterministic, offline, testable. An LLM
  is none of those things and never decides who to recommend.
- `reasoning.py` owns **only the English**. It takes the ranked picks + their
  SHAP factors and asks the Foundry `gpt-5.4-nano` deployment to write 2-4
  sentences. If that call fails (network, rate limit, bad config) the service
  returns the same recommendations with a templated, still-SHAP-grounded
  sentence. The LLM is a rendering layer, not a dependency.

### Config

Credentials come from the existing `.env` variables (no new names):
`MICROSOFT_FOUNDRY_OPENAI_ENDPOINT`, `MICROSOFT_FOUNDRY_KEY`. Optional:
`STATS_AGENT_LLM_MODEL` (deployment name, default `gpt-5.4-nano`),
`STATS_AGENT_DISABLE_LLM=1` (skip the LLM entirely, always use the fallback -
used by the tests and available for backtest runs).

### Tests

```bash
python -m pytest agents/stats/test_stats_agent.py
```

---

## The data-science pipeline

This part is **just the data science** - EDA, feature engineering, model
comparison, SHAP.

## What it produces

| Artifact | Path | What it is |
|---|---|---|
| EDA write-up | `docs/eda_summary.md` | Distributions of form, minutes, price changes, unavailability by position, in plain language |
| Model comparison | `docs/model_comparison.md` | The five-model bake-off, the winner, and which engineered features mattered |
| Metrics (machine-readable) | `docs/model_metrics.json` | MAE / RMSE / Spearman per model |
| Figures | `docs/figures/*.png` | Charts for both write-ups + the SHAP beeswarm |
| **Trained model** | `models/stats_model.pkl` | The winning model + feature list + split definition + metrics (joblib dict) |

## How to run

```bash
pip install -r agents/stats/requirements.txt

# 1. exploratory data analysis  -> docs/eda_summary.md
python -m agents.stats.eda

# 2. features -> chronological split -> 5 models -> SHAP -> save winner
python -m agents.stats.train
```

Run from the repo root. Input is `merged_gws_2025-26.csv` in the repo root
(vaastav/Fantasy-Premier-League format).

## The files

| File | Responsibility |
|---|---|
| `data.py` | Load the raw CSV, collapse double-gameweek rows to one-row-per-player-per-gameweek, build the `opponent_team` id -> name map, aggregate team-level match stats |
| `features.py` | All feature engineering. **Every feature is built from static facts or lagged history** - see the module docstring for the no-lookahead rule |
| `eda.py` | Writes `docs/eda_summary.md` and its figures |
| `train.py` | Chronological split, trains + compares the five models, runs SHAP on the winner, saves `models/stats_model.pkl` and `docs/model_comparison.md` |
| `schemas.py` | Pydantic request/response models - the shared specialist-agent contract |
| `model_runtime.py` | Loads `stats_model.pkl` + SHAP explainer once; scores a pool, ranks it, derives conviction and per-pick SHAP factors (the "numbers") |
| `reasoning.py` | Turns ranked picks + SHAP factors into 2-4 sentences via Foundry `gpt-5.4-nano`, with a deterministic fallback (the "prose") |
| `service.py` | FastAPI app: `POST /argue`, `GET /health`; loads the model in the lifespan handler |
| `test_stats_agent.py` | Contract shape, conviction ordering, empty-pool handling |

## Design decisions worth knowing

- **No lookahead, ever.** Predicting gameweek N uses only gameweeks 1..N-1
  plus pre-deadline facts (price, opponent, home/away). Rolling features are
  `.shift(1)` before the window. The train/validation split is chronological
  for the same reason - a random k-fold leaks the future backward.
- **`xP` is excluded from features.** It is FPL's own expected-points column;
  using it would make this a partial copy of FPL's model instead of an
  independent one.
- **MAE is the primary metric, not RMSE.** ~61% of rows are zeros and the
  score distribution has a long haul-shaped tail; RMSE just tracks the tail.
  Spearman (player ordering) is reported alongside because that is what the
  squad solver actually consumes.
- **The Poisson family is the statistically right call** for a non-negative,
  zero-inflated, count-like target - even though the tree models ultimately
  win on accuracy.

## Loading the model

```python
import joblib
bundle = joblib.load("models/stats_model.pkl")
model    = bundle["model"]        # a fitted sklearn / xgboost estimator
features = bundle["features"]     # column order model.predict expects
preds = model.predict(X[features])  # expected points for an upcoming gameweek
```
