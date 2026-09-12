"""Retrain trigger: re-fit the Stats model on the FULL accumulated dataset
(the frozen vaastav archive + the current season's silver data to date),
re-tune it with the same corrected (RMSE-based) objective as tune.py, and
apply the same selection rule already in place - reusing tune.py's actual
functions, not rebuilding them.

Called automatically by monitor.py when sustained degradation fires the
trigger; also runnable directly (``python -m agents.stats.retrain``) for a
manual/on-demand retrain.

ARCHITECTURE.md's model-lifecycle section is explicit that a routine retrain
does NOT need to repeat the full five-model comparison (naive/Poisson/RF/
XGBoost/MLP) - "a retuned XGBoost retrain is the normal case; the full
comparison is worth repeating at natural checkpoints (e.g. season
boundaries)". So this module only re-fits and re-tunes XGBoost, the same way
tune.py already does, just against a bigger, combined dataset.

WHY THIS DOESN'T CALL data/reconcile_archive.py
------------------------------------------------
That module maps the archive's columns onto silver's POSTGRES TABLE schema
specifically (player_id/team_id/opponent_team_id as DB foreign keys) - a
real, separate need (e.g. loading archive rows into the actual silver
tables), but not what's needed here. What THIS module needs is the shared
"gameweek frame" contract features/engineering.py already defines, which
BOTH `agents.stats.data.load_gameweeks()` (archive) and
`ingestion.silver.load_gameweek_frame()` (current season) already produce
correctly and independently - that shared contract, not reconcile_archive's
DB-schema mapping, is the actual reconciliation mechanism a feature build
needs. Notably, the join keys features/engineering.py needs (team and
opponent NAME strings) don't have reconcile_archive's stated problem
(team/fixture ID numbering isn't stable across seasons) at all - club names
are. Concatenating the two "gameweek frame" outputs directly is therefore
both simpler and more correct than routing through the DB-schema adapter.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features.engineering import build_features_from_frame
from ingestion import db as ingestion_db
from ingestion.silver import load_gameweek_frame

from . import data
from . import train as train_module
from . import tune as tune_module

RANDOM_STATE = train_module.RANDOM_STATE
WARMUP_LAST_GW = train_module.WARMUP_LAST_GW
BASELINE_PATH = train_module.MODELS / "model_baseline.json"

# How many of the CURRENT season's most-recent gameweeks to hold out as the
# validation set. Unlike train.py's fixed VALID_FROM_GW=30 (which assumes a
# nearly-complete single archive season), a retrain can fire at ANY point in
# the current season - so the holdout is a fixed COUNT of trailing
# gameweeks, not a fixed absolute GW number, and scales with however much
# current-season data actually exists.
CURRENT_SEASON_HOLDOUT_GWS = 5
# Below this many usable current-season gameweeks, there isn't enough data
# for a meaningful holdout - skip the retrain rather than force a fit on (or
# score against) too few rows to mean anything.
MIN_CURRENT_SEASON_GWS_FOR_HOLDOUT = 8


def build_combined_training_frame(conn) -> pd.DataFrame:
    """The frozen archive season + the current season's silver data to date,
    concatenated as "gameweek frames" (see the module docstring for why this
    - not data/reconcile_archive.py - is the reconciliation mechanism used).
    """
    archive_frame = data.load_gameweeks()
    current_frame = load_gameweek_frame(conn)
    return pd.concat([archive_frame, current_frame], ignore_index=True, sort=False)


def _season_aware_split(frame: pd.DataFrame, feats: list[str], target: str):
    """Chronological split for a MULTI-season frame: warm-up dropped from
    the start of EACH season independently (GW numbering restarts every
    season, so ``GW > WARMUP_LAST_GW`` already does the right thing without
    needing to know about season boundaries); the archive season sits
    ENTIRELY in train (it's all in the past); the current season's last
    ``CURRENT_SEASON_HOLDOUT_GWS`` gameweeks are the validation holdout.

    Returns ``(frame, train, valid, holdout_gws)``; ``holdout_gws`` is empty
    if there isn't enough current-season data yet for a meaningful holdout -
    callers must check for that before proceeding.
    """
    frame = frame[frame["GW"] > WARMUP_LAST_GW].reset_index(drop=True)
    current_gws = sorted(frame.loc[frame["season"] == "current", "GW"].unique().tolist())

    if len(current_gws) < MIN_CURRENT_SEASON_GWS_FOR_HOLDOUT:
        return frame, None, None, set()

    holdout_gws = set(current_gws[-CURRENT_SEASON_HOLDOUT_GWS:])
    is_holdout = (frame["season"] == "current") & frame["GW"].isin(holdout_gws)
    valid = frame[is_holdout].reset_index(drop=True)
    train = frame[~is_holdout].reset_index(drop=True)

    # tune.py's fold-builder (reused unmodified below) assumes "GW" is
    # monotonically increasing across the WHOLE training pool - true for a
    # single season, not true here (both seasons number their own
    # gameweeks 1..38). The archive is, in reality, entirely BEFORE the
    # current season, so offsetting the current season's GW by the
    # archive's own max GW gives one clean chronological sequence. This is
    # ONLY applied to `train` (tune_xgboost's fold-building + fitting
    # input) - `valid` keeps its real current-season GW numbers, since
    # that's what explain_winner()'s worked-example labels show.
    archive_max_gw = int(frame.loc[frame["season"] != "current", "GW"].max() or 0)
    train = train.copy()
    train["GW"] = train["GW"] + np.where(train["season"] == "current", archive_max_gw, 0)

    return frame, train, valid, holdout_gws


def _read_baseline_safe() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _log_retrain_event(conn, reason: str, old: dict | None, new: dict | None, deployed: bool, notes: str) -> None:
    conn.execute(
        """
        INSERT INTO retrain_log (triggered_at, trigger_reason, old_model_version, old_metrics,
                                  new_model_version, new_metrics, deployed, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            reason,
            old["model_version"] if old else None,
            json.dumps(old["holdout_metrics"]) if old else None,
            new["model_version"] if new else None,
            json.dumps(new["holdout_metrics"]) if new else None,
            deployed,
            notes,
        ),
    )
    conn.commit()


def run_retrain(reason: str, conn=None) -> dict:
    """Re-fit + re-tune XGBoost on the full accumulated dataset, apply the
    existing tuned-vs-untuned selection rule, and deploy whichever model
    wins - ALWAYS deploying something when a retrain actually runs (a fresh
    fit on more/newer data is itself the point of retraining; the selection
    rule decides only whether the re-tuned hyperparameters are kept on top
    of that refresh, or the plain untuned refit is deployed instead - see
    "Retrain history" in docs/monitoring.md for why that's the right
    reading of "only replace if it doesn't regress").

    Logs a `retrain_log` row unconditionally, including when skipped (not
    enough current-season data yet) - a skip is a reportable outcome too,
    not silence.
    """
    own_conn = conn is None
    conn = conn or ingestion_db.get_connection()
    try:
        combined = build_combined_training_frame(conn)
        frame_raw, feats, target = build_features_from_frame(combined)
        frame, train, valid, holdout_gws = _season_aware_split(frame_raw, feats, target)

        if not holdout_gws:
            n_current = int(frame.loc[frame["season"] == "current", "GW"].nunique())
            msg = (
                f"not enough current-season data yet for a retrain holdout "
                f"({n_current} usable GWs, need >= {MIN_CURRENT_SEASON_GWS_FOR_HOLDOUT}) - skipped"
            )
            print(f"[retrain] {msg}")
            _log_retrain_event(conn, reason, old=_read_baseline_safe(), new=None, deployed=False, notes=msg)
            return {"skipped": True, "reason": msg}

        Xtr = train[feats].to_numpy(dtype=float)
        ytr_nonneg = np.clip(train[target].to_numpy(dtype=float), 0, None)
        Xva = valid[feats].to_numpy(dtype=float)
        yva = valid[target].to_numpy(dtype=float)
        played_va = (valid["minutes"] > 0).to_numpy()

        naive_pred = (valid["pts_roll3"].to_numpy() + valid["pts_roll5"].to_numpy()) / 2.0
        naive_result = train_module._score("Naive (rolling 3/5 avg)", yva, naive_pred, played_va)

        untuned_model = XGBRegressor(
            **train_module.XGB_UNTUNED_PARAMS, objective="count:poisson",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        untuned_model.fit(Xtr, ytr_nonneg)
        untuned_result = train_module._score("XGBoost", yva, untuned_model.predict(Xva), played_va)
        untuned_result.beats_naive = untuned_result.mae < naive_result.mae - 1e-9

        print(f"[retrain] triggered: {reason}")
        print(
            f"[retrain] untuned XGBoost refit on combined dataset "
            f"({len(train):,} train rows, holdout GWs {sorted(holdout_gws)}): "
            f"MAE={untuned_result.mae:.3f} RMSE={untuned_result.rmse:.3f} "
            f"spearman={untuned_result.spearman:.3f}"
        )

        ctx = {
            "frame": frame, "train": train, "valid": valid,
            "features": feats, "target": target,
            "results": [naive_result, untuned_result],
            "fitted": {"XGBoost": untuned_model},
            "winner": untuned_result, "winner_model": untuned_model,
            "naive_mae": naive_result.mae,
            "X_valid": Xva, "y_valid": yva,
            "split_description": {
                "method": (
                    "season-aware chronological: full archive season in train; "
                    f"current season's last {len(holdout_gws)} GWs held out"
                ),
                "holdout_gws": sorted(holdout_gws),
                "warmup_last_gw": WARMUP_LAST_GW,
                "train_rows": len(train),
            },
        }

        # Reuses tune.py's ACTUAL tuning code unmodified - same RMSE
        # objective, same chronological forward-chaining folds, same
        # Spearman tracking. See _season_aware_split's docstring for why
        # `train`'s GW column is pre-remapped so tune.py's fold-builder
        # (which assumes one globally-monotonic GW column) works correctly
        # against this two-season combined frame without any changes to
        # tune.py itself.
        tune_out = tune_module.tune_xgboost(ctx)
        tuned = tune_out["tuned_result"]
        print(
            f"[retrain] tuned XGBoost: MAE={tuned.mae:.3f} RMSE={tuned.rmse:.3f} "
            f"spearman={tuned.spearman:.3f}"
        )

        old_baseline = _read_baseline_safe()
        passed, sel_reason = tune_module._passes_selection_rule(tuned, untuned_result)
        print(f"[retrain] selection rule: {'PASSED' if passed else 'FAILED'} - {sel_reason}")

        if passed:
            ctx["winner_model"] = tune_out["tuned_model"]
            ctx["winner"] = tuned
        # else: ctx["winner"]/["winner_model"] stay the untuned refit set above -
        # still a real, deployed improvement (more/fresher data), just not
        # the tuned hyperparameters.

        shap_out = train_module.explain_winner(ctx)
        train_module.save_model(ctx, shap_out)
        new_baseline = _read_baseline_safe()

        deployed_kind = "tuned" if passed else "untuned (re-fit on combined data)"
        print(f"[retrain] models/stats_model.pkl saved as the {deployed_kind} model, "
              f"version {new_baseline['model_version']}")

        _log_retrain_event(conn, reason, old=old_baseline, new=new_baseline, deployed=True, notes=sel_reason)

        return {
            "skipped": False,
            "old_baseline": old_baseline,
            "new_baseline": new_baseline,
            "selection_passed": passed,
            "selection_reason": sel_reason,
            "deployed_kind": deployed_kind,
        }
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    run_retrain(reason="manual invocation (python -m agents.stats.retrain)")
