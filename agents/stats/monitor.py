"""Monitoring job: after each gameweek's results are confirmed (the weekly
ingestion pipeline already pulled them via ``event/{gw}/live`` - this job
runs AFTER that step, not instead of it), join the predictions log against
actual results, compute a rolling RMSE, compare it to the backtest baseline,
and fire the retrain trigger if degradation has sustained for long enough.

Run:  python -m agents.stats.monitor

See docs/monitoring.md for the full loop (predictions log -> this job ->
retrain.py), including why RMSE (not MAE) is the metric here too - it's the
same reasoning tune.py's objective function documents: this system cares
about spotting big-haul/captaincy performances, which RMSE is sensitive to
and MAE is not.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .data import REPO_ROOT
from .retrain import run_retrain

log = logging.getLogger("stats_monitor")

BASELINE_PATH = REPO_ROOT / "models" / "model_baseline.json"

# --- monitoring thresholds -------------------------------------------------
# All configurable at call time; these are just sane defaults for the CLI.
DEFAULT_WINDOW_GWS = 5        # "trailing N gameweeks" - task asks for this to be configurable, default ~5
DEGRADE_THRESHOLD_PCT = 10.0  # rolling RMSE this much worse than the backtest baseline counts as degraded
RETRAIN_CONSECUTIVE_GWS = 2   # ... but only trigger a retrain after this many CONSECUTIVE degraded runs


def read_baseline() -> dict:
    """The backtest baseline: which model_version is deployed and what its
    held-out RMSE was, per train.py::save_model. Explicit, versioned config
    - not a number hardcoded into this file - so updating it is "retrain
    happens, save_model() writes a new one", not a manual edit.
    """
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"{BASELINE_PATH} not found - run `python -m agents.stats.train` (or tune.py) at least "
            "once to establish a deployed model and its baseline before monitoring can compare against it."
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def rolling_rmse(conn, model_version: str, window: int = DEFAULT_WINDOW_GWS) -> dict:
    """RMSE over the trailing ``window`` gameweeks of THIS model_version's
    predictions only (see the module docstring in predictions_log.py and
    train.py::_model_version) - a retrain naturally starts a fresh window
    rather than blending pre- and post-retrain predictions into one number,
    because a prediction from a different model_version is never included
    here at all, by construction.

    Uses the latest logged prediction per (player, gw) if a player was ever
    scored more than once for the same gameweek (e.g. re-run mid-week as
    fresh data came in) - a real, if rare, situation `predictions_log` (an
    append-only log) allows.

    Returns {"rmse": float | None, "n_gameweeks_used": int, "gameweeks_used":
    list[int], "n_predictions": int}. ``rmse`` is None if there is nothing
    to compare yet (no finished gameweeks with logged predictions for this
    model_version).
    """
    rows = conn.execute(
        """
        SELECT DISTINCT ON (pl.player_id, pl.gw)
            pl.gw, pl.predicted_points, pgs.total_points AS actual_points
        FROM predictions_log pl
        JOIN player_gameweek_stats pgs
            ON pgs.player_id = pl.player_id AND pgs.gw = pl.gw
        WHERE pl.model_version = %s
        ORDER BY pl.player_id, pl.gw, pl.logged_at DESC
        """,
        (model_version,),
    ).fetchall()

    if not rows:
        return {"rmse": None, "n_gameweeks_used": 0, "gameweeks_used": [], "n_predictions": 0}

    all_gws = sorted({r["gw"] for r in rows}, reverse=True)
    gameweeks_used = sorted(all_gws[:window])
    used = [r for r in rows if r["gw"] in gameweeks_used]

    sq_errors = [(r["predicted_points"] - r["actual_points"]) ** 2 for r in used]
    rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5

    return {
        "rmse": rmse,
        "n_gameweeks_used": len(gameweeks_used),
        "gameweeks_used": gameweeks_used,
        "n_predictions": len(used),
    }


def _previous_consecutive_degraded(conn, model_version: str) -> int:
    """How many consecutive prior monitoring runs (for THIS model_version
    only - a retrain resets the streak, deliberately) were already flagged
    degraded, most recent first. 0 if the most recent run wasn't degraded,
    or if there's no prior run at all.
    """
    row = conn.execute(
        """
        SELECT is_degraded, consecutive_degraded_gws FROM monitoring_runs
        WHERE model_version = %s
        ORDER BY evaluated_through_gw DESC, id DESC
        LIMIT 1
        """,
        (model_version,),
    ).fetchone()
    if row is None or not row["is_degraded"]:
        return 0
    return row["consecutive_degraded_gws"]


def evaluate_and_log(
    conn,
    *,
    window: int = DEFAULT_WINDOW_GWS,
    degrade_threshold_pct: float = DEGRADE_THRESHOLD_PCT,
    retrain_consecutive: int = RETRAIN_CONSECUTIVE_GWS,
    auto_retrain: bool = True,
) -> dict:
    """Run one monitoring cycle: rolling RMSE vs baseline, log the result,
    and fire a retrain if degradation has sustained for `retrain_consecutive`
    runs in a row. Returns the full result dict (also what gets logged), so
    a caller (the CLI below, or a future CronJob wrapper) can print/notify
    it without re-querying anything.
    """
    baseline = read_baseline()
    model_version = baseline["model_version"]
    baseline_rmse = baseline["holdout_metrics"]["rmse"]

    roll = rolling_rmse(conn, model_version, window=window)
    is_degraded = roll["rmse"] is not None and roll["rmse"] > baseline_rmse * (1 + degrade_threshold_pct / 100)

    prev_consecutive = _previous_consecutive_degraded(conn, model_version)
    consecutive = (prev_consecutive + 1) if is_degraded else 0

    evaluated_through_gw = max(roll["gameweeks_used"]) if roll["gameweeks_used"] else 0
    retrain_triggered = is_degraded and consecutive >= retrain_consecutive

    conn.execute(
        """
        INSERT INTO monitoring_runs (evaluated_through_gw, model_version, window_gws, gws_used,
                                      rolling_rmse, baseline_rmse, degrade_threshold_pct, is_degraded,
                                      consecutive_degraded_gws, retrain_triggered, run_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            evaluated_through_gw, model_version, window, roll["n_gameweeks_used"],
            roll["rmse"], baseline_rmse, degrade_threshold_pct, is_degraded,
            consecutive, retrain_triggered, datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

    result = {
        "model_version": model_version,
        "baseline_rmse": baseline_rmse,
        "rolling_rmse": roll["rmse"],
        "window_gws": window,
        "gws_used": roll["gameweeks_used"],
        "is_degraded": is_degraded,
        "consecutive_degraded_gws": consecutive,
        "retrain_triggered": retrain_triggered,
    }

    # "Visible/notifiable, not silent" (task's own words) - this is an
    # internal engineering action, not a decision about a real team, so it
    # doesn't go through the recommend-and-confirm human-approval flow
    # ARCHITECTURE.md defines for actual transfer recommendations; a clear,
    # loud log line (plus the monitoring_runs/retrain_log rows, which are
    # queryable) is the right amount of ceremony for this.
    if roll["rmse"] is None:
        log.warning("[monitor] no predictions yet for model_version=%s - nothing to evaluate", model_version)
    elif is_degraded:
        log.warning(
            "[monitor] DEGRADED: rolling RMSE %.3f over GWs %s vs baseline %.3f (+%.1f%% threshold), "
            "%d consecutive degraded run(s)",
            roll["rmse"], roll["gameweeks_used"], baseline_rmse, degrade_threshold_pct, consecutive,
        )
    else:
        log.info(
            "[monitor] OK: rolling RMSE %.3f over GWs %s vs baseline %.3f",
            roll["rmse"], roll["gameweeks_used"], baseline_rmse,
        )

    if retrain_triggered:
        reason = (
            f"rolling RMSE {roll['rmse']:.3f} exceeded baseline {baseline_rmse:.3f} by more than "
            f"{degrade_threshold_pct:.1f}% for {consecutive} consecutive monitoring runs "
            f"(model_version={model_version}, window={window} GWs, GWs {roll['gameweeks_used']})"
        )
        log.warning("[monitor] RETRAIN TRIGGERED: %s", reason)
        if auto_retrain:
            result["retrain_result"] = run_retrain(reason=reason, conn=conn)
        else:  # pragma: no cover - only used by tests exercising trigger logic without a real retrain
            log.warning("[monitor] auto_retrain=False - trigger fired but retrain was NOT run")

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from ingestion.db import get_connection

    conn = get_connection()
    try:
        result = evaluate_and_log(conn)
        print(json.dumps({k: v for k, v in result.items() if k != "retrain_result"}, indent=2, default=str))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
