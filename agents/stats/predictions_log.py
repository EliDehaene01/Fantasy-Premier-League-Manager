"""The predictions log: every real prediction the Stats agent's `/argue`
endpoint makes gets a row here (`predictions_log` in Postgres, schema owned
by ``ingestion/db.py`` per CLAUDE.md's "Data store" convention), tagged with
the model version that produced it.

This is what makes the monitoring job (``monitor.py``) possible: after a
gameweek finishes, its actual points can be joined against what was
predicted for it, without having to re-derive "what did we predict" from
anywhere else. The model-version tag is what lets that join stay honest
across a retrain - see ``train.py::_model_version`` and
``docs/monitoring.md``.
"""

from __future__ import annotations

from datetime import datetime, timezone


def log_predictions(conn, gameweek: int, picks: list, model_version: str) -> int:
    """Insert one row per pick (``model_runtime.Pick`` instances). Never
    raises on its own - see the try/except around the call site in
    service.py: a logging hiccup must not cost a live scoring response, the
    same principle that already governs the LLM call in reasoning.py.

    Returns the number of rows written.
    """
    if not picks:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = [(p.player_id, gameweek, p.predicted_points, model_version, now) for p in picks]
    conn.executemany(
        """
        INSERT INTO predictions_log (player_id, gw, predicted_points, model_version, logged_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        rows,
    )
    conn.commit()
    return len(rows)
