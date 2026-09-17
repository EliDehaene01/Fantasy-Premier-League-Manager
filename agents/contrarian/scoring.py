"""Deterministic differential-picking score for the Contrarian agent.

WHY THIS READS THE PREDICTIONS LOG INSTEAD OF CALLING THE STATS SERVICE
--------------------------------------------------------------------------
Contrarian needs a "quality" signal to weigh against ownership, and the
Stats agent's trained model is the obvious source of that signal. But
having Contrarian call the Stats service live, over HTTP, at request time,
would create a runtime dependency BETWEEN two specialist services - if
Stats is slow, down, or mid-redeploy, Contrarian would fail or hang along
with it, even though its own job (differential picking) has nothing to do
with Stats being reachable at that exact moment. Each specialist agent is
meant to be its own independently deployable/scalable service (see
CLAUDE.md, ARCHITECTURE.md section 8's per-service Deployments); a live
cross-service call on every `/argue` request breaks that independence and
turns a chain of five specialists into a single point of failure.

Instead, Contrarian reads the SAME `predicted_points` Stats already wrote
to `predictions_log` (agents/stats/predictions_log.py) the last time it ran
for this gameweek. This is a genuine architectural choice, not a shortcut
to avoid writing an HTTP client: Contrarian can run, be tested, and be
redeployed with zero knowledge of whether the Stats service is even up, at
the cost of using whatever prediction Stats last logged rather than a
guaranteed-fresh number computed on demand - an acceptable trade since both
agents run from the same weekly ingestion cycle in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pick:
    player_id: int
    name: str | None
    predicted_points: float
    conviction: float
    factors: list[str] = field(default_factory=list)


def _latest_predictions(conn, gameweek: int, player_ids: list[int]) -> dict[int, float]:
    """The latest logged predicted_points per player for this gameweek. A
    player can be scored more than once if /argue is re-run mid-week (see
    docs/monitoring.md), so this takes the most recent logged row per
    player, not an average of every logged attempt.
    """
    if not player_ids:
        return {}
    rows = conn.execute(
        """
        SELECT DISTINCT ON (player_id) player_id, predicted_points
        FROM predictions_log
        WHERE gw = %s AND player_id = ANY(%s)
        ORDER BY player_id, logged_at DESC
        """,
        (gameweek, player_ids),
    ).fetchall()
    return {r["player_id"]: float(r["predicted_points"]) for r in rows}


def _latest_ownership(conn, gameweek: int, player_ids: list[int]) -> dict[int, float]:
    """Most recent known ownership % (`selected`) at or before this
    gameweek. `selected` can be NULL for some rows (see ingestion/db.py's
    schema comment on player_gameweek_stats - it's best-effort), so those
    rows are skipped rather than treated as 0% owned.
    """
    if not player_ids:
        return {}
    rows = conn.execute(
        """
        SELECT DISTINCT ON (player_id) player_id, selected
        FROM player_gameweek_stats
        WHERE player_id = ANY(%s) AND gw <= %s AND selected IS NOT NULL
        ORDER BY player_id, gw DESC
        """,
        (player_ids, gameweek),
    ).fetchall()
    return {r["player_id"]: float(r["selected"]) for r in rows}


def rank_players(conn, gameweek: int, players: list, top_k: int) -> list[Pick]:
    """Ranks by the GAP between predicted quality and ownership - both
    normalized to a comparable 0-1 scale so neither signal dominates purely
    by unit size (predicted points might range 0-15, ownership 0-100). A
    player only ranks highly by being BOTH a real predicted performer AND
    genuinely low-owned - "genuinely differentiated picks, not just any
    low-owned player", not a pure ownership sort.

    Players Stats hasn't logged a prediction for this gameweek are skipped
    entirely - no quality signal means no defensible rank, and treating a
    missing prediction as 0 points would be a fabricated number, not "no
    data" (the same reasoning behind `predicted_points` being Optional on
    the shared contract - see shared/contracts.py).
    """
    if not players:
        return []

    player_ids = [p.player_id for p in players]
    predicted = _latest_predictions(conn, gameweek, player_ids)
    ownership = _latest_ownership(conn, gameweek, player_ids)

    eligible = [p for p in players if p.player_id in predicted]
    if not eligible:
        return []

    max_points = max(predicted[p.player_id] for p in eligible)
    max_points = max(max_points, 1e-9)  # guard divide-by-zero if every predicted value is <= 0

    scored = []
    for p in eligible:
        pts = predicted[p.player_id]
        own_pct = ownership.get(p.player_id, 0.0)
        quality_norm = max(pts, 0.0) / max_points
        ownership_norm = min(max(own_pct, 0.0), 100.0) / 100.0
        gap = quality_norm - ownership_norm
        factors = [
            f"predicted {pts:.1f} pts this gameweek",
            f"owned by {own_pct:.1f}% of managers",
        ]
        scored.append((p, pts, gap, factors))

    scored.sort(key=lambda t: t[2], reverse=True)
    top_k = min(top_k, len(scored))
    top = scored[:top_k]
    # Anchor is the strongest pick's own gap, floored at 0 - same
    # relative-conviction convention as agents/stats/model_runtime.py.
    anchor = max(top[0][2], 0.0) if top else 0.0

    picks = []
    for p, pts, gap, factors in top:
        conviction = (max(gap, 0.0) / anchor) if anchor > 0 else 0.0
        picks.append(
            Pick(
                player_id=p.player_id,
                name=p.name,
                predicted_points=round(pts, 2),
                conviction=round(min(max(conviction, 0.0), 1.0), 3),
                factors=factors,
            )
        )
    return picks
