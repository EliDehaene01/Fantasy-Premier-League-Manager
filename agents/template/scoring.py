"""Deterministic ownership-safety score for the Template agent.

Uses ownership % and net transfer momentum, both already in silver from
bootstrap-static (`player_gameweek_stats.selected`/`.transfers_balance` -
see ingestion/db.py). No dependency on predicted points from another agent
- this agent's whole point is following the crowd, not evaluating quality
independently, so it deliberately never reads the predictions log the way
Contrarian does (contrast agents/contrarian/scoring.py's module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ownership is the PRIMARY template signal; net transfer momentum is a
# smaller secondary nudge - is the crowd currently buying or fading this
# pick, on top of already being widely owned. Weighted well below 1.0 since
# a stampede of transfers on a still-low-owned player isn't "safe" the way
# high ownership itself is - ownership must dominate the score.
TRANSFER_MOMENTUM_WEIGHT = 0.2


@dataclass
class Pick:
    player_id: int
    name: str | None
    conviction: float
    factors: list[str] = field(default_factory=list)


def _latest_ownership_and_transfers(conn, gameweek: int, player_ids: list[int]) -> dict[int, tuple[float, float]]:
    """Most recent known ownership % and net transfer balance at or before
    this gameweek. Either field can be NULL for some rows (best-effort, see
    ingestion/db.py's schema comment); a row is eligible as long as at
    least one of the two is known, with the missing one treated as neutral
    (0) rather than skipping the player entirely.
    """
    if not player_ids:
        return {}
    rows = conn.execute(
        """
        SELECT DISTINCT ON (player_id) player_id, selected, transfers_balance
        FROM player_gameweek_stats
        WHERE player_id = ANY(%s) AND gw <= %s
          AND (selected IS NOT NULL OR transfers_balance IS NOT NULL)
        ORDER BY player_id, gw DESC
        """,
        (player_ids, gameweek),
    ).fetchall()
    return {
        r["player_id"]: (
            float(r["selected"]) if r["selected"] is not None else 0.0,
            float(r["transfers_balance"]) if r["transfers_balance"] is not None else 0.0,
        )
        for r in rows
    }


def rank_players(conn, gameweek: int, players: list, top_k: int) -> list[Pick]:
    """Ranks by ownership, with net-transfer momentum as a smaller
    secondary nudge - see TRANSFER_MOMENTUM_WEIGHT above. Players with no
    ownership/transfer data at all for this gameweek are skipped (no safety
    signal to rank them on).
    """
    if not players:
        return []

    player_ids = [p.player_id for p in players]
    data = _latest_ownership_and_transfers(conn, gameweek, player_ids)
    eligible = [p for p in players if p.player_id in data]
    if not eligible:
        return []

    max_abs_transfer = max((abs(data[p.player_id][1]) for p in eligible), default=0.0)
    max_abs_transfer = max(max_abs_transfer, 1.0)  # guard divide-by-zero when nobody's transfer balance moved

    scored = []
    for p in eligible:
        own_pct, transfers = data[p.player_id]
        ownership_norm = min(max(own_pct, 0.0), 100.0) / 100.0
        transfer_norm = max(min(transfers / max_abs_transfer, 1.0), -1.0)
        safety_score = ownership_norm + TRANSFER_MOMENTUM_WEIGHT * transfer_norm
        direction = "rising" if transfers > 0 else "falling" if transfers < 0 else "flat"
        factors = [
            f"owned by {own_pct:.1f}% of managers",
            f"net transfers {direction} ({transfers:+.0f} this window)",
        ]
        scored.append((p, safety_score, factors))

    scored.sort(key=lambda t: t[1], reverse=True)
    top_k = min(top_k, len(scored))
    top = scored[:top_k]
    # Anchor is the strongest pick's own score, floored at 0 - same
    # relative-conviction convention as agents/stats/model_runtime.py.
    anchor = max(top[0][1], 0.0) if top else 0.0

    picks = []
    for p, score, factors in top:
        conviction = (max(score, 0.0) / anchor) if anchor > 0 else 0.0
        picks.append(
            Pick(
                player_id=p.player_id,
                name=p.name,
                conviction=round(min(max(conviction, 0.0), 1.0), 3),
                factors=factors,
            )
        )
    return picks
