"""Deterministic fixture-favorability scoring for the Fixtures agent.

No ML model, no LLM, no dependency on any other agent's output - purely a
read of fixture difficulty (FDR, 1=easiest to 5=hardest) and fixture COUNT
per gameweek (0 fixtures = a blank gameweek, 2+ = a double) already sitting
in silver's `fixtures` table (see ingestion/db.py), for the next few
gameweeks starting at the one being argued for.

A double gameweek needs no special-cased bonus: summing each fixture's own
favorability within a gameweek already gives a team with two matches double
the score of a team with one similarly-easy match, which is exactly the
real extra scoring opportunity a double gameweek represents. A blank
gameweek likewise needs no special-cased penalty - it naturally contributes
zero to that gameweek's term, since there is nothing to sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# How many gameweeks ahead to look, and how much each one counts - the very
# next gameweek weighted heaviest, decaying for gameweeks further out, per
# the task's "weighted more heavily toward the immediate next gameweek than
# further-out ones". Three gameweeks is enough to catch an upcoming
# blank/double without diluting the signal too far into an uncertain future.
WINDOW_GWS = 3
GW_WEIGHTS = (1.0, 0.5, 0.25)


@dataclass
class Pick:
    player_id: int
    name: str | None
    position: str | None
    conviction: float
    factors: list[str] = field(default_factory=list)


def team_ids_for_players(conn, player_ids: list[int]) -> dict[int, int]:
    """Not underscore-prefixed: reused by the Chips agent
    (agents/chips/scoring.py) to find which teams our own squad's players
    belong to, rather than reimplementing the same lookup."""
    if not player_ids:
        return {}
    rows = conn.execute(
        "SELECT player_id, team_id FROM players WHERE player_id = ANY(%s)", (player_ids,)
    ).fetchall()
    return {r["player_id"]: r["team_id"] for r in rows}


def fixtures_by_team(conn, team_ids: list[int], from_gw: int) -> dict[int, dict[int, list[tuple[bool, int]]]]:
    """Returns ``{team_id: {gw: [(is_home, difficulty), ...]}}`` for the
    window. Every requested team_id gets an entry for every gameweek in the
    window (an empty list where there's no fixture row) - a team present in
    ``players`` but genuinely fixtureless in the window (a real blank, not a
    lookup failure) has to be distinguishable from "this player's team_id
    couldn't be resolved at all", which ``rank_players`` below handles
    separately.

    Not underscore-prefixed: this is the exact per-team-per-gameweek
    fixture-count data the Chips agent needs to detect blank/double
    gameweeks for our own squad (agents/chips/scoring.py) - reused
    directly rather than reimplementing the same fixture-counting query.
    """
    if not team_ids:
        return {}
    gws = list(range(from_gw, from_gw + WINDOW_GWS))
    rows = conn.execute(
        """
        SELECT gw, team_h, team_a, team_h_difficulty, team_a_difficulty
        FROM fixtures
        WHERE gw = ANY(%s) AND (team_h = ANY(%s) OR team_a = ANY(%s))
        """,
        (gws, team_ids, team_ids),
    ).fetchall()

    by_team: dict[int, dict[int, list[tuple[bool, int]]]] = {t: {gw: [] for gw in gws} for t in team_ids}
    for r in rows:
        if r["team_h"] in by_team:
            by_team[r["team_h"]][r["gw"]].append((True, r["team_h_difficulty"]))
        if r["team_a"] in by_team:
            by_team[r["team_a"]][r["gw"]].append((False, r["team_a_difficulty"]))
    return by_team


def favorability(team_fixtures: dict[int, list[tuple[bool, int]]], from_gw: int) -> tuple[float, list[str]]:
    """Weighted favorability score + human-readable factor phrases for one
    team's fixture window. Higher is better: favorability per fixture is
    ``6 - difficulty`` (difficulty 1..5 maps to favorability 5..1).

    Not underscore-prefixed: reused by the Chips agent's Wildcard-timing
    check (agents/chips/scoring.py) to judge whether our squad's near-term
    fixture run is currently poor enough to justify an overhaul.
    """
    total = 0.0
    factors: list[str] = []
    for i, weight in enumerate(GW_WEIGHTS):
        gw = from_gw + i
        fixtures_this_gw = team_fixtures.get(gw, [])
        if not fixtures_this_gw:
            factors.append(f"GW{gw}: blank (no fixture)")
            continue
        gw_score = sum(6 - difficulty for _, difficulty in fixtures_this_gw if difficulty is not None)
        total += weight * gw_score
        if len(fixtures_this_gw) > 1:
            factors.append(f"GW{gw}: double gameweek ({len(fixtures_this_gw)} fixtures)")
        else:
            is_home, difficulty = fixtures_this_gw[0]
            venue = "home" if is_home else "away"
            factors.append(f"GW{gw}: {venue}, FDR {difficulty}")
    return total, factors


def rank_players(conn, gameweek: int, players: list, top_k: int) -> list[Pick]:
    """``players`` is the ArgueRequest's ``PlayerEntry`` list. Ranks by
    fixture favorability over the next ``WINDOW_GWS`` gameweeks starting at
    ``gameweek``, weighted toward the immediate next one.
    """
    if not players:
        return []

    player_ids = [p.player_id for p in players]
    team_of = team_ids_for_players(conn, player_ids)
    team_ids = sorted({t for t in team_of.values() if t is not None})
    fixtures_by_team_map = fixtures_by_team(conn, team_ids, gameweek)

    scored = []
    for p in players:
        team_id = team_of.get(p.player_id)
        if team_id is None or team_id not in fixtures_by_team_map:
            scored.append((p, 0.0, ["no fixture data available for this player"]))
            continue
        score, factors = favorability(fixtures_by_team_map[team_id], gameweek)
        scored.append((p, score, factors))

    scored.sort(key=lambda t: t[1], reverse=True)
    top_k = min(top_k, len(scored))
    top = scored[:top_k]
    # Anchor is the strongest pick's own score, floored at 0 - mirrors
    # agents/stats/model_runtime.py's conviction calculation exactly, so
    # "conviction" means the same relative thing across every specialist.
    anchor = max(top[0][1], 0.0) if top else 0.0

    picks = []
    for p, score, factors in top:
        conviction = (max(score, 0.0) / anchor) if anchor > 0 else 0.0
        picks.append(
            Pick(
                player_id=p.player_id,
                name=p.name,
                position=p.position,
                conviction=round(min(max(conviction, 0.0), 1.0), 3),
                factors=factors,
            )
        )
    return picks
