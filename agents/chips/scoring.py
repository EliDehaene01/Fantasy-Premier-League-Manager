"""Deterministic chip-timing logic for the Chips agent.

Reasons about the FIXTURE CALENDAR SHAPE for OUR OWN squad, not individual
player quality - reuses agents/fixtures/scoring.py's fixture-lookup
functions (team_ids_for_players, fixtures_by_team, favorability) rather
than reimplementing fixture-counting, per this agent's own design: the
signal it needs (which of our team's fixtures are doubles/blanks, and how
favorable our near-term run is) is exactly what Fixtures agent already
computes, just applied to our tracked squad instead of an arbitrary
candidate pool.

WHY "NO CHIP THIS WEEK" IS THE EXPECTED DEFAULT, NOT AN EDGE CASE
--------------------------------------------------------------------------
FPL grants each of the four chips only twice a SEASON (38 gameweeks). A
genuinely chip-worthy fixture-calendar shape - several of our players
double- or blank-gameweeking at once, or a standout captain fixture - is a
real but INFREQUENT event; most gameweeks are ordinary. An agent that
suggested a chip every time it ran would burn all four chips in the first
month of the season, which is strictly worse than never suggesting one at
all. So every rule below has a real, calibrated bar to clear (see the
MIN_*/THRESHOLD constants), and the function's actual default return path -
reached whenever nothing clears its bar - is `chip=None` with an honest
explanation of why holding is correct. That default path is not a fallback
apologizing for a missing case; it is this agent doing its job correctly on
an ordinary week.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.fixtures.scoring import GW_WEIGHTS, WINDOW_GWS, favorability, fixtures_by_team, team_ids_for_players
from shared.contracts import ChipType

from .availability import available_chips

# Calibrated against a real ~15-player squad: "several" players affected
# means a meaningful fraction, not a majority (a double/blank gameweek
# rarely touches more than a handful of our specific 15 picks at once).
BENCH_BOOST_MIN_DOUBLE_PLAYERS = 3
FREE_HIT_MIN_BLANK_PLAYERS = 3
# FDR is 1 (easiest) to 5 (hardest) - 2 or lower is a genuinely favorable
# single fixture, not just "not bad".
TRIPLE_CAPTAIN_FAVORABLE_FDR = 2
# favorability() sums (6 - difficulty) per fixture, weighted by GW_WEIGHTS
# (1.0, 0.5, 0.25 - see agents/fixtures/scoring.py), so weights sum to 1.75.
# A team with one average (FDR 3) fixture every week scores 3 * 1.75 = 5.25
# - "neutral". This threshold sits meaningfully below that, so it only
# fires for a run that's genuinely poor (consistently hard fixtures, or a
# blank mixed into the window), not just slightly below average.
WILDCARD_POOR_FAVORABILITY_THRESHOLD = 3.5


@dataclass
class ChipVerdict:
    chip: ChipType | None
    confidence: float
    reasoning_factors: list[str] = field(default_factory=list)


def _our_squad_player_ids(conn, gameweek: int) -> list[int]:
    """Our most recently known squad at or before this gameweek, from
    my_team_state.picks (the raw FPL entry-picks shape: each pick has an
    "element" key holding the player_id - see ingestion/weekly.py's own
    `squad_ids = [p["element"] for p in picks.get("picks", [])]` for the
    same convention).
    """
    row = conn.execute(
        "SELECT picks FROM my_team_state WHERE gw <= %s ORDER BY gw DESC LIMIT 1", (gameweek,)
    ).fetchone()
    if row is None or not row["picks"]:
        return []
    return [p["element"] for p in row["picks"]]


def _best_captain_candidate(conn, gameweek: int, squad_ids: list[int]) -> int | None:
    """Whichever of our squad has the highest latest predicted_points for
    this gameweek, read from the Stats agent's predictions log - same
    service-independence reasoning as agents/contrarian/scoring.py (no live
    call to the Stats service). Returns None if Stats hasn't logged any
    prediction for our squad this gameweek yet.
    """
    if not squad_ids:
        return None
    rows = conn.execute(
        """
        SELECT DISTINCT ON (player_id) player_id, predicted_points
        FROM predictions_log
        WHERE gw = %s AND player_id = ANY(%s)
        ORDER BY player_id, logged_at DESC
        """,
        (gameweek, squad_ids),
    ).fetchall()
    if not rows:
        return None
    return max(rows, key=lambda r: r["predicted_points"])["player_id"]


def _average_squad_favorability(fixtures_map: dict, team_of: dict[int, int], squad_ids: list[int], gameweek: int) -> float | None:
    """Average of favorability() across our squad's DISTINCT teams (not
    per-player - a team fielding several of our players shouldn't count
    multiple times) over the WINDOW_GWS window starting at ``gameweek``.
    """
    scores = []
    seen_teams: set[int] = set()
    for pid in squad_ids:
        team_id = team_of.get(pid)
        if team_id is None or team_id in seen_teams:
            continue
        seen_teams.add(team_id)
        team_fixtures = fixtures_map.get(team_id)
        if team_fixtures is None:
            continue
        score, _ = favorability(team_fixtures, gameweek)
        scores.append(score)
    return (sum(scores) / len(scores)) if scores else None


def evaluate_chip_timing(conn, gameweek: int) -> ChipVerdict:
    """The core decision: is THIS gameweek's fixture calendar shape, for
    OUR squad, worth burning one of our two remaining uses of a chip on?
    Checked in a fixed priority order (bench boost, free hit, triple
    captain, wildcard) - see each block's comment for why. Falls through to
    ``chip=None`` if nothing clears its bar - the expected outcome most
    gameweeks (see this module's docstring).
    """
    squad_ids = _our_squad_player_ids(conn, gameweek)
    if not squad_ids:
        return ChipVerdict(chip=None, confidence=0.5, reasoning_factors=["no known squad for this gameweek - nothing to evaluate"])

    available = available_chips(conn)
    if not available:
        return ChipVerdict(chip=None, confidence=1.0, reasoning_factors=["every chip has already been used its maximum number of times this season"])

    team_of = team_ids_for_players(conn, squad_ids)
    team_ids = sorted({t for t in team_of.values() if t is not None})
    fixtures_map = fixtures_by_team(conn, team_ids, gameweek)

    def _this_gw_fixtures(pid: int) -> list[tuple[bool, int]]:
        team_id = team_of.get(pid)
        if team_id is None:
            return []
        return fixtures_map.get(team_id, {}).get(gameweek, [])

    double_players = [pid for pid in squad_ids if len(_this_gw_fixtures(pid)) >= 2]
    blank_players = [pid for pid in squad_ids if len(_this_gw_fixtures(pid)) == 0]

    # 1. Bench Boost - checked first: it needs the largest number of
    # affected players, so when it clears its bar the signal is the
    # strongest and least ambiguous of the four.
    if ChipType.BENCH_BOOST in available and len(double_players) >= BENCH_BOOST_MIN_DOUBLE_PLAYERS:
        confidence = min(0.5 + 0.1 * len(double_players), 0.95)
        return ChipVerdict(
            chip=ChipType.BENCH_BOOST,
            confidence=confidence,
            reasoning_factors=[
                f"{len(double_players)} of our squad's players have a double gameweek in GW{gameweek} "
                "(bench players also score, so a wide bench boost payoff is likely)"
            ],
        )

    # 2. Free Hit
    if ChipType.FREE_HIT in available and len(blank_players) >= FREE_HIT_MIN_BLANK_PLAYERS:
        confidence = min(0.5 + 0.1 * len(blank_players), 0.95)
        return ChipVerdict(
            chip=ChipType.FREE_HIT,
            confidence=confidence,
            reasoning_factors=[
                f"{len(blank_players)} of our squad's players have no fixture at all in GW{gameweek} "
                "(a blank gameweek for a meaningful part of our squad)"
            ],
        )

    # 3. Triple Captain
    if ChipType.TRIPLE_CAPTAIN in available:
        best = _best_captain_candidate(conn, gameweek, squad_ids)
        if best is not None:
            best_fixtures = _this_gw_fixtures(best)
            if len(best_fixtures) >= 2:
                return ChipVerdict(
                    chip=ChipType.TRIPLE_CAPTAIN,
                    confidence=0.75,
                    reasoning_factors=[f"our best-predicted captain candidate (player {best}) has a double gameweek in GW{gameweek}"],
                )
            if best_fixtures and min(d for _, d in best_fixtures if d is not None) <= TRIPLE_CAPTAIN_FAVORABLE_FDR:
                return ChipVerdict(
                    chip=ChipType.TRIPLE_CAPTAIN,
                    confidence=0.6,
                    reasoning_factors=[
                        f"our best-predicted captain candidate (player {best}) has a particularly "
                        f"favorable fixture (FDR <= {TRIPLE_CAPTAIN_FAVORABLE_FDR}) in GW{gameweek}"
                    ],
                )

    # 4. Wildcard - deliberately the least mechanically triggered: it isn't
    # tied to a specific gameweek shape the way the other three are (a
    # wildcard is about the state of our WHOLE squad's near-term run, not
    # one gameweek's fixture count), so it only fires on a genuinely poor
    # sustained run rather than any single-gameweek quirk.
    if ChipType.WILDCARD in available:
        avg_favorability = _average_squad_favorability(fixtures_map, team_of, squad_ids, gameweek)
        if avg_favorability is not None and avg_favorability < WILDCARD_POOR_FAVORABILITY_THRESHOLD:
            return ChipVerdict(
                chip=ChipType.WILDCARD,
                confidence=0.55,
                reasoning_factors=[
                    f"our squad's average fixture favorability over the next {WINDOW_GWS} gameweeks "
                    f"is notably poor ({avg_favorability:.1f}, weighted per GW{list(GW_WEIGHTS)}) - "
                    "a reasonable moment to consider an overhaul ahead of a better run"
                ],
            )

    return ChipVerdict(
        chip=None,
        confidence=0.8,
        reasoning_factors=["no fixture-calendar shape this gameweek justifies burning a scarce chip - hold for a better moment"],
    )
