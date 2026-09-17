"""Which chips are still available for our team.

Reads ``chip_usage`` (ingestion/db.py), NOT ``my_team_state.active_chip`` -
see that table's schema comment for the real gap this fixes: active_chip
only reflects whichever gameweek happened to be current each time the
weekly job ran, so it has real holes if the job started mid-season or
missed a week. ``chip_usage`` is populated from FPL's own authoritative
``entry/{id}/history/`` endpoint (ingestion/silver.py::upsert_chip_usage),
which is complete regardless of ingestion history.
"""

from __future__ import annotations

from shared.contracts import ChipType

# FPL grants each chip type TWICE per season (once per season half, since
# the 2023-24 rule change). The exact gameweek each season's second half
# starts isn't tracked anywhere in this project's ingestion pipeline (it
# isn't a `bootstrap-static` field this project already pulls), so this
# counts TOTAL uses this season against a flat cap rather than modeling
# the half-season eligibility boundary precisely.
# ponytail: flat per-season cap, not a real half-season calendar - upgrade
# if/when the season-half boundary gameweek is ever ingested.
MAX_USES_PER_CHIP = 2

# FPL's own raw chip codes (as stored in chip_usage.chip, matching
# my_team_state.active_chip's vocabulary) mapped onto the shared contract's
# ChipType enum.
FPL_CODE_TO_CHIP_TYPE = {
    "wildcard": ChipType.WILDCARD,
    "bboost": ChipType.BENCH_BOOST,
    "3xc": ChipType.TRIPLE_CAPTAIN,
    "freehit": ChipType.FREE_HIT,
}


def available_chips(conn) -> set[ChipType]:
    """Every chip type with fewer than MAX_USES_PER_CHIP recorded uses this
    season. A chip never played at all correctly counts as 0 uses (it's
    just absent from chip_usage), not an error.
    """
    rows = conn.execute("SELECT chip, COUNT(*) AS n FROM chip_usage GROUP BY chip").fetchall()
    used_counts = {r["chip"]: r["n"] for r in rows}
    return {
        chip_type
        for code, chip_type in FPL_CODE_TO_CHIP_TYPE.items()
        if used_counts.get(code, 0) < MAX_USES_PER_CHIP
    }
