"""Map the frozen vaastav archive (``merged_gws_2025-26.csv``, one row per
player-gameweek for a completed season) onto this project's self-built
silver schema (``ingestion/db.py``'s ``player_gameweek_stats`` table), so a
future retrain can read one consistent shape regardless of whether a row
came from last season's archive or this season's own pipeline.

This module does NOT write into the live Postgres store - the archive
stays frozen and separate, exactly as ARCHITECTURE.md describes ("last
season... stays frozen behind the already-trained model"). It only produces
an in-memory DataFrame shaped like ``player_gameweek_stats``, which a
retraining script can concatenate with a real silver export
(``ingestion.silver.load_gameweek_frame``-equivalent, once both are in the
same shape) before calling ``features.engineering.build_features_from_frame``
on the combined set.

COLUMN MAPPING
--------------
vaastav column      -> silver column              notes
------------------     -----------------------      -----
element              -> player_id                   vaastav's FPL element id; same numbering space as
                                                      silver's player_id AS LONG AS the player's id hasn't
                                                      been reused (FPL does not reuse retired ids within a
                                                      few seasons, but this is asserted, not guaranteed, by
                                                      the API - flag any mismatch rather than trust silently).
GW                   -> gw
minutes               -> minutes
total_points          -> total_points
goals_scored          -> goals_scored
assists               -> assists
clean_sheets          -> clean_sheets
goals_conceded        -> goals_conceded
own_goals             -> own_goals
penalties_saved       -> penalties_saved
penalties_missed      -> penalties_missed
yellow_cards          -> yellow_cards
red_cards             -> red_cards
saves                 -> saves
bonus                 -> bonus
bps                   -> bps
influence             -> influence
creativity            -> creativity
threat                -> threat
ict_index             -> ict_index
starts                -> starts
expected_goals        -> expected_goals
expected_assists      -> expected_assists
expected_goal_involvements -> expected_goal_involvements
expected_goals_conceded    -> expected_goals_conceded
was_home              -> was_home
value                 -> price                      silver stores GBP millions directly; vaastav's `value`
                                                      is 10x that (40 == GBP4.0m), so price = value / 10.
selected              -> selected
transfers_balance     -> transfers_balance

NO CLEAN EQUIVALENT - documented, not guessed:
  * `team` (name string) -> silver's `team_id` needs an id, not a name.
    There is no direct rename: the caller must look the id up from a
    teams-by-name table for the TARGET season (silver's `teams`), because
    FPL's team ids are NOT stable labels - they can be reassigned when clubs
    are promoted/relegated between seasons. This function does not attempt
    that join; it returns `team_name` as an informational column instead of
    guessing a `team_id`.
  * `opponent_team` -> silver's `opponent_team_id` has the same problem, one
    level worse: vaastav's `opponent_team` IS already a numeric id, but it's
    that ARCHIVE SEASON's id numbering, which is not guaranteed to match
    current-season silver's numbering for the same club. Returned as
    `opponent_team_id_archive_season` to make clear it is not directly
    comparable to silver's `opponent_team_id` without a season-specific
    team-id crosswalk this project does not currently build.
  * `fixture` -> silver's `fixture_id` has the identical problem (that
    season's fixture id numbering). Kept as `fixture_id_archive_season`,
    informational only.
  * `kickoff_time` -> silver keeps kickoff time on the `fixtures` table, not
    on `player_gameweek_stats` itself. Carried through as its own column
    here since `features.engineering` needs it (for `rest_days`) regardless
    of which table it lives in upstream.
  * `name` (display name string) / `position` -> silver keeps these on the
    `players` table, not `player_gameweek_stats`. Carried through as
    convenience columns (`features.engineering` groups by `name` and reads
    `position` directly) rather than dropped.
  * `round` -> exact duplicate of `GW`. Dropped.
  * `team_h_score` / `team_a_score` -> not stored on silver's
    `player_gameweek_stats` either (they live on `fixtures`), but
    `features.engineering`'s team-form aggregation needs them, so they are
    carried through here the same way `kickoff_time` is.
  * `transfers_in`, `transfers_out` (cumulative counts, distinct from
    `transfers_balance`) -> no silver column; not used by
    `features.engineering`. Dropped with this note rather than silently
    invented on the silver side.
  * `expected_goals_per_90`, `expected_assists_per_90`, `saves_per_90`,
    `starts_per_90`, and other pre-computed *_per_90 columns vaastav ships ->
    no silver equivalent. `features.engineering` computes its own per-90
    features from raw minutes + raw stats (rolling-window per-90, not
    single-gameweek per-90), so these are intentionally NOT reconciled -
    they'd be a different quantity wearing a similar name.
  * `defensive_contribution`, `clearances_blocks_interceptions`,
    `recoveries`, `tackles` -> present in the archive (2025-26 introduced FPL's
    new defensive-contribution scoring) but not yet modelled on the silver
    schema. Not dropped silently: flagged here as a known gap to add to
    `player_gameweek_stats` if a future feature needs them, since
    `event/{gw}/live`'s `stats` block does carry these fields too.
"""

from __future__ import annotations

import pandas as pd

# Straight 1:1 renames, vaastav column -> silver column.
DIRECT_RENAME = {
    "element": "player_id",
    "GW": "gw",
    "minutes": "minutes",
    "total_points": "total_points",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "own_goals": "own_goals",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "saves": "saves",
    "bonus": "bonus",
    "bps": "bps",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
    "starts": "starts",
    "expected_goals": "expected_goals",
    "expected_assists": "expected_assists",
    "expected_goal_involvements": "expected_goal_involvements",
    "expected_goals_conceded": "expected_goals_conceded",
    "was_home": "was_home",
    "selected": "selected",
    "transfers_balance": "transfers_balance",
}

# Columns that need a transform, not just a rename.
DERIVED = {
    "price": lambda df: df["value"] / 10.0,
}

# Columns kept through as informational/convenience, NOT silver's own
# player_gameweek_stats columns - see the "NO CLEAN EQUIVALENT" section above
# for why each one can't be a clean rename.
CARRY_THROUGH = {
    "team": "team_name",
    "opponent_team": "opponent_team_id_archive_season",
    "fixture": "fixture_id_archive_season",
    "kickoff_time": "kickoff_time",
    "name": "name",
    "position": "position",
    "team_h_score": "team_h_score",
    "team_a_score": "team_a_score",
}

# Present in the archive, dropped here with a documented reason (see module
# docstring) rather than silently discarded.
DROPPED_WITH_NO_EQUIVALENT = [
    "round", "transfers_in", "transfers_out", "value",
    "defensive_contribution", "clearances_blocks_interceptions",
    "recoveries", "tackles",
]


def map_archive_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with vaastav's columns renamed/derived onto
    silver's ``player_gameweek_stats`` shape, plus the informational
    carry-through columns documented above.

    ``df`` should already be through ``agents/stats/data.py::load_gameweeks``
    (double gameweeks collapsed to one row per player per GW) - this
    function only renames/derives columns, it doesn't do any more cleaning.
    """
    out = pd.DataFrame(index=df.index)

    for src, dst in DIRECT_RENAME.items():
        if src in df.columns:
            out[dst] = df[src]

    for dst, fn in DERIVED.items():
        out[dst] = fn(df)

    for src, dst in CARRY_THROUGH.items():
        if src in df.columns:
            out[dst] = df[src]

    return out
