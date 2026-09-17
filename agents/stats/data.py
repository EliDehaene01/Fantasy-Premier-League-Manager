"""Loading and cleaning the raw vaastav ``merged_gws`` file.

The raw file is one row per player per *fixture*. That is almost one row
per player per gameweek, except in a "double gameweek" (a team plays twice
in one gameweek) where a player gets two rows with the same ``GW``. Every
downstream step assumes one row per player per gameweek, so this module
collapses those doubles into a single aggregated row.

It also builds the ``opponent_team`` id -> team name lookup, which the raw
file does not give us directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is three levels up from this file: agents/stats/data.py -> repo/
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = REPO_ROOT / "merged_gws_2025-26.csv"

_SEASON_RE = re.compile(r"(\d{4}-\d{2})")


def _infer_season(csv_path: Path | str) -> str:
    """Pull a season label like "2025-26" out of the CSV filename.

    Used to tag rows with `season` (see features/engineering.py's INPUT
    CONTRACT) so a retrain combining this frozen archive with the current
    season's silver data never treats last season's GW38 as the row right
    before this season's GW1. Falls back to a generic label if the filename
    doesn't carry a season-shaped substring - better than crashing on an
    unusual path, and still distinct from silver's "current" tag.
    """
    m = _SEASON_RE.search(str(csv_path))
    return m.group(1) if m else "archive"

# Columns that describe the *outcome* of a match. They are only known AFTER
# kick-off, so a model predicting a gameweek's points must never read them
# from that same gameweek's row - only from earlier gameweeks (via rolling
# features). They are listed here so feature code can be explicit about it.
OUTCOME_COLUMNS = [
    "total_points", "minutes", "goals_scored", "assists", "bonus", "bps",
    "clean_sheets", "goals_conceded", "own_goals", "penalties_missed",
    "penalties_saved", "red_cards", "yellow_cards", "saves", "starts",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "team_h_score", "team_a_score",
    "defensive_contribution", "clearances_blocks_interceptions",
    "recoveries", "tackles",
]

# When we collapse a double gameweek into one row, counting stats get summed
# and everything else takes the first value.
_SUM_ON_COLLAPSE = [
    "minutes", "total_points", "goals_scored", "assists", "bonus", "bps",
    "clean_sheets", "goals_conceded", "own_goals", "penalties_missed",
    "penalties_saved", "red_cards", "yellow_cards", "saves", "starts",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "defensive_contribution", "clearances_blocks_interceptions",
    "recoveries", "tackles",
    "transfers_in", "transfers_out", "transfers_balance",
]


def _build_opponent_name_map(df: pd.DataFrame) -> dict[int, str]:
    """Work out which team name each ``opponent_team`` id refers to.

    Every fixture has exactly two teams in the data. Within one fixture,
    team A's row carries ``opponent_team = <id of B>`` and team B's row
    carries ``opponent_team = <id of A>``. So for each fixture we can pair
    "the other team's name" with "the id this team was given as opponent".
    Taking the majority vote over the whole season gives a clean id->name map.
    """
    pairs: list[tuple[int, str]] = []
    for _, grp in df.groupby("fixture"):
        teams = grp["team"].unique()
        if len(teams) != 2:
            continue
        a, b = teams
        # id that team B saw as its opponent == team A's id, and vice versa.
        id_of_a = grp.loc[grp["team"] == b, "opponent_team"].iloc[0]
        id_of_b = grp.loc[grp["team"] == a, "opponent_team"].iloc[0]
        pairs.append((int(id_of_a), a))
        pairs.append((int(id_of_b), b))

    pair_df = pd.DataFrame(pairs, columns=["opp_id", "team_name"])
    return (
        pair_df.groupby("opp_id")["team_name"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )


def load_gameweeks(csv_path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Return a cleaned frame: one row per player per gameweek.

    Adds helper columns:
      * ``opponent_name`` - human-readable opponent (from the id map)
      * ``n_fixtures``    - how many matches the player had that gameweek
                            (2 in a double gameweek, 1 otherwise)
      * ``price``         - ``value`` / 10, i.e. the on-screen price in Ā£m
    """
    df = pd.read_csv(csv_path)

    # Parse kick-off so we can compute rest days between fixtures later.
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)

    opp_map = _build_opponent_name_map(df)
    df["opponent_name"] = df["opponent_team"].map(opp_map)

    # --- collapse double gameweeks -----------------------------------------
    # Sum counting stats, take the first value of everything else.
    agg_spec: dict[str, str] = {}
    for col in df.columns:
        if col in ("name", "GW"):
            continue
        if col in _SUM_ON_COLLAPSE:
            agg_spec[col] = "sum"
        elif col == "kickoff_time":
            agg_spec[col] = "min"  # earliest match of the gameweek
        else:
            agg_spec[col] = "first"

    collapsed = (
        df.sort_values(["name", "GW", "kickoff_time"])
        .groupby(["name", "GW"], as_index=False)
        .agg(agg_spec)
    )
    n_fix = df.groupby(["name", "GW"]).size().rename("n_fixtures").reset_index()
    collapsed = collapsed.merge(n_fix, on=["name", "GW"], how="left")

    # value (price) is stored as an integer 10x the on-screen price (40 == 4.0).
    collapsed["price"] = collapsed["value"] / 10.0

    # Additive alias, not a rename - reconcile_archive.py's DIRECT_RENAME
    # still expects `element` present on this function's output and does
    # its own element -> player_id rename; this alias is purely so
    # features/engineering.py (which is source-agnostic between this
    # archive path and silver's live path) has a `player_id` column to
    # optionally carry through under the SAME name silver already uses,
    # rather than every downstream consumer needing to know it's called
    # `element` here specifically. See that module's INPUT CONTRACT note.
    collapsed["player_id"] = collapsed["element"]

    collapsed["season"] = _infer_season(csv_path)
    return collapsed.sort_values(["name", "GW"]).reset_index(drop=True)


def team_gameweek_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player rows up to one row per team per gameweek.

    Gives us team-level "how did this team do" numbers (goals for/against,
    expected goals for/against, whether they kept a clean sheet). Rolling
    versions of these become the "team form" and "opponent strength"
    features. Everything here is still an *outcome* - features must lag it.
    """
    rows = []
    for (team, gw), g in df.groupby(["team", "GW"]):
        # A player's row tells us the match score and whether they were home.
        gf = np.where(g["was_home"], g["team_h_score"], g["team_a_score"])
        ga = np.where(g["was_home"], g["team_a_score"], g["team_h_score"])
        # All players of a team share the same match score, so take the max
        # (handles the rare double-gameweek where two scores are present).
        goals_for = float(np.max(gf))
        goals_against = float(np.max(ga))
        rows.append(
            {
                "team": team,
                "GW": int(gw),
                "goals_for": goals_for,
                "goals_against": goals_against,
                # Team xG = sum of its players' individual expected goals.
                "xg_for": float(g["expected_goals"].sum()),
                # xG conceded is recorded per player but identical across a
                # team's outfielders; take the max as the team value.
                "xg_against": float(g["expected_goals_conceded"].max()),
                "clean_sheet": float(goals_against == 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["team", "GW"]).reset_index(drop=True)
