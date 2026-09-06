"""Feature engineering for the Stats agent's expected-points model.

THE ONE RULE IN THIS FILE
-------------------------
We are predicting a player's ``total_points`` for gameweek N. At the moment
we would make that prediction (before gameweek N's deadline) we know:

  * static facts: position, team, price, home/away, who the opponent is;
  * everything that happened in gameweeks 1 .. N-1.

We do NOT know anything about gameweek N's match itself (minutes played,
goals, xG, bonus, the score...). Every feature below is therefore built
either from static facts or from *lagged* history. The mechanism for the
lag is always the same: sort by gameweek within a player, then
``.shift(1)`` before any rolling window, so the current row can never see
its own outcome. Getting this wrong is the classic way to build a model
that looks brilliant offline and is useless in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_gameweeks, team_gameweek_table

# The windows we roll most stats over. 3 = "hot right now", 5 = "current
# form", 10 = "this-season baseline". Kept short because form in football
# decays fast and the season is only 38 gameweeks.
SHORT_W, MED_W, LONG_W = 3, 5, 10


# ---------------------------------------------------------------------------
# Team-level rolling tables (used for team form + opponent strength)
# ---------------------------------------------------------------------------
def _rolling_team_form(team_gw: pd.DataFrame) -> pd.DataFrame:
    """For each (team, GW) attach that team's form *coming into* the GW.

    All columns are shifted by one gameweek so they only ever describe
    completed matches.
    """
    t = team_gw.sort_values(["team", "GW"]).copy()
    grp = t.groupby("team")

    # .shift(1) first => the window ends at GW N-1, never includes GW N.
    t["team_gf_roll5"] = grp["goals_for"].shift(1).rolling(MED_W, min_periods=1).mean().values
    t["team_ga_roll5"] = grp["goals_against"].shift(1).rolling(MED_W, min_periods=1).mean().values
    t["team_xgf_roll5"] = grp["xg_for"].shift(1).rolling(MED_W, min_periods=1).mean().values
    t["team_xga_roll5"] = grp["xg_against"].shift(1).rolling(MED_W, min_periods=1).mean().values
    t["team_cs_roll5"] = grp["clean_sheet"].shift(1).rolling(MED_W, min_periods=1).mean().values
    return t[[
        "team", "GW",
        "team_gf_roll5", "team_ga_roll5", "team_xgf_roll5",
        "team_xga_roll5", "team_cs_roll5",
    ]]


# ---------------------------------------------------------------------------
# Player-level rolling helpers
# ---------------------------------------------------------------------------
def _roll_mean(grp, window: int) -> np.ndarray:
    """Lagged rolling mean: average of the previous ``window`` gameweeks."""
    return grp.shift(1).rolling(window, min_periods=1).mean().values


def _roll_sum(grp, window: int) -> np.ndarray:
    """Lagged rolling sum: total over the previous ``window`` gameweeks."""
    return grp.shift(1).rolling(window, min_periods=1).sum().values


def build_feature_frame(csv_path=None) -> tuple[pd.DataFrame, list[str], str]:
    """Return ``(frame, feature_columns, target_column)``.

    ``frame`` has one row per player per gameweek with all features plus the
    target. Rows from the first few gameweeks (no history yet) are kept but
    will be dropped by the training script's warm-up cut.
    """
    df = load_gameweeks(csv_path) if csv_path else load_gameweeks()
    df = df.sort_values(["name", "GW"]).reset_index(drop=True)

    # ----- team form + opponent strength ---------------------------------
    team_gw = team_gameweek_table(df)
    team_form = _rolling_team_form(team_gw)

    # Own team's form coming into the gameweek.
    df = df.merge(team_form, on=["team", "GW"], how="left")

    # Opponent's form coming into the gameweek: same table joined on the
    # opponent's name, columns renamed to "opp_*".
    opp_form = team_form.rename(
        columns={
            "team": "opponent_name",
            "team_gf_roll5": "opp_gf_roll5",
            "team_ga_roll5": "opp_ga_roll5",
            "team_xgf_roll5": "opp_xgf_roll5",
            "team_xga_roll5": "opp_xga_roll5",
            "team_cs_roll5": "opp_cs_roll5",
        }
    )
    df = df.merge(opp_form, on=["opponent_name", "GW"], how="left")

    # ----- per-player rolling history ----------------------------------------
    g_pts = df.groupby("name")["total_points"]
    g_min = df.groupby("name")["minutes"]
    g_start = df.groupby("name")["starts"]
    g_bonus = df.groupby("name")["bonus"]
    g_ict = df.groupby("name")["ict_index"]
    g_xgi = df.groupby("name")["expected_goal_involvements"]
    g_threat = df.groupby("name")["threat"]
    g_creativity = df.groupby("name")["creativity"]
    g_gc = df.groupby("name")["goals_conceded"]
    g_bps = df.groupby("name")["bps"]

    # === REQUIRED FEATURE 1: rolling form windows =========================
    # The bread and butter of FPL: recent points. Three horizons so the model
    # can tell "one good game" from "consistently good".
    df["pts_roll3"] = _roll_mean(g_pts, SHORT_W)
    df["pts_roll5"] = _roll_mean(g_pts, MED_W)
    df["pts_roll10"] = _roll_mean(g_pts, LONG_W)

    # === REQUIRED FEATURE 2: fixture-adjusted expected points =============
    # Recent scoring rate, then nudged up for an easy opponent and down for a
    # hard one. "opp_xga_roll5" is how many xG the opponent has been
    # conceding: high => leaky defence => scale the player's form UP.
    # Divide by a league-ish average (~1.4 xG/game) to centre the multiplier
    # near 1.0, and clip so one freak result can't blow up a prediction.
    league_avg_xg = 1.4
    fixture_multiplier = (df["opp_xga_roll5"] / league_avg_xg).clip(0.5, 1.8)
    df["fixture_adj_xp"] = df["pts_roll5"] * fixture_multiplier
    # Keep the raw multiplier as its own feature so a tree can use it
    # independently of current form.
    df["fixture_multiplier"] = fixture_multiplier

    # === REQUIRED FEATURE 3: per-90 normalisation ========================
    # Points/goals per 90 minutes played, so a high-impact sub isn't punished
    # for low minutes. sum(stat over window) / sum(minutes over window) * 90.
    mins5 = _roll_sum(g_min, MED_W)
    safe = mins5 > 0
    df["pts_per90_roll5"] = np.where(safe, _roll_sum(g_pts, MED_W) / np.where(safe, mins5, 1) * 90, 0.0)
    df["xgi_per90_roll5"] = np.where(safe, _roll_sum(g_xgi, MED_W) / np.where(safe, mins5, 1) * 90, 0.0)
    df["bps_per90_roll5"] = np.where(safe, _roll_sum(g_bps, MED_W) / np.where(safe, mins5, 1) * 90, 0.0)
    df["threat_per90_roll5"] = np.where(safe, _roll_sum(g_threat, MED_W) / np.where(safe, mins5, 1) * 90, 0.0)
    # creativity per 90: proxy for set-piece / chance-creation duty, since
    # corner and free-kick takers rack up creativity score.
    df["creativity_per90_roll5"] = np.where(
        safe, _roll_sum(g_creativity, MED_W) / np.where(safe, mins5, 1) * 90, 0.0
    )

    # === REQUIRED FEATURE 4: price-change momentum =======================
    # The FPL price algorithm moves a player up when many managers buy him -
    # a crude but real crowd-sourced signal of expected returns.
    g_price = df.groupby("name")["price"]
    df["price_now"] = df["price"]  # level: the market's season-long prior
    df["price_delta3"] = (df["price"] - g_price.shift(SHORT_W)).fillna(0.0)
    df["price_delta5"] = (df["price"] - g_price.shift(MED_W)).fillna(0.0)
    # direction of the last move: +1 rising, -1 falling, 0 flat.
    df["price_momentum_sign"] = np.sign(df["price"] - g_price.shift(1)).fillna(0.0)

    # === EXTRA FEATURES (one-line justification each) ====================

    # Home advantage: teams and players simply score more at home.
    df["is_home"] = df["was_home"].astype(int)

    # Rest days since last fixture: short turnarounds mean fatigue and rotation.
    kickoff_prev = df.groupby("name")["kickoff_time"].shift(1)
    rest = (df["kickoff_time"] - kickoff_prev).dt.total_seconds() / 86400
    df["rest_days"] = rest.clip(2, 21).fillna(7.0)

    # Rolling minutes: the single biggest precondition for points - you can't
    # score sitting on the bench.
    df["minutes_roll3"] = _roll_mean(g_min, SHORT_W)
    df["minutes_roll5"] = _roll_mean(g_min, MED_W)

    # Start rate: how often the player was in the XI recently - "nailed on" vs
    # rotation risk.
    df["start_rate_roll5"] = _roll_mean(g_start, MED_W)

    # Minutes trend: last-3 minutes minus last-5 minutes. Positive => the
    # player is working his way into the team right now (or back from injury).
    df["minutes_trend"] = df["minutes_roll3"] - df["minutes_roll5"]

    # Double gameweek flag: two matches this GW roughly doubles the points
    # ceiling, and the model should know that.
    df["is_double_gw"] = (df["n_fixtures"] > 1).astype(int)

    # Recent bonus points: bonus is sticky - players who dominate a match tend
    # to keep doing it, and it's 1-3 points the base stats don't capture.
    df["bonus_roll5"] = _roll_mean(g_bonus, MED_W)

    # Rolling ICT index: FPL's own influence/creativity/threat composite; a
    # broad "how involved is this player" number that leads returns.
    df["ict_roll5"] = _roll_mean(g_ict, MED_W)

    # Opponent defensive strength (goals conceded per game, recent): weak
    # defences hand out attacking returns.
    df["opp_def_conceded_roll5"] = df["opp_ga_roll5"]
    # Opponent xG conceded (recent): the less-noisy version of the line above.
    df["opp_def_xgc_roll5"] = df["opp_xga_roll5"]
    # Opponent attacking threat (recent goals for): matters for defenders/GK -
    # a dangerous opponent lowers clean-sheet odds.
    df["opp_attack_roll5"] = df["opp_gf_roll5"]

    # Own team attacking form (recent goals for): good attacking sides create
    # more chances, lifting every attacker's expected points.
    df["team_attack_roll5"] = df["team_gf_roll5"]
    # Own team clean-sheet rate (recent): drives defender and goalkeeper points.
    df["team_cs_rate_roll5"] = df["team_cs_roll5"]
    # Own team xG conceded (recent): forward-looking "are we solid at the back",
    # useful for a defender's clean-sheet expectation.
    df["team_xgc_roll5"] = df["team_xga_roll5"]

    # Goals conceded while this player was on (recent): catches a defender in a
    # leaky side even if the team-level number looks okay.
    df["conceded_roll5"] = _roll_mean(g_gc, MED_W)

    # Ownership (selected-by, lagged one GW): very high ownership is the crowd
    # saying "reliable points asset". Lagged to stay honest about timing.
    df["ownership_lag1"] = df.groupby("name")["selected"].shift(1).fillna(0.0)

    # Net transfer momentum (transfers_balance, lagged one GW): the live market
    # moving toward or away from a player, before the price actually changes.
    df["transfer_balance_lag1"] = df.groupby("name")["transfers_balance"].shift(1).fillna(0.0)

    # Season-so-far average points (expanding mean, lagged): a longer memory
    # than the 5-GW window - regresses hot/cold streaks toward the player's
    # true level.
    df["pts_season_avg"] = (
        df.groupby("name")["total_points"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Games of history available (lagged count): lets the model discount
    # rolling features computed from only one or two games.
    df["games_played"] = df.groupby("name").cumcount().astype(float)

    # Position as one-hot: GK/DEF/MID/FWD score under different rules, so the
    # model needs the position explicitly, not just implied by other stats.
    pos_dummies = pd.get_dummies(df["position"], prefix="pos").astype(int)
    df = pd.concat([df, pos_dummies], axis=1)

    # ----- assemble the feature list ------------------------------------
    feature_cols = [
        # required 1: rolling form
        "pts_roll3", "pts_roll5", "pts_roll10",
        # required 2: fixture-adjusted expected points
        "fixture_adj_xp", "fixture_multiplier",
        # required 3: per-90
        "pts_per90_roll5", "xgi_per90_roll5", "bps_per90_roll5",
        "threat_per90_roll5", "creativity_per90_roll5",
        # required 4: price momentum
        "price_now", "price_delta3", "price_delta5", "price_momentum_sign",
        # extras
        "is_home", "rest_days",
        "minutes_roll3", "minutes_roll5", "start_rate_roll5", "minutes_trend",
        "is_double_gw",
        "bonus_roll5", "ict_roll5",
        "opp_def_conceded_roll5", "opp_def_xgc_roll5", "opp_attack_roll5",
        "team_attack_roll5", "team_cs_rate_roll5", "team_xgc_roll5",
        "conceded_roll5",
        "ownership_lag1", "transfer_balance_lag1",
        "pts_season_avg", "games_played",
        *list(pos_dummies.columns),
    ]

    target = "total_points"

    # Fill the handful of NaNs that survive (a player's very first GW has no
    # history at all). 0 is the right default for "no information yet".
    df[feature_cols] = df[feature_cols].fillna(0.0)

    keep = ["name", "position", "team", "GW", "minutes", target, *feature_cols]
    return df[keep].reset_index(drop=True), feature_cols, target


if __name__ == "__main__":
    frame, feats, tgt = build_feature_frame()
    print(f"{len(frame):,} rows, {len(feats)} features, target = {tgt!r}")
    print(frame[["name", "GW", "pts_roll5", "fixture_adj_xp", tgt]].head(12))
