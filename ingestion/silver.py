"""Silver layer: parse bronze JSON into the relational tables defined in
``db.py``, upserting on each table's natural key so re-running ingestion for
a gameweek that's already there replaces rows instead of duplicating them.

Every ``upsert_*`` function takes already-parsed JSON (what ``bronze.py``'s
``fetch_and_land_*`` functions return) - it never calls the API itself. That
split (bronze fetches + lands, silver only parses what's handed to it) is
what makes the idempotency tests possible: silver parsing can be exercised
against a fixed JSON fixture with no network involved at all.

FPL API SHAPE NOTES (why some of this looks like more than a straight column
rename):
  * ``event/{gw}/live/``'s ``elements[i]["stats"]`` has minutes, points,
    goals etc. for that gameweek, but NOT which fixture it was, the
    opponent, home/away, price, ownership, or transfer balance. Opponent/
    home-away come from ``elements[i]["explain"][0]["fixture"]``
    cross-referenced against the fixtures table (already upserted first,
    every ingestion run). Price/ownership/transfers_balance simply aren't in
    this endpoint at all; see the docstring on ``upsert_player_gameweek_stats``
    for how each mode handles that gap.
  * A double gameweek gives a player two entries in ``explain`` (one per
    fixture) but ``stats`` is already the gameweek TOTAL across both - so
    ``n_fixtures = len(explain)`` and home/away/opponent come from picking
    the first fixture (an accepted approximation for a rare case; the model
    only uses opponent-strength features for one row anyway).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from psycopg2.extras import Json

from . import db  # noqa: F401 - only used for the "db.Connection" type hints below

# FPL's element_type id -> the position label the rest of this project uses
# (matches vaastav's convention: "GK"/"DEF"/"MID"/"FWD", not FPL's own
# "GKP"/"DEF"/"MID"/"FWD" short names) so silver-sourced and archive-sourced
# frames need no further position translation downstream.
POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Master data: teams, players, gameweeks, fixtures
# ---------------------------------------------------------------------------
def upsert_teams(conn: "db.Connection", bootstrap: dict) -> None:
    now = _now()
    rows = [
        (t["id"], t["name"], t.get("short_name"), t.get("strength"), now)
        for t in bootstrap["teams"]
    ]
    conn.executemany(
        """
        INSERT INTO teams (team_id, name, short_name, strength, updated_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT(team_id) DO UPDATE SET
            name=excluded.name, short_name=excluded.short_name,
            strength=excluded.strength, updated_at=excluded.updated_at
        """,
        rows,
    )
    conn.commit()


def upsert_players(conn: "db.Connection", bootstrap: dict) -> None:
    now = _now()
    rows = [
        (
            e["id"], e["web_name"], e.get("first_name"), e.get("second_name"),
            e.get("team"), POSITION_BY_ELEMENT_TYPE.get(e.get("element_type")), now,
        )
        for e in bootstrap["elements"]
    ]
    conn.executemany(
        """
        INSERT INTO players (player_id, web_name, first_name, second_name, team_id, position, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(player_id) DO UPDATE SET
            web_name=excluded.web_name, first_name=excluded.first_name,
            second_name=excluded.second_name, team_id=excluded.team_id,
            position=excluded.position, updated_at=excluded.updated_at
        """,
        rows,
    )
    conn.commit()


def upsert_gameweeks(conn: "db.Connection", bootstrap: dict) -> None:
    now = _now()
    rows = [
        (
            ev["id"], ev.get("name"), ev.get("deadline_time"),
            int(bool(ev.get("finished"))), int(bool(ev.get("is_current"))),
            ev.get("average_entry_score"), now,
        )
        for ev in bootstrap["events"]
    ]
    conn.executemany(
        """
        INSERT INTO gameweeks (gw, name, deadline_time, finished, is_current, average_entry_score, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(gw) DO UPDATE SET
            name=excluded.name, deadline_time=excluded.deadline_time,
            finished=excluded.finished, is_current=excluded.is_current,
            average_entry_score=excluded.average_entry_score, updated_at=excluded.updated_at
        """,
        rows,
    )
    conn.commit()


def upsert_fixtures(conn: "db.Connection", fixtures: list) -> None:
    now = _now()
    rows = [
        (
            f["id"], f.get("event"), f.get("team_h"), f.get("team_a"),
            f.get("team_h_score"), f.get("team_a_score"), f.get("kickoff_time"),
            int(bool(f.get("finished"))), f.get("team_h_difficulty"), f.get("team_a_difficulty"), now,
        )
        for f in fixtures
    ]
    conn.executemany(
        """
        INSERT INTO fixtures (fixture_id, gw, team_h, team_a, team_h_score, team_a_score,
                               kickoff_time, finished, team_h_difficulty, team_a_difficulty, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(fixture_id) DO UPDATE SET
            gw=excluded.gw, team_h=excluded.team_h, team_a=excluded.team_a,
            team_h_score=excluded.team_h_score, team_a_score=excluded.team_a_score,
            kickoff_time=excluded.kickoff_time, finished=excluded.finished,
            team_h_difficulty=excluded.team_h_difficulty, team_a_difficulty=excluded.team_a_difficulty,
            updated_at=excluded.updated_at
        """,
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# player_gameweek_stats - the core accumulating fact table
# ---------------------------------------------------------------------------
_STAT_FIELDS = [
    "minutes", "total_points", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence",
    "creativity", "threat", "ict_index", "starts", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
]


def upsert_player_gameweek_stats(
    conn: "db.Connection",
    gw: int,
    event_live: dict,
    *,
    now_cost_by_player: dict[int, float] | None = None,
) -> int:
    """Upsert one gameweek's worth of ``event/{gw}/live`` into silver.

    ``now_cost_by_player`` (player_id -> current price in GBP millions) is
    optional and only ever an approximation: it's the price *at ingestion
    time*, not necessarily at that gameweek's deadline. Pass it from a fresh
    ``bootstrap-static`` pull when ingesting the gameweek that JUST finished
    (the weekly path - see weekly.py) where "current price" and "price at
    that deadline" are close enough to be useful. Leave it ``None`` when
    backfilling older gameweeks (see backfill.py's docstring): using today's
    price for a gameweek from months ago would be actively wrong, not just
    approximate, so we store NULL instead of guessing.

    ``selected`` (ownership) and ``transfers_balance`` are not available from
    this endpoint at all, for any gameweek - they are always stored NULL
    here. ``weekly.py`` optionally refines them for OUR OWN squad's ~15
    players afterwards, via ``element-summary`` (see that module).

    Returns the number of player-gameweek rows written.
    """
    now = _now()

    # Player -> team lookup, needed to resolve opponent/home-away from the
    # fixtures table (event/live's own "stats" has no fixture context).
    players = {
        r["player_id"]: r["team_id"]
        for r in conn.execute("SELECT player_id, team_id FROM players")
    }
    fixtures_by_id = {
        r["fixture_id"]: r
        for r in conn.execute("SELECT fixture_id, team_h, team_a FROM fixtures WHERE gw = %s", (gw,))
    }

    rows = []
    for el in event_live["elements"]:
        player_id = el["id"]
        stats = el.get("stats", {})
        explain = el.get("explain") or []
        n_fixtures = len(explain)

        team_id = players.get(player_id)
        opponent_team_id = None
        was_home = None
        if explain:
            fixture = fixtures_by_id.get(explain[0].get("fixture"))
            if fixture is not None and team_id is not None:
                if fixture["team_h"] == team_id:
                    was_home, opponent_team_id = 1, fixture["team_a"]
                elif fixture["team_a"] == team_id:
                    was_home, opponent_team_id = 0, fixture["team_h"]

        row = [player_id, gw]
        row += [stats.get(f) for f in _STAT_FIELDS]
        row += [
            team_id, opponent_team_id, was_home, n_fixtures,
            (now_cost_by_player or {}).get(player_id),  # price: best-effort, see docstring
            None,  # selected: not available from this endpoint
            None,  # transfers_balance: not available from this endpoint
            "event_live", now,
        ]
        rows.append(tuple(row))

    columns = (
        ["player_id", "gw"] + _STAT_FIELDS
        + ["team_id", "opponent_team_id", "was_home", "n_fixtures", "price", "selected", "transfers_balance", "source", "ingested_at"]
    )
    placeholders = ", ".join("%s" for _ in columns)
    update_cols = [c for c in columns if c not in ("player_id", "gw")]
    update_clause = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
    conn.executemany(
        f"""
        INSERT INTO player_gameweek_stats ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(player_id, gw) DO UPDATE SET {update_clause}
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def refine_player_gameweek_from_element_summary(
    conn: "db.Connection", player_id: int, element_summary: dict
) -> int:
    """Fill in the gaps ``event/live`` leaves for ONE player, from their own
    ``element-summary`` history - specifically ``value`` (price) and
    ``selected``/``transfers_balance``, which are genuinely per-gameweek
    historical figures in this endpoint (unlike ``event/live``).

    Deliberately scoped to one player at a time and called only for the
    handful of players we actually care about precisely (our own squad -
    see weekly.py), not the full player pool: looping this over ~700 players
    every gameweek is exactly the "~700 calls instead of one" cost the task
    calls out avoiding for bulk backfill.
    """
    now = _now()
    rows = [
        (
            (h.get("value") / 10.0) if h.get("value") is not None else None,
            h.get("selected"),
            h.get("transfers_balance"),
            now,
            player_id,
            h["round"],
        )
        for h in element_summary.get("history", [])
    ]
    conn.executemany(
        """
        UPDATE player_gameweek_stats
        SET price = COALESCE(%s, price), selected = %s, transfers_balance = %s, ingested_at = %s
        WHERE player_id = %s AND gw = %s
        """,
        rows,
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# my_team_state - our own squad/bank/transfers
# ---------------------------------------------------------------------------
def upsert_my_team_state(
    conn: "db.Connection", gw: int, entry: dict, picks: dict, transfers: list
) -> None:
    now = _now()
    history = picks.get("entry_history", {})
    conn.execute(
        """
        INSERT INTO my_team_state (gw, bank, team_value, total_points, overall_rank,
                                    event_transfers, event_transfers_cost, points_on_bench,
                                    active_chip, picks, transfers, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(gw) DO UPDATE SET
            bank=excluded.bank, team_value=excluded.team_value,
            total_points=excluded.total_points, overall_rank=excluded.overall_rank,
            event_transfers=excluded.event_transfers, event_transfers_cost=excluded.event_transfers_cost,
            points_on_bench=excluded.points_on_bench, active_chip=excluded.active_chip,
            picks=excluded.picks, transfers=excluded.transfers, updated_at=excluded.updated_at
        """,
        (
            gw,
            (history.get("bank") or 0) / 10.0,
            (history.get("value") or 0) / 10.0,
            history.get("total_points") or entry.get("summary_overall_points"),
            history.get("overall_rank") or entry.get("summary_overall_rank"),
            history.get("event_transfers"),
            history.get("event_transfers_cost"),
            history.get("points_on_bench"),
            picks.get("active_chip"),
            Json(picks.get("picks", [])),
            Json(transfers),
            now,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# The "gold-ready" read path: reshape silver back into the common gameweek
# frame that features/engineering.py consumes (see that module's INPUT
# CONTRACT). This is what makes silver double as the live feature source,
# not just a serving cache.
# ---------------------------------------------------------------------------
def load_gameweek_frame(conn: "db.Connection") -> pd.DataFrame:
    """One row per player per gameweek, joined up with names/positions/prices
    and shaped exactly like ``agents/stats/data.py::load_gameweeks()``'s
    output - the shape ``features.engineering.build_features_from_frame``
    expects. Rows with missing price/selected/transfers_balance (backfilled
    gameweeks - see backfill.py) come through as NaN, which the shared
    feature module already treats as "no information yet".
    """
    query = """
        SELECT
            pgs.gw AS "GW",
            p.web_name AS name,
            p.position AS position,
            t.name AS team,
            ot.name AS opponent_name,
            pgs.was_home AS was_home,
            pgs.n_fixtures AS n_fixtures,
            pgs.price AS price,
            pgs.minutes AS minutes,
            pgs.starts AS starts,
            pgs.total_points AS total_points,
            pgs.bonus AS bonus,
            pgs.bps AS bps,
            pgs.ict_index AS ict_index,
            pgs.threat AS threat,
            pgs.creativity AS creativity,
            pgs.expected_goals AS expected_goals,
            pgs.expected_goal_involvements AS expected_goal_involvements,
            pgs.expected_goals_conceded AS expected_goals_conceded,
            pgs.goals_conceded AS goals_conceded,
            pgs.selected AS selected,
            pgs.transfers_balance AS transfers_balance,
            f.kickoff_time AS kickoff_time,
            f.team_h_score AS team_h_score,
            f.team_a_score AS team_a_score
        FROM player_gameweek_stats pgs
        JOIN players p ON p.player_id = pgs.player_id
        LEFT JOIN teams t ON t.team_id = pgs.team_id
        LEFT JOIN teams ot ON ot.team_id = pgs.opponent_team_id
        LEFT JOIN fixtures f ON f.gw = pgs.gw
            AND ((f.team_h = pgs.team_id AND pgs.was_home = 1)
                 OR (f.team_a = pgs.team_id AND pgs.was_home = 0))
        ORDER BY p.web_name, pgs.gw
    """
    df = pd.read_sql_query(query, conn)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["was_home"] = df["was_home"].fillna(0).astype(bool)
    # Silver only ever holds ONE season (the in-progress one - see
    # ARCHITECTURE.md's "Backfill scope"), so a constant tag is enough to
    # keep this frame's gameweeks from bleeding into the archive's own GW
    # numbering when agents/stats/retrain.py concatenates the two - see
    # features/engineering.py's INPUT CONTRACT note on `season`.
    df["season"] = "current"
    return df
