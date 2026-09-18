"""Seeds a historical season's vaastav archive into an isolated Postgres
schema, shaped exactly like the live silver tables (``teams``, ``fixtures``,
``players``, ``player_gameweek_stats`` - see ``ingestion/db.py``), so the
backtest engine can run the SAME DB-backed agent scoring code
(Fixtures/Contrarian/Template/Chips's ``scoring.py`` modules) against
historical data instead of live data, rather than maintaining a second,
divergent implementation of each agent's logic for backtest purposes.

Unlike ``data/reconcile_archive.py`` (which only maps ``player_gameweek_stats``
for retraining, deliberately staying out of Postgres), this module DOES
write to Postgres - but only ever into an isolated per-season schema
(``backtest_<season>``, e.g. ``backtest_2025_26``), the same test-isolation
mechanism every DB-touching test suite in this project already uses
(``get_connection(schema=...)``). It never touches the live ``public``
schema.

**Why real FDR, not a proxy**: the archive's ``fixtures.csv`` already carries
``team_h_difficulty``/``team_a_difficulty`` - FPL's own computed rating at
the time, not a result-derived proxy. Using it directly, rather than
inventing a strength-based approximation, was a specific instruction after
checking the archive actually has it (see the session that built this).

**News agent has no seed here, deliberately**: `chance_of_playing_this_round`
is a live, forward-looking field the archive has no historical equivalent
for, and deriving one from that gameweek's own outcome (e.g. inferring
injury status from minutes actually played) would leak the outcome into a
pre-match signal - the exact no-lookahead trap this project has already
caught elsewhere. The News agent contributes nothing during backtest
(``backtest/agents_bridge.py`` returns an empty AgentArgument for it) - a
named, honest limitation of the historical backtest, not an approximation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from agents.stats.data import REPO_ROOT, load_gameweeks
from ingestion.db import get_connection

ARCHIVE_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def _cached_csv(season: str, filename: str) -> Path:
    """Downloads once, then reuses the local copy - same "fetch specific
    files via raw.githubusercontent.com" approach TODO.md's Phase 0 already
    established for ``merged_gws_<season>.csv``.
    """
    path = REPO_ROOT / f"{filename}_{season}.csv"
    if not path.exists():
        resp = requests.get(f"{ARCHIVE_BASE_URL}/{season}/{filename}.csv", timeout=30)
        resp.raise_for_status()
        path.write_text(resp.text, encoding="utf-8")
    return path


def schema_name(season: str) -> str:
    return f"backtest_{season.replace('-', '_')}"


def seed_backtest_schema(season: str = "2025-26", *, merged_gws_path: Path | None = None) -> str:
    """Seeds ``backtest_<season>`` with this season's teams, fixtures,
    players, and player_gameweek_stats from the archive. Idempotent - reruns
    upsert rather than duplicate (every insert below is ``ON CONFLICT DO
    UPDATE``), so re-seeding after a partial failure is always safe.

    Returns the schema name, for the caller to pass to
    ``ingestion.db.get_connection(schema=...)``.
    """
    now = datetime.now(timezone.utc).isoformat()
    schema = schema_name(season)
    conn = get_connection(schema=schema)

    teams_df = pd.read_csv(_cached_csv(season, "teams"))
    conn.executemany(
        """INSERT INTO teams (team_id, name, short_name, strength, updated_at)
           VALUES (%s,%s,%s,%s,%s)
           ON CONFLICT (team_id) DO UPDATE SET name=EXCLUDED.name, short_name=EXCLUDED.short_name,
               strength=EXCLUDED.strength, updated_at=EXCLUDED.updated_at""",
        [(int(r.id), r.name, r.short_name, int(r.strength), now) for r in teams_df.itertuples()],
    )
    team_id_by_name = dict(zip(teams_df["name"], teams_df["id"]))

    fixtures_df = pd.read_csv(_cached_csv(season, "fixtures"))
    conn.executemany(
        """INSERT INTO fixtures
           (fixture_id, gw, team_h, team_a, team_h_score, team_a_score, kickoff_time, finished,
            team_h_difficulty, team_a_difficulty, updated_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (fixture_id) DO UPDATE SET gw=EXCLUDED.gw, finished=EXCLUDED.finished,
               team_h_score=EXCLUDED.team_h_score, team_a_score=EXCLUDED.team_a_score,
               updated_at=EXCLUDED.updated_at""",
        [
            (
                int(r.id), int(r.event) if pd.notna(r.event) else None, int(r.team_h), int(r.team_a),
                int(r.team_h_score) if pd.notna(r.team_h_score) else None,
                int(r.team_a_score) if pd.notna(r.team_a_score) else None,
                r.kickoff_time, int(bool(r.finished)),
                int(r.team_h_difficulty) if pd.notna(r.team_h_difficulty) else None,
                int(r.team_a_difficulty) if pd.notna(r.team_a_difficulty) else None,
                now,
            )
            for r in fixtures_df.itertuples()
            if pd.notna(r.event)  # postponed/unscheduled fixtures carry no gameweek - not usable
        ],
    )

    gws = load_gameweeks(merged_gws_path) if merged_gws_path else load_gameweeks()
    gws["team_id"] = gws["team"].map(team_id_by_name)
    if gws["team_id"].isna().any():
        unresolved = sorted(gws.loc[gws["team_id"].isna(), "team"].unique())
        raise ValueError(f"backtest_seed: team name(s) in the archive don't match teams.csv: {unresolved}")

    players_df = gws.sort_values("GW").drop_duplicates("player_id", keep="last")
    conn.executemany(
        """INSERT INTO players (player_id, web_name, team_id, position, updated_at)
           VALUES (%s,%s,%s,%s,%s)
           ON CONFLICT (player_id) DO UPDATE SET web_name=EXCLUDED.web_name, team_id=EXCLUDED.team_id,
               position=EXCLUDED.position, updated_at=EXCLUDED.updated_at""",
        [(int(r.player_id), r.name, int(r.team_id), r.position, now) for r in players_df.itertuples()],
    )

    stat_cols = [
        "minutes", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves",
        "bonus", "bps", "influence", "creativity", "threat", "ict_index", "starts",
        "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    ]
    rows = []
    for r in gws.itertuples():
        rows.append(
            (
                int(r.player_id), int(r.GW),
                *[getattr(r, c) if pd.notna(getattr(r, c)) else None for c in stat_cols],
                int(r.team_id), int(r.opponent_team) if pd.notna(r.opponent_team) else None,
                int(bool(r.was_home)) if pd.notna(r.was_home) else None, int(r.n_fixtures),
                float(r.price) if pd.notna(r.price) else None,
                float(r.selected) if pd.notna(r.selected) else None,
                int(r.transfers_balance) if pd.notna(r.transfers_balance) else None,
                "vaastav_archive", now,
            )
        )
    columns = "player_id, gw, " + ", ".join(stat_cols) + ", team_id, opponent_team_id, was_home, n_fixtures, price, selected, transfers_balance, source, ingested_at"
    placeholders = ", ".join(["%s"] * (len(stat_cols) + 11))
    update_cols = [c for c in ("team_id", "opponent_team_id", "was_home", "n_fixtures", "price", "selected", "transfers_balance", *stat_cols)]
    update_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
    conn.executemany(
        f"""INSERT INTO player_gameweek_stats ({columns}) VALUES ({placeholders})
            ON CONFLICT (player_id, gw) DO UPDATE SET {update_clause}""",
        rows,
    )

    conn.commit()
    return schema
