"""Tests for the Chips agent service.

Uses an isolated Postgres schema (CHIPS_AGENT_TEST_SCHEMA - same mechanism
ingestion's own test suite and the Fixtures/Contrarian/Template agents'
tests use) seeded with two clearly separated synthetic scenarios, both real
squads at real gameweeks against real fixture rows (no mocking of the
scoring logic itself):

  * GW10 ("double gameweek" squad): 3 of 5 squad players' teams each play
    TWICE in GW10 - enough to clear BENCH_BOOST_MIN_DOUBLE_PLAYERS (3), so
    bench_boost should fire.
  * GW20 ("ordinary" squad): 5 squad players, each with exactly one
    average-difficulty (FDR 3) fixture every gameweek across the whole
    3-gameweek window - no doubles, no blanks, no favorable captain
    fixture, and a neutral fixture-favorability average - nothing should
    clear any chip's bar, so the verdict must be chip=None.

Neither the `teams` nor `predictions_log` table needs seeding: fixture
lookups (agents/fixtures/scoring.py, reused here) key off `players.team_id`
and `fixtures.team_h`/`team_a` directly, and the ordinary-gameweek scenario
deliberately has no logged prediction, so Triple Captain has nothing to
short-circuit on either.
"""

from __future__ import annotations

import os

os.environ["CHIPS_AGENT_DISABLE_LLM"] = "1"
os.environ["CHIPS_AGENT_TEST_SCHEMA"] = "test_chips_agent"

import pytest
from fastapi.testclient import TestClient
from psycopg2.extras import Json

from agents.chips.service import app
from ingestion.db import get_connection

SCHEMA = "test_chips_agent"


def _picks(player_ids: list[int]) -> list[dict]:
    return [
        {"element": pid, "position": i + 1, "multiplier": 1, "is_captain": i == 0, "is_vice_captain": i == 1}
        for i, pid in enumerate(player_ids)
    ]


@pytest.fixture(scope="module")
def seeded_conn():
    conn = get_connection(schema=SCHEMA)
    conn.execute("TRUNCATE players, fixtures, my_team_state, chip_usage, predictions_log")
    now = "2026-01-01T00:00:00Z"

    # --- GW10 squad: 3 of 5 players' teams have a double gameweek -------
    double_squad = [601, 602, 603, 604, 605]
    conn.executemany(
        "INSERT INTO players (player_id, web_name, team_id, position, updated_at) VALUES (%s,%s,%s,%s,%s)",
        [(pid, f"Player {pid}", pid, "MID", now) for pid in double_squad],  # team_id == player_id, 1:1 for simplicity
    )
    fixture_rows = []
    fid = 90000
    # players 601, 602, 603: double gameweek 10 (two fixtures each)
    for team_id, opp_a, opp_b in [(601, 611, 612), (602, 613, 614), (603, 615, 616)]:
        fid += 1
        fixture_rows.append((fid, 10, team_id, opp_a, 2, 4, 0, now))
        fid += 1
        fixture_rows.append((fid, 10, opp_b, team_id, 4, 3, 0, now))
    # players 604, 605: a single, ordinary GW10 fixture each
    for team_id, opp in [(604, 617), (605, 618)]:
        fid += 1
        fixture_rows.append((fid, 10, team_id, opp, 3, 3, 0, now))
    conn.executemany(
        """INSERT INTO fixtures
           (fixture_id, gw, team_h, team_a, team_h_difficulty, team_a_difficulty, finished, updated_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        fixture_rows,
    )
    conn.execute(
        "INSERT INTO my_team_state (gw, picks, updated_at) VALUES (%s,%s,%s)",
        (10, Json(_picks(double_squad)), now),
    )

    # --- GW20 squad: an ordinary run, no unusual fixture shape at all ----
    ordinary_squad = [701, 702, 703, 704, 705]
    conn.executemany(
        "INSERT INTO players (player_id, web_name, team_id, position, updated_at) VALUES (%s,%s,%s,%s,%s)",
        [(pid, f"Player {pid}", pid, "MID", now) for pid in ordinary_squad],
    )
    fixture_rows = []
    for team_id in ordinary_squad:
        for gw in (20, 21, 22):
            fid += 1
            opp = team_id + 100
            fixture_rows.append((fid, gw, team_id, opp, 3, 3, 0, now))  # FDR 3 every week - neutral
    conn.executemany(
        """INSERT INTO fixtures
           (fixture_id, gw, team_h, team_a, team_h_difficulty, team_a_difficulty, finished, updated_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        fixture_rows,
    )
    conn.execute(
        "INSERT INTO my_team_state (gw, picks, updated_at) VALUES (%s,%s,%s)",
        (20, Json(_picks(ordinary_squad)), now),
    )

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def client(seeded_conn):
    with TestClient(app) as c:
        yield c


def test_double_gameweek_squad_triggers_bench_boost(client):
    resp = client.post("/argue", json={"gameweek": 10, "players": []})
    assert resp.status_code == 200
    body = resp.json()

    assert body["chip_recommendation"]["chip"] == "bench_boost"
    assert 0.0 < body["chip_recommendation"]["confidence"] <= 1.0
    assert "double gameweek" in body["chip_recommendation"]["reasoning"]


def test_ordinary_gameweek_returns_no_chip(client):
    resp = client.post("/argue", json={"gameweek": 20, "players": []})
    assert resp.status_code == 200
    body = resp.json()

    assert body["chip_recommendation"]["chip"] is None
    # "don't play anything" is still a real, confident verdict, not an empty one.
    assert body["chip_recommendation"]["confidence"] > 0.0
    assert body["chip_recommendation"]["reasoning"]


def test_argue_returns_contract_shape(client):
    resp = client.post("/argue", json={"gameweek": 20, "players": []})
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"agent", "recommendations", "vetoes", "reasoning", "chip_recommendation"}
    assert body["agent"] == "chips"
    # Chips argues for TIMING, not players - both stay empty on every response.
    assert body["recommendations"] == []
    assert body["vetoes"] == []
    assert isinstance(body["reasoning"], str) and body["reasoning"]

    chip_rec = body["chip_recommendation"]
    assert chip_rec is not None
    assert set(chip_rec) == {"chip", "confidence", "reasoning"}
    assert chip_rec["chip"] in ("wildcard", "bench_boost", "triple_captain", "free_hit", None)
    assert 0.0 <= chip_rec["confidence"] <= 1.0
    assert isinstance(chip_rec["reasoning"], str) and chip_rec["reasoning"]


def test_health_reports_db_connected(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["db_connected"] is True
