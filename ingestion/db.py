"""Connection + schema for the bronze/silver store.

Backed by Postgres (``DATABASE_URL`` in ``.env``) - this used to be sqlite,
with a ponytail note here explaining that as a deliberate stand-in until a
real Postgres instance existed. One now does, so this module was migrated
onto it. What actually changed, migrating from the previous sqlite version:

  * the connection helper now opens a psycopg2 connection instead of a
    sqlite3 one;
  * ``bronze_responses.params``/``.payload`` and
    ``my_team_state.picks``/``.transfers`` are ``JSONB`` instead of ``TEXT``
    (Postgres can index and query into these directly; sqlite had no JSON
    type, so those columns were plain text holding a JSON string);
  * ``bronze_responses.id`` is ``BIGSERIAL`` instead of
    ``INTEGER ... AUTOINCREMENT`` (sqlite's autoincrement spelling).

Everything else - every ``INSERT ... ON CONFLICT ... DO UPDATE`` upsert in
bronze.py/silver.py, every column name and type otherwise - is unchanged:
that SQL is already ANSI/Postgres-compatible, which is exactly why the
original sqlite version called this migration "mostly a connection swap".

The one thing that ISN'T just a connection swap: sqlite3's paramstyle is
``?`` (qmark); psycopg2's is ``%s`` (pyformat). Every parameterized query in
bronze.py/silver.py had its placeholders updated for that reason alone - a
mechanical dialect difference, not a behaviour change.

``Connection`` below is a thin subclass adding sqlite3.Connection-style
``.execute()``/``.executemany()`` convenience methods directly on the
connection object (psycopg2's connection only exposes `.cursor()`) - that
shim is what let every other call site in this package keep calling
``conn.execute(...)`` unchanged, exactly as it did against sqlite.
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extras
from psycopg2 import sql as pgsql

try:  # load .env locally (DATABASE_URL, FPL_TEAM_ID); harmless if
    # python-dotenv isn't installed in prod - matches agents/stats/service.py
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class Connection(psycopg2.extensions.connection):
    """psycopg2 connection with sqlite3.Connection-style convenience methods."""

    def execute(self, query: str, params=None):
        cur = self.cursor()
        cur.execute(query, params)
        return cur

    def executemany(self, query: str, seq_of_params):
        cur = self.cursor()
        cur.executemany(query, seq_of_params)
        return cur


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# bronze_responses is intentionally APPEND-ONLY (no upsert, no natural key
# beyond the identity id) - it is the replay/debug log, so every raw pull is
# kept, including repeats. Every silver table below it, by contrast, is
# upserted on its natural key so re-running ingestion for an already-ingested
# gameweek replaces rows instead of duplicating them (see bronze.py / silver.py).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS bronze_responses (
    id BIGSERIAL PRIMARY KEY,
    endpoint TEXT NOT NULL,
    params JSONB NOT NULL,
    payload JSONB NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bronze_endpoint ON bronze_responses(endpoint, ingested_at);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT,
    strength INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    web_name TEXT NOT NULL,
    first_name TEXT,
    second_name TEXT,
    team_id INTEGER,
    position TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gameweeks (
    gw INTEGER PRIMARY KEY,
    name TEXT,
    deadline_time TEXT,
    finished INTEGER NOT NULL DEFAULT 0,
    is_current INTEGER NOT NULL DEFAULT 0,
    average_entry_score REAL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id INTEGER PRIMARY KEY,
    gw INTEGER,
    team_h INTEGER,
    team_a INTEGER,
    team_h_score INTEGER,
    team_a_score INTEGER,
    kickoff_time TEXT,
    finished INTEGER NOT NULL DEFAULT 0,
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    updated_at TEXT NOT NULL
);

-- The core accumulating fact table: one row per player per gameweek.
-- `price`, `selected` and `transfers_balance` are best-effort - the FPL API
-- has no historical-per-gameweek view of them for a bulk backfill (see
-- backfill.py). They are NULL when unknown; features.engineering already
-- treats missing values as "no information yet" (fillna(0.0)), so a NULL
-- here never crashes the feature pipeline, it just means slightly less
-- signal for that row.
CREATE TABLE IF NOT EXISTS player_gameweek_stats (
    player_id INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    minutes INTEGER, total_points INTEGER, goals_scored INTEGER, assists INTEGER,
    clean_sheets INTEGER, goals_conceded INTEGER, own_goals INTEGER,
    penalties_saved INTEGER, penalties_missed INTEGER, yellow_cards INTEGER,
    red_cards INTEGER, saves INTEGER, bonus INTEGER, bps INTEGER,
    influence REAL, creativity REAL, threat REAL, ict_index REAL, starts INTEGER,
    expected_goals REAL, expected_assists REAL, expected_goal_involvements REAL,
    expected_goals_conceded REAL,
    team_id INTEGER, opponent_team_id INTEGER, was_home INTEGER, n_fixtures INTEGER,
    price REAL, selected REAL, transfers_balance INTEGER,
    source TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (player_id, gw)
);

-- Our own squad/bank/transfer state per gameweek - the one thing no
-- third-party dataset has, since it's account-specific.
CREATE TABLE IF NOT EXISTS my_team_state (
    gw INTEGER PRIMARY KEY,
    bank REAL, team_value REAL, total_points INTEGER, overall_rank INTEGER,
    event_transfers INTEGER, event_transfers_cost INTEGER, points_on_bench INTEGER,
    active_chip TEXT,
    picks JSONB,      -- list of {player_id, squad_position, multiplier, is_captain, is_vice_captain}
    transfers JSONB,  -- list of {player_in, player_out, cost, time}, all transfers to date
    updated_at TEXT NOT NULL
);

-- Every chip played this season, from the FPL API's authoritative
-- entry/{id}/history/ endpoint (ingestion/silver.py::upsert_chip_usage) -
-- NOT derived from my_team_state.active_chip above, which only reflects
-- whichever gameweek happened to be current each time the weekly job ran
-- and so has real gaps if that job started mid-season or missed a week.
-- The Chips agent (agents/chips/) reads this table to know which chips are
-- still available, not my_team_state. (chip, gw) is the natural key since
-- the same chip type can legitimately be played twice in a season (FPL
-- grants one of each chip per season half) but never twice in one gameweek.
CREATE TABLE IF NOT EXISTS chip_usage (
    chip TEXT NOT NULL,        -- FPL's own raw chip code: "wildcard", "3xc", "bboost", "freehit"
    gw INTEGER NOT NULL,
    played_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (chip, gw)
);

-- Stats agent model lifecycle (docs/monitoring.md documents this loop in
-- full). Append-only, like bronze_responses - a prediction is a point-in-
-- time record of what the model said, not something later rows replace.
CREATE TABLE IF NOT EXISTS predictions_log (
    id BIGSERIAL PRIMARY KEY,
    player_id INTEGER NOT NULL,
    gw INTEGER NOT NULL,
    predicted_points REAL NOT NULL,
    -- Which stats_model.pkl (+ hyperparameters) made this prediction - see
    -- train.py::_model_version. This is what lets the monitoring job tell a
    -- pre-retrain prediction apart from a post-retrain one instead of
    -- blending both into one rolling error number.
    model_version TEXT NOT NULL,
    logged_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_log_gw ON predictions_log(gw);
CREATE INDEX IF NOT EXISTS idx_predictions_log_model_version ON predictions_log(model_version);

-- One row per monitoring job execution (agents/stats/monitor.py), so the
-- rolling-RMSE-vs-baseline comparison and every retrain decision is
-- queryable after the fact, not just visible in a log line at the time.
CREATE TABLE IF NOT EXISTS monitoring_runs (
    id BIGSERIAL PRIMARY KEY,
    evaluated_through_gw INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    window_gws INTEGER NOT NULL,
    gws_used INTEGER NOT NULL,      -- how many gameweeks actually went into rolling_rmse (can be < window_gws just after a retrain)
    rolling_rmse REAL,
    baseline_rmse REAL NOT NULL,
    degrade_threshold_pct REAL NOT NULL,
    is_degraded BOOLEAN NOT NULL,
    consecutive_degraded_gws INTEGER NOT NULL,
    retrain_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    run_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_monitoring_runs_gw ON monitoring_runs(evaluated_through_gw);

-- One row per retrain event - fired automatically by monitor.py when the
-- degradation trigger sustains, or run manually. Always written, even when
-- the retrained model doesn't end up replacing the deployed one, so "we
-- tried and it didn't help" is as visible as "we tried and it worked".
CREATE TABLE IF NOT EXISTS retrain_log (
    id BIGSERIAL PRIMARY KEY,
    triggered_at TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    old_model_version TEXT,
    old_metrics JSONB,
    new_model_version TEXT,
    new_metrics JSONB,
    deployed BOOLEAN NOT NULL,   -- did the retrained model actually replace the deployed one
    notes TEXT
);
"""

# ---------------------------------------------------------------------------
# News agent's RAG corpus (pgvector) - kept OUT of _SCHEMA above deliberately.
# ---------------------------------------------------------------------------
# _SCHEMA runs unconditionally on every get_connection() call across the
# whole project (ingestion, Stats agent monitoring, tests...). The vector
# store depends on the `pgvector` Postgres EXTENSION, which is a native
# binary that has to be installed on the Postgres server itself - it is not
# guaranteed to be present (e.g. a fresh local install doesn't have it by
# default). If this table were folded into _SCHEMA, every single connection
# anywhere in the project would start failing the moment pgvector isn't
# installed, over one optional feature. So this is its own function,
# called explicitly only by the News agent's own code (ingest.py,
# retrieval.py), which is the only code that actually needs it.
_NEWS_VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_passages (
    id BIGSERIAL PRIMARY KEY,
    -- Which of the two scraped corpora this passage came from (see
    -- ARCHITECTURE.md section 4) - kept distinguishable at retrieval time so
    -- a caller can ask for availability-relevant passages specifically vs.
    -- general coverage, rather than one undifferentiated pool.
    corpus TEXT NOT NULL CHECK (corpus IN ('team_news', 'general_news')),
    source_url TEXT NOT NULL,
    headline TEXT,
    published_at TEXT,
    chunk_text TEXT NOT NULL,
    -- text-embedding-3-small's native dimensionality.
    embedding VECTOR(1536),
    scraped_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_passages_corpus ON news_passages(corpus);

-- Entity linking, done at ingestion time (not left to semantic search alone
-- at query time - see agents/news/entity_linking.py for why common surnames
-- make that unreliable). One passage can mention more than one player.
CREATE TABLE IF NOT EXISTS news_passage_players (
    passage_id BIGINT NOT NULL REFERENCES news_passages(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL,
    PRIMARY KEY (passage_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_news_passage_players_player ON news_passage_players(player_id);
"""


def register_pgvector_adapter(conn: Connection) -> None:
    """Teach psycopg2 how to send/receive pgvector's ``vector`` type.

    Without this, psycopg2's default adapter turns a Python list into a
    Postgres ARRAY literal (``{1,2,3}``), not pgvector's expected
    ``[1,2,3]`` syntax - embeddings would silently fail to compare/insert
    correctly. Must be called once per connection that touches the
    ``embedding`` column (``ensure_pgvector_schema`` does it automatically
    after confirming the extension exists; call it directly on a connection
    that already knows the extension is there, e.g. a long-lived service
    connection, without re-running schema creation).
    """
    from pgvector.psycopg2 import register_vector

    register_vector(conn)


def ensure_pgvector_schema(conn: Connection) -> bool:
    """Create the pgvector extension + the News agent's vector-backed tables,
    if the extension is actually available on this Postgres instance.

    Returns True if the schema is ready to use, False if pgvector isn't
    installed (the CREATE EXTENSION call itself fails) - callers should
    check this and fail with a clear, actionable message rather than a
    cryptic "type vector does not exist" error the first time they try to
    insert a row.
    """
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception:
        conn.rollback()
        return False
    conn.execute(_NEWS_VECTOR_SCHEMA)
    conn.commit()
    register_pgvector_adapter(conn)
    return True


def get_connection(dsn: str | None = None, *, schema: str | None = None) -> Connection:
    """Open a connection to the bronze/silver Postgres store.

    ``dsn`` defaults to ``$DATABASE_URL``. ``schema`` is optional and mainly
    for tests: passing e.g. ``schema="test_ingestion"`` creates (if needed)
    and switches to an isolated schema via ``search_path``, so test runs
    never touch the ``public`` schema real ingestion writes to. Production
    code calls this with no ``schema`` and gets the connection's normal
    default (``public``).

    Safe to call repeatedly - schema/table creation is idempotent (``IF NOT
    EXISTS`` everywhere), matching the idempotency the rest of this layer
    promises for the weekly CronJob.

    Also safe to call CONCURRENTLY (e.g. several services' containers all
    starting at once against a brand-new, empty database - found via a
    genuine docker-compose-up-from-scratch run, not something the long-
    lived local dev Postgres ever exercised): Postgres's own "IF NOT
    EXISTS" only checks-then-creates, which is NOT atomic across sessions -
    two connections can both see "doesn't exist yet" and both try to
    create it, raising a UniqueViolation on pg_type. The advisory lock
    below serializes just the schema-creation step across connections so
    concurrent startups queue instead of racing.
    """
    dsn = dsn or os.environ["DATABASE_URL"]
    conn = psycopg2.connect(
        dsn, connection_factory=Connection, cursor_factory=psycopg2.extras.DictCursor
    )
    if schema:
        with conn.cursor() as cur:
            cur.execute(pgsql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(pgsql.Identifier(schema)))
            cur.execute(pgsql.SQL("SET search_path TO {}").format(pgsql.Identifier(schema)))
        conn.commit()
    with conn.cursor() as cur:
        # Arbitrary constant, just needs to be the same across every caller -
        # this lock has no meaning beyond "one schema-creation at a time".
        # The commit MUST happen before the unlock, not after - releasing
        # the lock first would let the next waiting session start checking
        # "IF NOT EXISTS" against this transaction's NOT-YET-committed
        # changes, racing all over again just with a narrower window
        # (caught by testing this fix against a truly fresh database
        # instead of trusting the lock alone to be correct).
        cur.execute("SELECT pg_advisory_lock(727100)")
        try:
            cur.execute(_SCHEMA)
            conn.commit()
        finally:
            cur.execute("SELECT pg_advisory_unlock(727100)")
    return conn
