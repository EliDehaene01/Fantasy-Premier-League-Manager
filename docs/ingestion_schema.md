# Ingestion pipeline: bronze / silver / gold

The self-built FPL API ingestion pipeline (`ingestion/`), and where it sits
next to the training data pipeline (`agents/stats/`, `features/`).

## Why this is self-built

A third-party dataset (FPL-Core-Insights was the one considered) was set
aside in favour of building this in-house, for two reasons:

1. **It's the point of the exercise.** This project exists to demonstrate
   real data-engineering work (API integration, idempotent upserts, a
   layered bronze/silver/gold pipeline, a CronJob-shaped entrypoint), not to
   assemble someone else's ready-made table. Depending on a third-party
   dataset here would outsource exactly the part of the project meant to
   show that skill.
2. **No dependency on someone else's pipeline staying maintained.** A
   community dataset can go stale, change shape, or disappear; the official
   FPL API is the primary source everyone else's dataset is built from
   anyway, so pulling from it directly removes a layer of risk.

## The three layers

```
FPL API  ->  bronze (raw JSON, replay log)  ->  silver (relational tables)  ->  gold (feature-ready)
```

### Bronze - `ingestion/bronze.py`, `ingestion/fpl_client.py`

Every API response is landed **exactly as received** into one table,
`bronze_responses` (`ingestion/db.py`):

| column | meaning |
|---|---|
| `endpoint` | which API call this was (`bootstrap-static`, `fixtures`, `event-live`, `element-summary`, `entry`, `entry-picks`, `entry-transfers`) |
| `params` | request parameters, as `JSONB` (e.g. `{"gw": 5}`) |
| `payload` | the raw JSON response body, untouched, as `JSONB` |
| `ingested_at` | UTC timestamp of the pull |

Bronze is **append-only** - no upsert, no natural key beyond an
autoincrement id. Every pull is kept, including repeats, because this table
exists purely as a replay/debug log: if a bug is later found in how silver
parses a response, the fix is "reprocess bronze", not "the original data is
gone, wait for the API to give it back."

`ingestion/fpl_client.py` is the only place that ever calls the live API -
it sets a descriptive `User-Agent`, sleeps briefly before every request, and
times out after 15s, since this is a public but undocumented API with no
published rate limit or SLA to lean on.

### Silver - `ingestion/silver.py`

Bronze JSON, parsed into relational tables and **upserted** on each table's
natural key, so re-running ingestion for a gameweek that's already there
replaces rows instead of duplicating them (this is what makes the weekly
CronJob safe to re-run, retry, or overlap with itself):

| table | natural key | what it holds |
|---|---|---|
| `teams` | `team_id` | the 20 clubs |
| `players` | `player_id` | every player: name, team, position |
| `gameweeks` | `gw` | deadline, finished/current flags, average score |
| `fixtures` | `fixture_id` | every match: teams, kickoff, score, FDR |
| `player_gameweek_stats` | (`player_id`, `gw`) | **the core accumulating fact table** - one row per player per gameweek |
| `my_team_state` | `gw` | our own squad/bank/transfer state that gameweek - the one thing no third-party dataset has |
| `chip_usage` | (`chip`, `gw`) | every chip played this season - see below, added for the Chips agent |

`player_gameweek_stats` is built from `event/{gw}/live/` - one API call per
gameweek covering every player, rather than looping `element-summary` per
player (~700 calls). The trade-off: `event/live` doesn't carry price,
ownership, or transfer-balance for that historical gameweek, so those three
columns are `NULL` for backfilled rows. `weekly.py` fills them in two ways:
current price from a fresh `bootstrap-static` pull (a same-day approximation,
fine for "the gameweek that just finished"), and precise historical price /
ownership / transfer-balance for **our own squad only** (~15 players, via
`element-summary`, bounded and cheap - see `silver.refine_player_gameweek_from_element_summary`).

### Chip usage - a real gap found and fixed while building the Chips agent

The Chips agent needs to know which of the four chips (wildcard, bench
boost, triple captain, free hit) are still available this season. The
obvious first place to look, `my_team_state.active_chip`, turned out not
to answer that reliably: it's populated per-gameweek from
`entry/{id}/event/{gw}/picks/`, but `my_team_state` itself is only ever
written by `weekly.py` (never backfilled), so `active_chip` only reflects
whichever gameweek happened to be current each time the weekly job
actually ran - a real hole if that job started mid-season or missed a
week.

The fix: `chip_usage` is populated from a new bronze pull,
`bronze.fetch_and_land_entry_history` / `fpl_client.get_entry_history`
(`entry/{id}/history/`), FPL's own authoritative record of every chip
played this season (`chips: [{name, time, event}, ...]`) - complete
regardless of ingestion history. `silver.upsert_chip_usage` upserts it on
`(chip, gw)`, called from `weekly.py` alongside the existing entry/
transfers pulls. `agents/chips/availability.py` reads this table, not
`active_chip`, to compute which chips are still available (a flat cap of 2
uses per chip per season - see that module's docstring for why the exact
half-season eligibility boundary isn't modeled).

### Gold - `features/engineering.py`, read via `ingestion/silver.load_gameweek_frame`

"Gold" isn't a separate table - it's the output of the shared feature
engineering module (rolling form, fixture adjustment, per-90 normalisation,
price momentum, cold-start handling) run on top of silver. The same module
is imported by both the offline training script (fed the frozen archive) and
anything reading live silver data, so training and serving can never
silently drift apart. See `features/engineering.py`'s docstring for the
common "gameweek frame" shape both paths produce before this module ever
sees the data.

## Backfill vs. weekly

- **`ingestion/backfill.py`** populates silver for every finished gameweek of
  the **current** season, gameweek 1 through whatever's most recently
  finished, one `event/{gw}/live` call per gameweek.
- **`ingestion/weekly.py`** is the same underlying bronze/silver code, scoped
  to just what's new: the gameweek that just finished, refreshed fixtures,
  and our own entry/picks/transfers state. This is what the Kubernetes
  CronJob (ARCHITECTURE.md section 8) actually runs on a schedule.

### Why backfill stops at the start of the *current* season

The FPL API has no gameweek-level view of a **completed** season. Once a
season ends, `element-summary/{id}`'s `history` (per-gameweek breakdown) is
cleared, and only `history_past` remains - one row per player per *season*
(season totals: total points, minutes, etc.), not per gameweek. There is no
endpoint that returns a finished season's `event/{gw}/live` after the fact.
So last season genuinely cannot be backfilled this way, at any cost in API
calls - the granular data simply isn't there any more.

The vaastav archive (`merged_gws_2025-26.csv`) remains the sole source for
completed seasons, frozen behind the already-trained model. The current
season is different: it's in progress, so every gameweek that has finished
is still available at full per-player granularity via `event/{gw}/live` -
which is exactly what `backfill_current_season` loops over, from gameweek 1
onward.

## Schema reconciliation - `data/reconcile_archive.py`

The vaastav archive and this project's silver schema are separate designs
with different column names and shapes. `data/reconcile_archive.py` maps
the archive's columns onto silver's `player_gameweek_stats` shape (documented
inline, including every column that has no clean equivalent on either side -
team/opponent/fixture ids in particular, since FPL's id numbering isn't
guaranteed stable across seasons). It does not write into the live store;
the archive stays frozen and separate, and this module exists so a future
retraining script can read both sources in one consistent shape.

## Storage: Postgres

The bronze/silver store is Postgres, connected to via `$DATABASE_URL` (set in
`.env`). This used to be sqlite - a deliberate stand-in documented in
`ingestion/db.py` while no Postgres instance existed yet - and was migrated
once one did. What actually changed in that migration:

  * the connection helper (`ingestion/db.py::get_connection`) opens a
    psycopg2 connection instead of a sqlite3 one, wrapped in a small
    `Connection` subclass that adds sqlite3-style `.execute()`/
    `.executemany()` methods directly on the connection object - this is
    what let every other call site in `bronze.py`/`silver.py` keep calling
    `conn.execute(...)` completely unchanged;
  * `bronze_responses.params`/`.payload` and `my_team_state.picks`/
    `.transfers` are `JSONB` now, not `TEXT` holding a JSON string - inserted
    via `psycopg2.extras.Json(...)` rather than `json.dumps(...)`, so
    Postgres receives them typed as JSON directly (confirmed working with a
    real `payload->'elements'->0->'stats'->>'total_points'`-style JSONB path
    query against real ingested data);
  * every SQL placeholder changed from sqlite's `?` to psycopg2's `%s` - a
    mechanical paramstyle difference, not a behaviour change. Every
    `INSERT ... ON CONFLICT ... DO UPDATE` upsert, every column name and
    type, was otherwise unchanged - that SQL was already Postgres-compatible.

**One real incompatibility surfaced by the migration, worth knowing about:**
`sqlite3.Row.keys()` returns a plain list; `psycopg2.extras.DictRow.keys()`
(the row type `get_connection` now uses, chosen because - like
`sqlite3.Row` - it supports both `row[0]` and `row["col"]`) returns a
one-shot `OrderedDict` key iterator. Comparing two of those with `==`
directly is always `False` (object identity, not content) regardless of
whether the columns actually match, which broke one existing test
(`test_backfill_vs_weekly_same_row_shape`) until its comparison was changed
to `list(row.keys()) == list(row.keys())`. That's a fix to a genuine
driver-level API difference, not a loosened assertion - see the note in
`ingestion/test_ingestion.py` for the full explanation, including the other
fixture-level (not assertion-level) adaptation the test suite needed: no
`:memory:` equivalent exists in Postgres, so tests run against an isolated
`test_ingestion` schema (truncated before each test) rather than the real
`public` schema production ingestion writes to.

```
python -m ingestion backfill              # GW1 through the latest finished GW
python -m ingestion backfill --upto-gw 5  # a specific range
python -m ingestion weekly                 # the newly-finished GW + fixtures + our entry state
```

Both default to `$DATABASE_URL`; pass `--database-url` to point elsewhere
(e.g. a different environment).
