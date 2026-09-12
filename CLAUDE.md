# CLAUDE.md

Project memory for Claude Code. Keep this file short — see `README.md` for what this is, `ARCHITECTURE.md` for how it fits together, `TODO.md` for current build phase. Don't duplicate their content here; only what changes how you should work.

## What this is

Multi-agent system managing a real Fantasy Premier League squad. LangGraph orchestrates six agents (5 specialists + 1 manager), each its own containerized service. Two modes: backtest (historical, offline) and live weekly (real API, human-in-the-loop approval).

## Hard constraints — do not violate

- **No lookahead in backtest data slicing.** Any function feeding data to an agent during backtest must only see data available before that gameweek's deadline. If you're unsure whether a column could leak future information (e.g. `xP`, next-gameweek-derived stats), flag it rather than assuming it's safe.
- **Injuries agent output is a hard constraint on the Manager, not a vote.** The Manager must never select a player the Injuries agent has flagged as unavailable, regardless of predicted points.
- **No auto-apply to the live FPL account.** The system recommends; a human approves and applies manually. Don't wire up code that writes to the real FPL account without an explicit separate task asking for it.
- **RAG-retrieved text must pass through Prompt Shields (document-attack detection) before reaching any agent's context.** This applies to anything pulled from external injury/team-news sources.
- **Microsoft Foundry is the model + guardrail layer only.** Don't introduce Foundry's own Agent Service as a second orchestrator — LangGraph owns the state machine.

## Repo layout (target)

```
agents/            # one subdir per specialist service (stats/, fixtures/, injuries/, contrarian/, template/, chips/, manager/)
orchestrator/       # the LangGraph graph definition + Postgres checkpointer
solver/             # PuLP/OR-Tools squad-selection service
ingestion/          # data pulls: FPL API (live), GitHub archive (backtest)
notifier/           # recommend-and-confirm delivery
k8s/                # manifests: namespace, CronJob, Deployments, ConfigMaps/Secrets
docs/               # dataset schemas, notebooks (EDA, model comparison, SHAP)
```

## Conventions

- Python 3.11+, type hints on public functions.
- Formatting/linting: TBD in Phase 0 — check `pyproject.toml` once it exists rather than assuming a tool.
- Each specialist service is a small FastAPI app with one clear endpoint; keep them thin.
- Tests live next to the code they test (`test_*.py`), not in a parallel tree.

## Commands

Fill in as they're established in Phase 0/1 — placeholders below, update once real:
- Run backtest: `TBD`
- Run tests: `pytest ingestion features agents/stats` (ingestion tests need a running Postgres — see below)
- Ingestion (bronze/silver): `python -m ingestion backfill` / `python -m ingestion weekly`
- Local multi-service dev: `docker compose up`
- Apply k8s manifests locally: `kubectl apply -f k8s/ --context kind-fpl-agents`

## Data store

Postgres, local instance, connected to via `DATABASE_URL` in `.env` (not
committed — ask the user rather than guessing credentials if it's ever
missing). `ingestion/db.py` owns the connection helper and schema; see
`docs/ingestion_schema.md` for the bronze/silver table layout. Ingestion
tests run against an isolated `test_ingestion` schema on the same instance,
truncated before each test — they never touch the `public` schema real
ingestion writes to.

## Current phase

See `TODO.md` for the authoritative checklist. Check it at the start of a session rather than assuming which phase is active.
