# CLAUDE.md

Project memory for Claude Code. Keep this file short — see `README.md` for what this is, `ARCHITECTURE.md` for how it fits together, `TODO.md` for current build phase. Don't duplicate their content here; only what changes how you should work.

## What this is

Multi-agent system managing a real Fantasy Premier League squad. LangGraph orchestrates seven agents (six specialists + one manager), each its own containerized service, plus a React/GitHub Pages frontend that reads static exports. Two modes: backtest (historical, offline, auto-accepts proposals) and live weekly (real API, human-in-the-loop approval via the exported site).

## Hard constraints — do not violate

- **No lookahead in backtest data slicing.** Any function feeding data to an agent during backtest must only see data available before that gameweek's deadline. If you're unsure whether a column could leak future information (e.g. `xP`, next-gameweek-derived stats), flag it rather than assuming it's safe.
- **Only the News agent's `vetoes` are a hard constraint on the Manager — its `recommendations` are a normal vote.** The Manager must never select a player the News agent has flagged as unavailable in `vetoes`, regardless of predicted points. Its `recommendations` (notable positive coverage) get weighed like any other specialist's, not treated as authoritative.
- **Manager aggregation weights are fixed and config-driven, never decided by an LLM per run.** The same inputs must always produce the same squad — this is what makes the backtest and its ablations meaningful. If a task seems to call for the LLM adjusting how much to trust an agent, that's a sign to stop and flag it, not implement it.
- **The reaction round is text-only and decision-inert.** Agents may see each other's first-round outputs and write a reaction, but that reaction must never alter `conviction`, `recommendations`, or `vetoes` — it exists for transcript richness, not to reopen the decision.
- **No auto-apply to the live FPL account.** The system recommends; a human approves and applies manually. Don't wire up code that writes to the real FPL account without an explicit separate task asking for it.
- **RAG-retrieved text must pass through Prompt Shields (document-attack detection) before reaching any agent's context.** This applies to anything pulled from external injury/team-news sources.
- **Microsoft Foundry is the model + guardrail layer only.** Don't introduce Foundry's own Agent Service as a second orchestrator — LangGraph owns the state machine.
- **The frontend never talks to Postgres or any backend, directly or indirectly.** It only reads static JSON files exported by the pipeline. This is what makes GitHub Pages hosting safe with no public API surface — don't reintroduce a live backend call to "simplify" something later.

## Repo layout (target)

```
agents/            # one subdir per specialist service (stats/, fixtures/, news/, contrarian/, template/, chips/)
manager/            # aggregation, weighting, solver call, captain selection, narration
solver/             # PuLP squad-selection service
orchestrator/       # the LangGraph graph definition + Postgres checkpointer
ingestion/          # data pulls: FPL API (live), GitHub archive (backtest)
frontend-export/    # exports pending/final gameweek state as static JSON, commits to the frontend repo
frontend/           # React app, deployed to GitHub Pages, reads only the exported static JSON
shared/             # contracts.py (the shared agent response schema), agent_service.py (Foundry call + fallback + FastAPI scaffolding)
k8s/                # manifests: namespace, CronJob, Deployments, ConfigMaps/Secrets -- local Docker Desktop Kubernetes, not cloud
docs/               # dataset schemas, notebooks (EDA, model comparison, SHAP), design writeups
```

## Conventions

- Python 3.11+, type hints on public functions.
- Dependencies: per-package `requirements.txt` (not a monorepo-wide `pyproject.toml`) — each service is separately containerized, so a shared manifest would mix unrelated services' deps. Decided and documented during the first repo cleanup pass.
- Each specialist service is a small FastAPI app with one clear endpoint; keep them thin.
- Tests live next to the code they test (`test_*.py`), not in a parallel tree.

## Commands

Fill in as they're established in Phase 0/1 — placeholders below, update once real:
- Run backtest: `TBD`
- Run tests: `TBD`
- Local multi-service dev: `docker compose up`
- Apply k8s manifests locally: `kubectl apply -f k8s/ --context kind-fpl-agents`

## Current phase

See `TODO.md` for the authoritative checklist. Check it at the start of a session rather than assuming which phase is active.
