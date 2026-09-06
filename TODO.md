# Build plan

## Phase 0 — Setup
- [ ] Repo scaffold (project structure, `pyproject.toml`/`requirements.txt`, linting, `CLAUDE.md`)
- [ ] Pull the 2025-26 (and prior season) `merged_gws.csv` files from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) (sparse-checkout or fetch specific files via raw.githubusercontent.com rather than cloning the whole multi-season repo); confirm gameweek coverage is complete; document schema in `/docs`
- [ ] Test the official FPL public API endpoints (`bootstrap-static`, `fixtures`, `entry/{team_id}`) — no registration or auth needed, it's fully public; find and note your own team's `entry_id` (visible in the URL once logged into fantasy.premierleague.com); set a descriptive `User-Agent` header and add basic caching/backoff since it's not an officially-documented developer product
- [ ] Create a Microsoft Foundry project (via ai.azure.com or the Azure Portal) in a resource group
- [ ] Deploy the model(s) each agent will call (consider Foundry's Model Router for cost/quality balancing across the simpler specialist agents vs. the Manager's synthesis step); note endpoint + key
- [ ] Enable Content Safety / Prompt Shields on the Foundry project (needed later in Phase 6, but easiest to turn on now alongside the rest of the resource setup)
- [ ] Decide on and set up the shared data store (Postgres) for normalized data, LangGraph checkpoints, and transcripts
- [ ] Local dev environment: Python env, Docker, a local Kubernetes cluster (kind or minikube) for testing manifests before any real cluster

## Phase 1 — Core agent logic

### Data science pipeline (Stats agent)
- [x] EDA on the historical dataset (form, minutes, price changes, injury frequency) — `agents/stats/eda.py` → `docs/eda_summary.md`
- [x] Feature engineering (the 4 required: rolling form, fixture-adjusted xP, per-90, price momentum; + 20 justified extras + position one-hots) — `agents/stats/features.py`
- [x] Time-respecting train/validation split (chronological GW6-29 train / GW30-38 validate, no random k-fold)
- [x] Train and compare five approaches: naive baseline, Poisson regression, Random Forest, XGBoost, small MLP — XGBoost wins (val MAE 0.958 vs naive 1.034); see `docs/model_comparison.md`
- [x] Add SHAP explainability on the winner (`models/stats_model.pkl`), dead-weight features called out in the write-up
- [x] Stats agent FastAPI service (`agents/stats/service.py`): `POST /argue`, model loaded once at startup, per-pick SHAP factors grounding the `reasoning` text, Foundry `gpt-5.4-nano` for the prose (deterministic fallback if the call fails), tests in `agents/stats/test_stats_agent.py`. Shared specialist-agent contract documented in `ARCHITECTURE.md` §2.

### RAG pipeline (Injuries agent)
- [ ] Source a corpus of injury/team-news text (scrape or API)
- [ ] Build chunking + embedding + vector store for retrieval
- [ ] Wire retrieval into the Injuries agent's context alongside structured availability fields

### Fine-tuning (Injuries agent)
- [ ] Label a small dataset for injury-severity/return-timeline classification
- [ ] Fine-tune a small classifier (or embedding model) for this task
- [ ] Evaluate against a zero-shot prompting baseline (precision/recall) and document the comparison

### Remaining agents
- [ ] Write the Fixtures agent (FDR scoring, blank/double gameweek detection)
- [ ] Write the Contrarian agent (ownership-vs-points logic)
- [ ] Write the Template agent (ownership/net-transfers logic)
- [ ] Write the Chips agent (chip-timing logic against the fixture calendar)
- [ ] Build the solver tool (PuLP/OR-Tools: budget, formation, per-club cap, transfer-hit cost)
- [ ] Write the Manager agent (aggregation, hard-constraint handling from Injuries, solver call, captain/vice logic)

## Phase 2 — Backtest engine
- [ ] Implement state tracking (squad, bank, free transfers, chips used) across gameweeks
- [ ] Implement the walk-forward loop with strict no-lookahead data slicing
- [ ] Stub/skip guardrail calls during backtest (no real external text, avoid unnecessary cost)
- [ ] Implement scoring against actual historical results, including transfer-hit and chip effects
- [ ] Implement benchmark comparisons (FPL average manager score, "never transfer" baseline)
- [ ] Implement per-agent ablation runs
- [ ] Run a full-season backtest and write up results

## Phase 3 — Orchestration (LangGraph)
- [ ] Define the typed state schema (squad, bank, free transfers, chips, current gameweek, debate arguments, solver result, approval status)
- [ ] Build the StateGraph: ingestion → specialists → manager → solver → approval node
- [ ] Add conditional edges (e.g. infeasible solver result routes back to the manager)
- [ ] Wire up a Postgres checkpointer for persistence across weekly runs
- [ ] Implement the human-approval `interrupt()` / `Command(resume=...)` flow
- [ ] Test a full backtest gameweek and a full local "live" dry run end-to-end

## Phase 4 — Containerization
- [ ] Write a Dockerfile per specialist service (stats, fixtures, injuries, contrarian, template, chips, manager, solver, ingestion, notifier, orchestrator)
- [ ] Local `docker compose` setup to verify the orchestrator can reach every specialist service before moving to Kubernetes

## Phase 5 — Kubernetes deployment
- [ ] Create the `fpl-agents` namespace and base manifests
- [ ] Deploy the `orchestrator` (Deployment + Service)
- [ ] Deploy each specialist service (Deployment or Job, depending on lifecycle)
- [ ] Set up ConfigMaps/Secrets for Microsoft Foundry endpoint/keys and other config
- [ ] Set up the shared data store as a Kubernetes-managed resource (or point to an external one)
- [ ] Write the `deadline-checker` CronJob and the `ingestion` Job
- [ ] Write the `notifier` Job and wire up email/Slack/webhook delivery

## Phase 6 — Guardrails (Microsoft Foundry)
- [ ] Enable Content Safety / Prompt Shields on the Foundry project
- [ ] Wire user-prompt-attack detection in front of any agent that takes external/human input
- [ ] Wire document-attack detection in front of the Injuries agent's RAG-retrieved passages specifically
- [ ] Test with a deliberately "poisoned" test news article to confirm indirect injection is actually caught
- [ ] (Optional) Add groundedness detection on the Injuries agent's summaries

## Phase 7 — Human-in-the-loop recommend & confirm
- [ ] Design the recommendation message format (transfers, captain, chip call, short "why" summary from the debate)
- [ ] Implement the notification delivery (email/Slack/webhook)
- [ ] Document the manual apply step (how you take the recommendation and apply it in the real FPL team)
- [ ] (Stretch) Add a lightweight approve/reject action that resumes the LangGraph interrupt and logs the decision

## Phase 8 — Observability & portfolio polish
- [ ] Log full agent debate transcripts per gameweek to persistent storage
- [ ] Build a small dashboard or report view for backtest results and ablations
- [ ] (Optional) Wire up Microsoft Foundry's evaluation/observability dashboards for agent quality tracking
- [ ] Record a demo (screen capture or GIF) of a live weekly run producing a recommendation
- [ ] Write up the project (README polish, architecture diagram, backtest results, ablation findings)

## Phase 9 — Stretch goals
- [ ] Auto-apply mode against the real FPL account (unofficial API, session-based auth)
- [ ] Mini-league-aware strategy
- [ ] Joint reasoning across chip timing and fixture swings
