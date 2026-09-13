# Build plan

## Phase 0 — Setup
- [ ] Repo scaffold (project structure ✅, dependency management ✅ — per-package `requirements.txt` chosen over a root `pyproject.toml`, see CLAUDE.md's Conventions section — linting still TBD, `CLAUDE.md` ✅)
- [ ] Pull the 2025-26 (and prior season) `merged_gws.csv` files from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) (sparse-checkout or fetch specific files via raw.githubusercontent.com rather than cloning the whole multi-season repo); confirm gameweek coverage is complete; document schema in `/docs`
- [ ] Test the official FPL public API endpoints (`bootstrap-static`, `fixtures`, `entry/{team_id}`) — no registration or auth needed, it's fully public; find and note your own team's `entry_id` (visible in the URL once logged into fantasy.premierleague.com); set a descriptive `User-Agent` header and add basic caching/backoff since it's not an officially-documented developer product
- [ ] Create a Microsoft Foundry project (via ai.azure.com or the Azure Portal) in a resource group
- [ ] Deploy the model(s) each agent will call (consider Foundry's Model Router for cost/quality balancing across the simpler specialist agents vs. the Manager's synthesis step); note endpoint + key
- [ ] Enable Content Safety / Prompt Shields on the Foundry project (needed later in Phase 6, but easiest to turn on now alongside the rest of the resource setup)
- [x] Decide on and set up the shared data store (Postgres) for normalized data, LangGraph checkpoints, and transcripts (local Postgres instance running, `DATABASE_URL` in `.env`, `ingestion/db.py` connects to it — the silver layer now lives there; LangGraph checkpointer/transcript tables are still TBD, that's Phase 3 scope)
- [ ] Local dev environment: Python env, Docker, a local Kubernetes cluster (kind or minikube) for testing manifests before any real cluster

## Phase 1 — Core agent logic

### Data science pipeline (Stats agent)
- [ ] EDA on the historical dataset (form, minutes, price changes, injury frequency)
- [ ] Feature engineering (rolling form windows, fixture-adjusted expected points, per-90 normalization)
- [ ] Time-respecting train/validation split (chronological, no random k-fold)
- [ ] Train and compare five approaches: naive rolling-average baseline, Poisson/regularized linear regression, Random Forest, XGBoost, and a small MLP
- [ ] Add SHAP explainability, wire top-feature explanations into the Stats agent's argument
- [ ] Handle cold-start explicitly (early season, thin rolling-window history) in the feature engineering module

### Model lifecycle (training -> serving -> monitoring -> retraining)
- [x] Refactor feature engineering into one shared module imported by both the training script and the live Stats agent, to eliminate train/serve skew (`features/engineering.py`; cold-start handling for <3-GW players added alongside it)
- [x] Add hyperparameter tuning (e.g. Optuna) for the winning model, using the same chronological split as before -- no random k-fold here either (`agents/stats/tune.py`; Optuna TPE, forward-chaining folds inside the training pool only. First pass optimized MAE and picked a model that regressed RMSE ~9.6% -- corrected to optimize RMSE directly with a selection rule requiring both RMSE and Spearman not to regress; the re-tuned model deployed now is only marginally different from the untuned one, honestly reported as such rather than oversold -- see docs/model_comparison.md's "Hyperparameter tuning" section)
- [x] Build the weekly ingestion pipeline against the official FPL API, layered bronze (raw JSON per endpoint) -> silver (normalized: players, teams, gameweeks, fixtures, player_gameweek_stats, my_team_state) -> gold (feature-ready, via the shared feature module); covers bootstrap-static, fixtures, event/{gw}/live, element-summary/{id}, and entry/{id} + its sub-endpoints for our own squad/bank/transfer state (`ingestion/`; Postgres via `DATABASE_URL` — migrated off the sqlite stand-in once a local instance existed)
- [x] Backfill silver from gameweek 1 of the current season (available via the API since the season's in progress); do not attempt to backfill last season this way -- the API only has season-level totals for completed seasons, not gameweek granularity (`ingestion/backfill.py`)
- [x] Build a schema-reconciliation/adapter step mapping the vaastav archive's columns onto the self-built silver schema, so future retraining can combine both sources consistently (`data/reconcile_archive.py`)
- [x] Build a predictions log: every live prediction (player, gameweek, predicted points) gets stored for later comparison against actual results (`agents/stats/predictions_log.py`, wired into `/argue`; tagged with `model_version` so a retrain never blends old/new predictions - see docs/monitoring.md)
- [x] Build the monitoring job: after each gameweek's results are final, join predictions log against actuals, compute a rolling error metric (`agents/stats/monitor.py`; RMSE, not MAE - matches tune.py's corrected objective; baseline read from `models/model_baseline.json`, written automatically by `train.py::save_model`)
- [x] Define and implement a retrain trigger: rolling error past threshold -> rerun training on the accumulated dataset -> redeploy the new model artifact (`agents/stats/retrain.py`; fires automatically from monitor.py after 2 consecutive degraded runs, actually re-fits + re-tunes + redeploys, not a stub; season-aware combined archive+current-season dataset - required a small season-aware extension to features/engineering.py, verified backward-compatible)
- [ ] Re-run the full five-model comparison at natural checkpoints (e.g. season boundaries) rather than assuming XGBoost stays the best choice forever

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
