# Build plan

## Known issues
- [ ] agents/stats's test_stats_agent.py and test_monitor.py can't be collected due to a Windows Application Control/Smart App Control policy blocking scipy's compiled DLLs on this machine -- pre-existing environment issue, not caused by any change in this project; needs local allowlisting or a re-signed scipy build to resolve, deferred for now

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
- [ ] EDA on the historical dataset (form, minutes, price changes, injury frequency)
- [ ] Feature engineering (rolling form windows, fixture-adjusted expected points, per-90 normalization)
- [ ] Time-respecting train/validation split (chronological, no random k-fold)
- [ ] Train and compare five approaches: naive rolling-average baseline, Poisson/regularized linear regression, Random Forest, XGBoost, and a small MLP
- [ ] Add SHAP explainability, wire top-feature explanations into the Stats agent's argument
- [ ] Handle cold-start explicitly (early season, thin rolling-window history) in the feature engineering module

### Model lifecycle (training -> serving -> monitoring -> retraining)
- [ ] Refactor feature engineering into one shared module imported by both the training script and the live Stats agent, to eliminate train/serve skew
- [ ] Add hyperparameter tuning (e.g. Optuna) for the winning model, using the same chronological split as before -- no random k-fold here either
- [ ] Build the weekly ingestion pipeline against the official FPL API, layered bronze (raw JSON per endpoint) -> silver (normalized: players, teams, gameweeks, fixtures, player_gameweek_stats, my_team_state) -> gold (feature-ready, via the shared feature module); covers bootstrap-static, fixtures, event/{gw}/live, element-summary/{id}, and entry/{id} + its sub-endpoints for our own squad/bank/transfer state
- [ ] Backfill silver from gameweek 1 of the current season (available via the API since the season's in progress); do not attempt to backfill last season this way -- the API only has season-level totals for completed seasons, not gameweek granularity
- [ ] Build a schema-reconciliation/adapter step mapping the vaastav archive's columns onto the self-built silver schema, so future retraining can combine both sources consistently
- [ ] Build a predictions log: every live prediction (player, gameweek, predicted points) gets stored for later comparison against actual results
- [ ] Build the monitoring job: after each gameweek's results are final, join predictions log against actuals, compute a rolling error metric
- [ ] Define and implement a retrain trigger: rolling error past threshold -> rerun training on the accumulated dataset -> redeploy the new model artifact
- [ ] Re-run the full five-model comparison at natural checkpoints (e.g. season boundaries) rather than assuming XGBoost stays the best choice forever

### Two-tier availability and news pipeline (News agent)
- [ ] Build Tier 1: deterministic check of `chance_of_playing_this_round` and `news` (already in silver) -- no LLM call for clear-cut cases
- [ ] Scrape and tag two corpora at ingestion time: the FPL site's Team News tab (availability-focused, populates ~24h before deadline -- scrape on the same trigger as weekly ingestion) and the general News section (form write-ups, price-change articles -- where notable positive coverage lives)
- [ ] Build entity linking: match player names/surnames in scraped text to player_id via the players silver table, at ingestion time
- [ ] Set up pgvector on the existing Postgres instance; chunk and embed both corpora with text-embedding-3-small
- [ ] Build Tier 2 retrieval: filter by entity-linking tag, rank by embedding similarity
- [ ] Wire Prompt Shields document-attack detection on every retrieved passage before it reaches the agent's context
- [ ] Extend the shared agent contract: add a `vetoes` field (player_id, status OUT/DOUBT/FIT, confidence, grounding snippet); make `predicted_points` optional in `recommendations` for this agent, since qualitative buzz doesn't come with a real number
- [ ] Wire Tier 2 to populate `vetoes` for ambiguous availability cases and `recommendations` for notable positive coverage -- these are different judgments feeding different contract fields, not one output

### Fine-tuning (News agent)
- [ ] Label a small dataset for injury-severity/return-timeline classification -- scoped to availability only, not the positive-coverage judgment
- [ ] Fine-tune a small classifier (or embedding model) for this task
- [ ] Evaluate against a zero-shot prompting baseline (precision/recall) and document the comparison

### Remaining agents
- [x] Write the Fixtures agent (FDR scoring, blank/double gameweek detection)
- [x] Write the Contrarian agent (ownership-vs-points logic, reading from the Stats predictions log rather than a live call)
- [x] Write the Template agent (ownership/net-transfers logic)
- [x] Write the Chips agent (chip-timing logic against the fixture calendar, extending the shared contract with a `chip_recommendation` field)

### Solver
- [ ] Build the solver as a PuLP LP: budget, per-club cap, formation, squad-size constraints
- [ ] Price transfer-hit cost (-4pts beyond free allowance) directly into the objective
- [ ] Add chip-aware constraints: unlimited free transfers under Wildcard/Free Hit (distinguish Free Hit's one-gameweek revert from Wildcard's permanence)
- [ ] Hard-exclude News-vetoed OUT players from the candidate pool before solving
- [ ] Implement infeasibility handling: retry once with one additional transfer hit allowed, then surface "no valid plan found" rather than looping

### Manager
- [ ] Implement the multiplicative aggregation formula (predicted_points x (1 + sum of weight_i x conviction_i)) across the four adjustment agents (Fixtures, Contrarian, Template, News recommendations)
- [ ] Make weights fixed and config-driven (starting values: Fixtures = Contrarian = Template = 1.0, News = 1.3), not LLM-decided -- document as an untuned baseline to revisit with backtest evidence
- [ ] Implement News vetoes handling: OUT hard-excludes (solver-side), DOUBT applies a steep multiplicative penalty scaled by confidence
- [ ] Implement deterministic captain/vice-captain selection (highest/second-highest adjusted_score among the finalized starting 11)
- [ ] Build the Manager's LLM narration call (explains the already-computed decision; does not decide)
- [ ] Build the bounded reaction round: each agent sees all first-round outputs, writes a text-only reaction that cannot alter conviction/recommendations/vetoes

## Phase 2 — Backtest engine
- [ ] Implement state tracking (squad, bank, free transfers, chips used) across gameweeks
- [ ] Implement the walk-forward loop with strict no-lookahead data slicing
- [ ] Stub/skip guardrail calls during backtest (no real external text, avoid unnecessary cost)
- [ ] Implement scoring against actual historical results, including transfer-hit and chip effects
- [ ] Implement benchmark comparisons (FPL average manager score, "never transfer" baseline)
- [ ] Implement per-agent ablation runs
- [ ] Run a full-season backtest and write up results

## Phase 3 — Orchestration (LangGraph)
- [ ] Define the typed state schema (squad, bank, free transfers, chips, current gameweek, each agent's raw output, reaction-round text, solver result, approval status)
- [ ] Build the StateGraph: ingestion -> six specialist calls (genuinely parallel) -> bounded reaction round -> manager -> solver -> mode-dependent branch
- [ ] Implement the mode branch: live pauses at the human-approval interrupt; backtest skips it and auto-accepts the Manager's proposal
- [ ] Add conditional edges (e.g. infeasible solver result routes back to the manager for the retry-with-extra-hit logic)
- [ ] Wire up a Postgres checkpointer for persistence across weekly runs
- [ ] Implement the human-approval `interrupt()` / `Command(resume=...)` flow; rejection/timeout just logs as declined, no retry loop
- [ ] Test a full backtest gameweek and a full local "live" dry run end-to-end

## Phase 4 — Containerization
- [ ] Write a Dockerfile per specialist service (stats, fixtures, news, contrarian, template, chips, manager, solver, ingestion, frontend-export, orchestrator)
- [ ] Local `docker compose` setup to verify the orchestrator can reach every specialist service before moving to Kubernetes

## Phase 5 — Kubernetes deployment
- [ ] Create the `fpl-agents` namespace and base manifests, on local Docker Desktop Kubernetes (not a cloud cluster -- deliberate cost/portfolio-value decision, see ARCHITECTURE.md 8b)
- [ ] Deploy the `orchestrator` (Deployment + Service)
- [ ] Deploy each specialist service (Deployment or Job, depending on lifecycle)
- [ ] Set up ConfigMaps/Secrets for Microsoft Foundry endpoint/keys and other config
- [ ] Decide and document whether Postgres runs inside the Kubernetes namespace or stays as the external local container already in use -- either is fine, but pick one deliberately rather than leaving it ambiguous
- [ ] Write the `deadline-checker` CronJob and the `ingestion` Job
- [ ] Write the `frontend-export` Job: exports pending-approval and final-state JSON per gameweek, commits/pushes to the frontend repo
- [ ] Set up Windows Task Scheduler's "wake this computer" option so the local CronJob actually fires if the machine is asleep -- test this once, don't assume it works

## Phase 5a — Frontend
- [ ] Scaffold a React app, deployed to GitHub Pages
- [ ] Build the per-gameweek view: chosen/proposed team, full six-agent transcript (first round + bounded reaction round), actual points once played, shown against the backtest's average-manager benchmark
- [ ] Build the pending-vs-final state rendering (same underlying gameweek record, two states)
- [ ] Confirm the app only ever reads the exported static JSON -- no calls to Postgres or any backend, directly or indirectly

## Phase 6 — Guardrails (Microsoft Foundry)
- [ ] Enable Content Safety / Prompt Shields on the Foundry project
- [ ] Wire user-prompt-attack detection in front of any agent that takes external/human input
- [ ] Wire document-attack detection in front of the News agent's RAG-retrieved passages specifically
- [ ] Test with a deliberately "poisoned" test news article to confirm indirect injection is actually caught
- [ ] (Optional) Add groundedness detection on the News agent's summaries

## Phase 7 — Human-in-the-loop recommend & confirm
- [ ] Confirm the exported JSON (Phase 5a) carries everything needed for review: transfers, captain, chip call, full transcript -- this is the recommendation format, there's no separate notification message to design
- [ ] Document the manual apply step (how you take the recommendation and apply it in the real FPL team)
- [ ] (Stretch) Add a lightweight approve/reject action that resumes the LangGraph interrupt and logs the decision

## Phase 8 — Observability & portfolio polish
- [ ] Log full agent debate transcripts per gameweek to persistent storage
- [ ] Build a small dashboard or report view for backtest results and ablations
- [ ] (Optional) Wire up Microsoft Foundry's evaluation/observability dashboards for agent quality tracking
- [ ] Record a demo (screen capture or GIF) of a live weekly run producing a recommendation, and of the frontend showing a played gameweek
- [ ] Write up the project (README polish, architecture diagram, backtest results, ablation findings)

## Phase 9 — Stretch goals
- [ ] Auto-apply mode against the real FPL account -- design already settled (GitHub Actions workflow triggered on PR merge, unofficial session-based login, dry-run period before a real submit); deliberately not built yet, revisit once recommend-and-confirm has run reliably over real gameweeks
- [ ] Genuine multi-turn debate where agents revise positions based on each other's arguments, replacing the current decision-inert reaction round -- deferred since it would break the fixed-weight aggregation's reproducibility guarantee
- [ ] Mini-league-aware strategy
- [ ] Joint reasoning across chip timing and fixture swings
