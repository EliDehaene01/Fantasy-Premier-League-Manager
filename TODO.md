# Build plan

## Known issues
- Local Postgres (port 5433, per `.env`) must be running for any DB-touching test (chips/contrarian/fixtures/template agents, stats monitor, ingestion) -- `docker compose up` once Phase 4 exists; until then start the container manually.

## Phase 0 — Setup
- [ ] Repo scaffold (project structure, `pyproject.toml`/`requirements.txt`, linting, `CLAUDE.md`)
- [ ] Pull the 2025-26 (and prior season) `merged_gws.csv` files from [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) (sparse-checkout or fetch specific files via raw.githubusercontent.com rather than cloning the whole multi-season repo); confirm gameweek coverage is complete; document schema in `/docs`
- [ ] Test the official FPL public API endpoints (`bootstrap-static`, `fixtures`, `entry/{team_id}`) — no registration or auth needed, it's fully public; find and note your own team's `entry_id` (visible in the URL once logged into fantasy.premierleague.com); set a descriptive `User-Agent` header and add basic caching/backoff since it's not an officially-documented developer product
- [ ] Create a Microsoft Foundry project (via ai.azure.com or the Azure Portal) in a resource group
- [ ] Deploy the model(s) each agent will call (consider Foundry's Model Router for cost/quality balancing across the simpler specialist agents vs. the Manager's synthesis step); note endpoint + key
- [ ] Enable Content Safety / Prompt Shields on the Foundry project (needed later in Phase 6, but easiest to turn on now alongside the rest of the resource setup)
- [ ] Decide on and set up the shared data store (Postgres) for normalized data, LangGraph checkpoints, and transcripts
- [ ] Local dev environment: Python env, Docker, Docker Compose

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
- [x] Build the solver as a PuLP LP: budget, per-club cap, formation, squad-size constraints
- [x] Price transfer-hit cost (-4pts beyond free allowance) directly into the objective
- [x] Add chip-aware constraints: unlimited free transfers under Wildcard/Free Hit -- the solver treats both identically (unlimited this gameweek); distinguishing Free Hit's one-gameweek revert from Wildcard's permanence is squad-state tracking across gameweeks, owned by Phase 2's backtest state tracking / the future Manager, not a single /solve call
- [x] Hard-exclude News-vetoed OUT players from the candidate pool before solving
- [x] Implement infeasibility handling: retry once with one additional transfer hit allowed, then surface "no valid plan found" rather than looping

### Manager
- [x] Implement the multiplicative aggregation formula (predicted_points x (1 + sum of weight_i x conviction_i)) across the four adjustment agents (Fixtures, Contrarian, Template, News recommendations)
- [x] Make weights fixed and config-driven (starting values: Fixtures = Contrarian = Template = 1.0, News = 1.3), not LLM-decided -- document as an untuned baseline to revisit with backtest evidence
- [x] Implement News vetoes handling: OUT hard-excludes (solver-side), DOUBT applies a steep multiplicative penalty scaled by confidence
- [x] Implement deterministic captain/vice-captain selection (highest/second-highest adjusted_score among the finalized starting 11)
- [x] Build the Manager's LLM narration call (explains the already-computed decision; does not decide)
- [x] Build the bounded reaction round: each agent sees all first-round outputs, writes a text-only reaction that cannot alter conviction/recommendations/vetoes

## Phase 2 — Backtest engine
- [x] Implement state tracking (squad, bank, free transfers, chips used) across gameweeks
- [x] Implement the walk-forward loop with strict no-lookahead data slicing
- [x] Stub/skip guardrail calls during backtest (no real external text, avoid unnecessary cost) -- extended to all reasoning/narration LLM calls too (backtest/agents_bridge.py::disable_all_llm_calls), not just guardrails; only the deterministic scoring/solving logic runs for real
- [x] Implement scoring against actual historical results, including transfer-hit and chip effects
- [x] Implement benchmark comparisons -- "never transfer" baseline implemented for real; FPL average-manager score is honestly unavailable (see docs/backtest_results.md for the verified reasoning, not repeated here)
- [x] Implement per-agent ablation runs (backtest/engine.py::run_ablations)
- [ ] Run a full-season backtest and write up results

## Phase 3 — Orchestration (LangGraph)
- [x] Define the typed state schema (squad, bank, free transfers, chips, current gameweek, each agent's raw output, reaction-round text, solver result, approval status)
- [x] Build the StateGraph: six specialist calls (genuinely parallel) -> bounded reaction round -> manager (which internally calls the solver, per ARCHITECTURE.md 6c's node description) -> mode-dependent branch -- "ingestion" is upstream of this graph, not a node inside it (see orchestrator/graph.py's module docstring for why)
- [x] Implement the mode branch: live pauses at the human-approval interrupt; backtest skips it and auto-accepts the Manager's proposal
- [x] Add conditional edges: infeasible Manager result routes to a `declined` terminal, not back to the Manager -- the retry-with-extra-hit logic already runs INSIDE one Manager call (solver/optimizer.py's own two-attempt solve), so a graph-level loop would just re-run the identical infeasible solve (see orchestrator/graph.py's module docstring)
- [x] Wire up a Postgres checkpointer for persistence across weekly runs (`orchestrator/run.py::postgres_checkpointer`; the orchestrator's own tests use an in-memory checkpointer, consistent with every other DB-touching component in this repo needing a live Postgres to test against for real)
- [x] Implement the human-approval `interrupt()` / `Command(resume=...)` flow; rejection/timeout just logs as declined, no retry loop
- [x] Test a full backtest gameweek and a full local "live" dry run end-to-end (fake service callers standing in for live HTTP/Postgres -- Phase 4 containerization hasn't happened yet, so a true multi-service integration run isn't possible until then)

## Phase 4 — Containerization
- [x] Write a Dockerfile per specialist service (stats, fixtures, news, contrarian, template, chips, manager, solver, ingestion, orchestrator) -- frontend-export deferred, see Phase 5a note below; it doesn't exist as code yet
- [x] Local `docker compose` setup to verify the orchestrator can reach every specialist service -- verified for real (not just built): brought the stack up from a genuinely fresh Postgres twice, all 9 services reachable and DB-connected, orchestrator reaching every one over the compose network

## Phase 5 — Local deployment (Docker Compose)
Kubernetes was built, deployed, and run successfully against real data first, then abandoned for Compose after a Docker Desktop kind-mode image-visibility limitation couldn't be resolved -- full account in ARCHITECTURE.md 8b, not repeated here.

- [x] `docker-compose.yml` defines all ten services (nine app containers + Postgres) with real env vars/ports/dependencies, matching what the (now-deleted) k8s manifests defined -- verified `docker compose up -d`: all nine app services healthy, every DB-backed one showing `db_connected: true` against the real, recovered Postgres data (not a fresh empty instance)
- [x] `ingestion` modeled as a Compose "jobs"-profile service (`docker compose run --rm ingestion <mode>`), not a persistent one -- verified `auto` mode's real 24-36h deadline check inside the Compose network end to end
- [x] Replace the CronJob-based weekly trigger with `scripts/weekly_pipeline.py` (`docker compose up -d` then `docker compose run --rm ingestion auto`) -- same deadline-aware logic as the CronJob, unchanged
- [x] Point the existing Windows Task Scheduler "FPL pipeline wake" task (already configured, `WakeToRun`, daily) at the new script instead of its old start-Docker-Desktop-only action -- couldn't be applied directly (`Set-ScheduledTask` needs admin elevation this session didn't have); `scripts/update_wake_task.ps1` does it in one run as Administrator
- [ ] Write the `frontend-export` step: exports pending-approval and final-state JSON per gameweek, commits/pushes to the frontend repo -- not yet wired into `scripts/weekly_pipeline.py`

## Phase 5a — Frontend
- [x] Scaffold a React app (Vite + React, frontend/) -- NOT yet deployed to GitHub Pages (a one-way, publicly-visible action deliberately left for an explicit ask rather than assumed; base path is already configured and ready in vite.config.js)
- [x] Build the per-gameweek view: chosen/proposed team, full six-agent transcript (first round + bounded reaction round), actual points once played -- shows the real per-player/captain-doubled points total, not the average-manager benchmark (unavailable, see docs/backtest_results.md)
- [x] Build the pending-vs-final state rendering (same underlying gameweek record, two states) -- verified in a real browser against two genuine gameweeks from the actual backtest pipeline (not fabricated fixture data), see frontend/public/data/gw9.json (final) and gw10.json (pending)
- [x] Confirm the app only ever reads the exported static JSON -- no calls to Postgres or any backend, directly or indirectly -- verified by grepping frontend/src for every fetch()/XHR/axios call, not assumed: exactly two, both to static JSON under public/data/

## Phase 6 — Guardrails (Microsoft Foundry)
- [ ] Enable Content Safety / Prompt Shields on the Foundry project
- [ ] Wire user-prompt-attack detection in front of any agent that takes external/human input
- [ ] Wire document-attack detection in front of the News agent's RAG-retrieved passages specifically
- [ ] Test with a deliberately "poisoned" test news article to confirm indirect injection is actually caught
- [ ] (Optional) Add groundedness detection on the News agent's summaries

## Phase 7 — Human-in-the-loop recommend & confirm
- [x] Confirm the exported JSON (Phase 5a) carries everything needed for review: transfers, captain, chip call, full transcript -- verified against a real gameweek run through the live k8s orchestrator (not a fixture), see docs/human_in_the_loop.md
- [x] Document the manual apply step (how you take the recommendation and apply it in the real FPL team) -- docs/human_in_the_loop.md
- [x] (Stretch) Add a lightweight approve/reject action that resumes the LangGraph interrupt and logs the decision -- scripts/approve_gameweek.py, run for real against the live k8s deployment: triggered a real gameweek (POST /run, paused at the interrupt with a genuine feasible squad from all six live specialist services), then resumed it (POST /resume) and got back `"approval_status": "approved"` with the full checkpointed state. This live run caught three real bugs, all fixed: langgraph-checkpoint-postgres needing psycopg[binary] (bare psycopg has no libpq in a minimal image), the Postgres checkpointer's connection being silently closed by garbage collection (the context-manager generator holding it open wasn't kept referenced), and orchestrator/requirements.txt missing fastapi/uvicorn entirely (caught earlier in containerization, same root cause: untested code path)

## Phase 8 — Observability & portfolio polish
- [x] Log full agent debate transcripts per gameweek to persistent storage -- already satisfied by the LangGraph Postgres checkpointer (orchestrator/run.py::postgres_checkpointer), verified for real: queried the `checkpoints` table after a live k8s run and confirmed the full first_round/reactions/manage_result state is actually persisted under the `live-gw<N>` thread id. No separate logging mechanism needed or built.
- [x] Build a small dashboard or report view for backtest results and ablations -- frontend's "Season backtest" tab (frontend/src/components/SeasonResults.jsx), reading real data (frontend/public/data/season_2025_26.json, copied from backtest/results.json), verified rendering in a real browser
- [ ] (Optional) Wire up Microsoft Foundry's evaluation/observability dashboards for agent quality tracking -- needs real Azure/Foundry resources this session doesn't have; left for whoever sets up the actual Foundry project (see TODO.md Phase 0)
- [ ] Record a demo (screen capture or GIF) of a live weekly run producing a recommendation, and of the frontend showing a played gameweek -- a manual/human artifact (screen recording), not something producible in this session; the underlying live run and frontend rendering it would capture are both already verified for real (this session's k8s /run+/resume test, and the browser screenshots of the frontend)
- [x] Write up the project (README polish, architecture diagram, backtest results, ablation findings) -- README.md's new "Backtest results" and "Status" sections, with the real numbers from this run

## Phase 9 — Stretch goals
- [ ] Auto-apply mode against the real FPL account -- design already settled (GitHub Actions workflow triggered on PR merge, unofficial session-based login, dry-run period before a real submit); deliberately not built yet, revisit once recommend-and-confirm has run reliably over real gameweeks
- [ ] Genuine multi-turn debate where agents revise positions based on each other's arguments, replacing the current decision-inert reaction round -- deferred since it would break the fixed-weight aggregation's reproducibility guarantee
- [ ] Mini-league-aware strategy
- [ ] Joint reasoning across chip timing and fixture swings
