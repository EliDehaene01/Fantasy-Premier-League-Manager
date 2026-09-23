# FPL multi-agent manager

A multi-agent AI system that manages a real Fantasy Premier League (FPL) squad. A set of specialist agents — orchestrated as a **LangGraph** state machine, each running as its own containerized service — debate transfers, captaincy and chip strategy every gameweek, weighing form, fixtures, injuries, ownership and squad rules before a manager agent finalizes a recommendation for a human to approve.

This is a project built to demonstrate practical skills with agentic AI frameworks, multi-service system design, and production-style deployment (Docker + Docker Compose) — not just a single LLM calling tools in a loop.

## What it does

- Seven agents (six specialists + one manager) hold a structured debate over each gameweek's squad decision, each arguing from a different signal: predicted points, fixture difficulty, availability and notable news coverage, differential picks, template safety, and chip timing.
- The Stats agent's predictions come from a properly built data science pipeline (feature engineering, time-respecting cross-validation, model comparison, SHAP explainability) — not a single quick model.
- The News agent checks structured availability data deterministically first (no LLM needed for clear-cut cases), then uses retrieval-augmented generation over scraped FPL news text for the ambiguous middle — vetoing players on availability grounds, but also surfacing genuinely notable positive coverage (a manager's press-conference praise, a breakout performance) as a normal recommendation, weighted like any other specialist rather than a hard constraint. A small fine-tuned classifier handles the availability-severity judgment specifically, evaluated against a zero-shot baseline.
- A constraint solver (PuLP) turns the debate into a feasible 15-man squad within FPL's budget and formation rules — Stats' predicted points as the base, the other agents' conviction as fixed, config-driven multiplicative adjustments (News weighted slightly higher, reflecting its qualitative, closer-to-inside-information signal), never an LLM-decided weighting, so the same inputs always produce the same squad.
- The system runs in two modes:
  - **Backtest** — walks forward through a full historical season (using [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)'s `merged_gws.csv` archive, which is genuinely gameweek-by-gameweek, unlike the Kaggle snapshots we initially considered) gameweek by gameweek, without lookahead, and reports final points against real benchmarks.
  - **Live weekly** — runs on a schedule ahead of each real gameweek deadline, pulls fresh data from the official FPL API, and publishes a **recommend-and-confirm** proposal to a static site: the human reviews and approves manually before anything is applied to the real team; nothing writes to the live account automatically.
- A React frontend, deployed to GitHub Pages and reading only static exported data (no live backend calls), shows each gameweek's proposed or final squad, the full agent debate transcript, and — once played — the actual points scored against a benchmark.
- Each specialist agent runs as its own containerized service, called from a LangGraph state machine that persists squad/bank/chip state across gameweeks and pauses for human approval before anything is finalized.
- LLM calls are hosted through Microsoft Foundry (formerly Azure AI Foundry), with Prompt Shields guarding against both direct prompt injection and indirect/document-based injection — the latter is a real risk here, since the News agent ingests external news text.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full breakdown of agents, data flow, and deployment topology, and [TODO.md](./TODO.md) for the build plan.

## Why this project

Fantasy Premier League is a genuinely hard weekly optimization problem under uncertainty (form, injuries, fixture swings, budget) with a built-in way to measure success: points. That makes it a good vehicle for showing:

- Agentic reasoning where agents legitimately disagree and a human-in-the-loop step matters (this isn't a rubber-stamp "confirm" — the recommendation carries real trade-offs).
- Combining LLM-based reasoning with classical ML (form/points prediction), RAG, a small fine-tuned classifier, and classical optimization (the solver) rather than asking an LLM to do everything.
- Running a genuinely multi-service, containerized system (separate containers per agent, orchestrated with Docker Compose) rather than a single-process demo — not cluster-scale distribution, but real service separation with independent Dockerfiles, ports, and health checks.
- End-to-end production-style deployment: containerization, a scheduled weekly trigger (Windows Task Scheduler wakes the machine, a script brings the Compose stack up and checks the real deadline), and a real operational cadence tied to a real-world deadline.
- Enterprise-relevant tooling: LangGraph for stateful orchestration, Microsoft Foundry for model hosting and guardrails — the kind of stack most companies actually build agent systems on top of.

## Tech stack

- **Orchestration**: Python, LangGraph (typed state graph, checkpointed persistence, human-in-the-loop interrupts)
- **Model hosting & guardrails**: Microsoft Foundry (model deployments, Content Safety / Prompt Shields for direct + indirect prompt injection)
- **ML**: XGBoost / Random Forest / Poisson regression / MLP for expected-points prediction, SHAP for explainability
- **RAG**: pgvector over scraped FPL Team News and general News text for the News agent
- **Fine-tuning**: small classifier for injury severity, evaluated against a zero-shot baseline
- **Optimization**: PuLP for squad-selection constraints
- **Data**: FPL public API, no auth required (live); vaastav/Fantasy-Premier-League GitHub archive (backtest)
- **Infra**: Docker, Docker Compose (ten services — nine app containers plus Postgres — scheduled weekly via Windows Task Scheduler + a trigger script)
- **Frontend**: React, deployed to GitHub Pages, reading only static exported JSON — no live backend
- **Recommend-and-confirm**: no push notification — the weekly static export to GitHub Pages is the review surface

## Backtest results

Full 2025-26 season (GW2–38), walked forward with no lookahead, real trained
model, real solver, real archive data — not illustrative numbers:

| | Total points |
|---|---|
| This system | **2182** |
| Never-transfer baseline | 1398 |
| FPL average manager | not available (see `backtest/benchmarks.py` — no per-gameweek figure exists for a completed prior season, in the archive or the live API; not fabricated) |

784 points ahead of the never-transfer baseline over the season — the
transfer/chip machinery is doing real work, not just picking a good initial
XI and coasting.

**Per-agent ablations** (season rerun with one specialist removed at a
time — the delta is what that agent was actually worth this season):

| Agent | Delta |
|---|---|
| Chips | +86 |
| Template | +63 |
| Contrarian | +0 |
| News | +0 (expected — contributes nothing during backtest, see below) |
| Fixtures | −16 |

Chips and Template contributed the most; Contrarian's signal existed but
never actually swung a decision this particular season; Fixtures was
mildly net-negative — a real, if counterintuitive, finding, reported
honestly rather than smoothed over. See `backtest/results.json` for the
full per-gameweek breakdown, and the frontend's "Season backtest" tab for
the same data rendered.

**News's backtest limitation, stated plainly**: it contributes nothing
during backtest runs (empty recommendations/vetoes every gameweek) because
`chance_of_playing_this_round` is a live, forward-looking field with no
honest historical equivalent in the archive — deriving one from that
gameweek's own outcome (e.g. inferring an injury from minutes actually
played) would leak the result into what's supposed to be a pre-match
signal. Live mode is unaffected; this is a backtest-only gap, documented
in `data/backtest_seed.py`.

## Status

Solver, Manager, LangGraph orchestration, backtest engine (run for real
over a full season, see above), containerization, and a working local
Docker Compose deployment are built and verified — all ten services
(six specialists, Manager, solver, orchestrator, Postgres) reach a
genuinely healthy, DB-connected state from a real `docker compose up -d`,
and the orchestrator's `/run` + `/resume` interrupt cycle has been run for
real against live specialist services, not just unit tests. Kubernetes was
also built and deployed successfully for several hours against real data,
but was abandoned in favor of Compose after a Docker Desktop kind-mode
image-visibility limitation on the development machine couldn't be
resolved (see ARCHITECTURE.md 8b and TODO.md for the full record — kept
as an honest account of what was tried, not erased). The frontend reads
real exported gameweek and season data, verified rendering in an actual
browser. See [TODO.md](./TODO.md) for the authoritative, itemized
checklist — remaining open items are mainly Microsoft Foundry guardrail
wiring for the agents other than News (needs real Azure resources), the
`frontend-export` step (exists as a Python module, not yet wired into the
scheduled weekly pipeline), and actually publishing the frontend to
GitHub Pages (a one-way, publicly-visible action left for an explicit
decision rather than assumed).

## License

TBD.
