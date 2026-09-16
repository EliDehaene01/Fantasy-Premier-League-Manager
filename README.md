# FPL multi-agent manager

A multi-agent AI system that manages a real Fantasy Premier League (FPL) squad. A set of specialist agents — orchestrated as a **LangGraph** state machine, each running as its own containerized service — debate transfers, captaincy and chip strategy every gameweek, weighing form, fixtures, injuries, ownership and squad rules before a manager agent finalizes a recommendation for a human to approve.

This is a portfolio project built to demonstrate practical skills with agentic AI frameworks, distributed systems, and production deployment (Docker + Kubernetes) — not just a single LLM calling tools in a loop.

## What it does

- Six agents (five specialists + one manager) hold a structured debate over each gameweek's squad decision, each arguing from a different signal: predicted points, fixture difficulty, injuries/availability, differential picks, template safety, and chip timing.
- The Stats agent's predictions come from a properly built data science pipeline (feature engineering, time-respecting cross-validation, model comparison, SHAP explainability) — not a single quick model.
- The News agent checks structured availability data deterministically first (no LLM needed for clear-cut cases), then uses retrieval-augmented generation over scraped FPL news text for the ambiguous middle — vetoing players on availability grounds, but also surfacing genuinely notable positive coverage (a manager's press-conference praise, a breakout performance) as a normal recommendation, weighted like any other specialist rather than a hard constraint. A small fine-tuned classifier handles the availability-severity judgment specifically, evaluated against a zero-shot baseline.
- A constraint solver (PuLP/OR-Tools) turns the debate into a feasible 15-man squad within FPL's budget and formation rules.
- The system runs in two modes:
  - **Backtest** — walks forward through a full historical season (using [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)'s `merged_gws.csv` archive, which is genuinely gameweek-by-gameweek, unlike the Kaggle snapshots we initially considered) gameweek by gameweek, without lookahead, and reports final points against real benchmarks.
  - **Live weekly** — runs on a schedule ahead of each real gameweek deadline, pulls fresh data from the official FPL API, and sends a **recommend-and-confirm** notification: the human reviews and approves before anything is applied to the real team.
- Each specialist agent runs as its own containerized service, called from a LangGraph state machine that persists squad/bank/chip state across gameweeks and pauses for human approval before anything is finalized.
- LLM calls are hosted through Microsoft Foundry (formerly Azure AI Foundry), with Prompt Shields guarding against both direct prompt injection and indirect/document-based injection — the latter is a real risk here, since the News agent ingests external news text.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full breakdown of agents, data flow, and deployment topology, and [TODO.md](./TODO.md) for the build plan.

## Why this project

Fantasy Premier League is a genuinely hard weekly optimization problem under uncertainty (form, injuries, fixture swings, budget) with a built-in way to measure success: points. That makes it a good vehicle for showing:

- Agentic reasoning where agents legitimately disagree and a human-in-the-loop step matters (this isn't a rubber-stamp "confirm" — the recommendation carries real trade-offs).
- Combining LLM-based reasoning with classical ML (form/points prediction), RAG, a small fine-tuned classifier, and classical optimization (the solver) rather than asking an LLM to do everything.
- Running a genuinely distributed multi-agent system (separate containers/pods per agent) rather than a single-process demo.
- End-to-end production deployment: containerization, Kubernetes scheduling, and a real operational cadence (weekly, tied to a real-world deadline).
- Enterprise-relevant tooling: LangGraph for stateful orchestration, Microsoft Foundry for model hosting and guardrails — the kind of stack most companies actually build agent systems on top of.

## Tech stack

- **Orchestration**: Python, LangGraph (typed state graph, checkpointed persistence, human-in-the-loop interrupts)
- **Model hosting & guardrails**: Microsoft Foundry (model deployments, Content Safety / Prompt Shields for direct + indirect prompt injection)
- **ML**: XGBoost / Random Forest for expected-points prediction, SHAP for explainability
- **RAG**: vector store over injury/team-news text for the News agent
- **Fine-tuning**: small classifier for injury severity, evaluated against a zero-shot baseline
- **Optimization**: PuLP or OR-Tools for squad-selection constraints
- **Data**: FPL public API, no auth required (live); vaastav/Fantasy-Premier-League GitHub archive (backtest)
- **Infra**: Docker, Kubernetes (CronJob, Deployments, Services, ConfigMaps/Secrets)
- **Notifications**: email/Slack/webhook for the recommend-and-confirm step

## Status

Early planning stage — see [TODO.md](./TODO.md) for current progress and next steps.

## License

TBD.
