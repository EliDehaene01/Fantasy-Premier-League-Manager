"""LangGraph orchestration (ARCHITECTURE.md 6c). Sequences the six
specialist calls (genuinely parallel), the bounded reaction round, the
Manager, and the mode-dependent branch (live pauses for human approval;
backtest auto-accepts) - and owns the Postgres checkpointer that persists
state across weekly runs.

``state.py`` - the typed graph state.
``callers.py`` - injectable HTTP callers to each service; tests substitute
fakes so the graph's own wiring is verifiable without six live services and
a database.
``graph.py`` - the StateGraph definition (see its module docstring for two
scope notes: why "ingestion" isn't a node inside this graph, and why the
infeasibility conditional edge doesn't loop back into the Manager).
``run.py`` - thin entrypoints (``run_backtest_gameweek``,
``run_live_gameweek``) the backtest engine and the live weekly trigger call.
"""
