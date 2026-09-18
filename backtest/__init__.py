"""ARCHITECTURE.md 9's backtest engine: walks forward through a historical
season, gameweek by gameweek, with no lookahead, scoring the Manager's
proposal against actual results.

``data/backtest_seed.py`` (not in this package - shared with the rest of
the archive-wrangling code in ``data/``) seeds one season's vaastav archive
into an isolated Postgres schema, shaped like the live silver tables, so
the real DB-backed agents (Fixtures/Contrarian/Template/Chips) run their
actual scoring logic against historical data rather than a second,
divergent backtest-only implementation.

``state.py`` - the per-season SeasonState (squad, bank, free transfers,
chips used) carried across gameweeks.
``agents_bridge.py`` - in-process callers (no HTTP, no live LLM) wiring
each specialist's real scoring function to the seeded schema, for
orchestrator.graph.build_graph's injectable caller parameters. News
contributes nothing (see data/backtest_seed.py's module docstring for why).
``scoring.py`` - scores a finalized squad against actual results, applies
hit/chip point effects, and computes the season-state update (new squad,
bank, free transfers, chips used).
``engine.py`` - the walk-forward loop, with per-agent ablation support.
``benchmarks.py`` - season-total comparison against the FPL average-manager
score and a "never transfer" baseline.
"""
