"""Manager - aggregation, weighting, solver call, captain selection,
narration (ARCHITECTURE.md 6b). Turns six agents' first-round outputs into
one deterministic squad decision, then explains it - never decides via LLM.

``config.py`` - fixed, config-driven weights (CLAUDE.md hard constraint:
never LLM-decided per run).
``aggregation.py`` - the multiplicative formula, veto handling, captain
selection. Pure functions, no DB/LLM/HTTP - the actual decision logic.
``reaction.py`` - the bounded, text-only reaction round.
``narration.py`` - the LLM call that explains an already-computed decision.
``service.py`` - the thin FastAPI wrapper (``POST /manage``, ``POST
/react``) that composes the above and calls the solver.

Deliberately DB-free, same as ``solver/``: static player facts
(price/position/club) come in on the request rather than being queried
here, so this stays as testable as the solver itself. The orchestrator
(LangGraph, not yet built) is the natural place to assemble that from
silver before calling this service.
"""
