"""Squad-selection solver - a PuLP linear program, not an agent.

Turns a pool of candidate players (each already carrying the Manager's
computed ``adjusted_score`` - see ARCHITECTURE.md 6b) into a legal 15-man
squad plus starting 11, respecting budget/formation/club-cap constraints and
pricing transfer hits into the objective. No LLM call, no debate - this is
the deterministic "opinions become a legal squad" step described in
ARCHITECTURE.md 6a.

``optimizer.py`` holds the actual LP (pure function, no FastAPI/DB
dependency, easy to unit test in isolation); ``service.py`` is the thin
FastAPI wrapper exposing it as ``POST /solve``.
"""
