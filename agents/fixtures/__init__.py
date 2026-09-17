"""Fixtures agent: rule-based scoring of upcoming fixture favorability.

No ML model, no LLM in the scoring path, no dependency on any other agent -
purely a read of fixture difficulty (FDR) and fixture COUNT (blank/double
gameweeks) already sitting in the `fixtures` silver table (see
``ingestion/db.py``). See ``scoring.py`` for the exact weighting rule and
``service.py``/``reasoning.py`` for how a Foundry LLM turns the deterministic
ranking into prose (same two-step shape as the Stats agent - see
``agents/stats/service.py``'s module docstring for why that split exists).
"""
