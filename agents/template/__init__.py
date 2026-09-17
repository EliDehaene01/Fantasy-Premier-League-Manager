"""Template agent: high-ownership, "safe" picks that protect overall rank.

Rule-based scoring from ownership % and net transfer momentum, both already
in silver from bootstrap-static - no dependency on predicted points from
another agent, since this agent's whole point is following the crowd, not
evaluating quality independently (contrast with Contrarian, which
deliberately DOES read the Stats agent's predictions). See scoring.py for
the exact ranking rule and service.py/reasoning.py for how a Foundry LLM
turns the deterministic ranking into prose (same two-step shape as the
Stats agent).
"""
