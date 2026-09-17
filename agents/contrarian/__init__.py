"""Contrarian agent: low-ownership, high-upside differential picks.

Rule-based scoring, not an ML model of its own - the "quality" half of its
signal is read from the Stats agent's predictions log rather than computed
here (see scoring.py's module docstring for why: service independence, not
a shortcut). See scoring.py for the exact ranking rule and
service.py/reasoning.py for how a Foundry LLM turns the deterministic
ranking into prose (same two-step shape as the Stats agent).
"""
