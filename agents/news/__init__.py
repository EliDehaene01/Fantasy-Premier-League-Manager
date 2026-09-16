"""News agent: availability veto + notable-coverage recommendations.

Two tiers, and they exist separately on purpose - not just as a code
structure preference, but for cost and reliability reasons:

  * **Tier 1** (``tier1.py``) is a few comparisons against fields already
    sitting in silver (``chance_of_playing_this_round``, ``news``). It costs
    nothing, never fails, and resolves the large majority of players (who
    have no fitness concern at all, or an unambiguous "0% - Ruled out"
    entry) with total certainty. Running an LLM call for every one of ~700
    players every gameweek, when a plain threshold check already gives the
    right answer for most of them, would be slower, costlier, and strictly
    less reliable (a model can misread a clear case; a threshold check
    cannot).
  * **Tier 2** (``tier2.py`` + retrieval/scraping) exists for what Tier 1
    genuinely can't resolve: the ambiguous middle (a 50-75% chance, a vague
    "assessed" note) and a second, unrelated job Tier 1 has no way to do at
    all - spotting notable positive coverage a structured field was never
    going to capture. It costs real money and can fail (network, rate
    limits), so it only runs for the players Tier 1 couldn't already
    settle, plus the general-news scan.

This module produces TWO different kinds of output from the same agent
(see agents/stats/schemas.py's ``AgentArgument``):
  * ``vetoes`` - a hard constraint. The Manager must never select a player
    flagged OUT/DOUBT here, regardless of any other agent's predicted
    points. Grounded in either a Tier 1 rule or a Tier 2 LLM read of
    retrieved text.
  * ``recommendations`` - a normal vote, exactly like every other
    specialist's. Notable positive coverage is a *soft* signal (a press-
    conference mention, a breakout write-up), not something the Manager
    should treat as authoritative just because it came from the same agent
    that also produces hard vetoes.
"""
