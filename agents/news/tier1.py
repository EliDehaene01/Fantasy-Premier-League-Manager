"""Tier 1: deterministic availability check - no LLM, no retrieval, no cost.

Reads exactly two fields already flowing through silver from
``bootstrap-static`` (see ``ingestion/silver.py``): ``chance_of_playing_this_round``
(FPL's own 0/25/50/75/100/None scale) and ``news`` (free text FPL's editors
attach - "Hamstring injury - Expected back 12 Oct", "Suspended until 25 Oct").

WHY THIS EXISTS SEPARATELY FROM TIER 2 (see also agents/news/__init__.py)
--------------------------------------------------------------------------
Cost: an LLM call per player, every gameweek, for ~700 players, to re-derive
a verdict a simple threshold already gives for free, is pure waste for the
cases where the structured field is already unambiguous.

Reliability: "0% chance of playing" or the literal word "Suspended" is not
open to interpretation - a rule either matches or it doesn't, deterministically,
every time. An LLM asked the same question can occasionally misread even an
unambiguous case (hallucinate nuance that isn't there, get distracted by
unrelated context). For the clear-cut majority, a rule is not just cheaper
than a model call, it is MORE reliable, not less - which is exactly why
Tier 1 runs first and only hands Tier 2 what's genuinely left over.

THE EXACT RULES
----------------
Given (chance_of_playing_this_round: float | None, news: str | None):

  1. chance_of_playing_this_round == 0            -> OUT, confidence 1.0
  2. news matches an explicit ruled-out/suspension
     phrase (case-insensitive; see _OUT_PHRASES)  -> OUT, confidence 1.0
  3. chance_of_playing_this_round == 100,
     or (chance is None/missing AND news is empty) -> FIT, confidence 1.0
     (FPL leaves both fields blank/null for a fully fit player with no
     injury news at all - absence of a concern IS the fit signal here,
     not something to leave unresolved.)
  4. anything else (25%, 50%, 75%, a chance value
     with no matching explicit phrase, or news text
     that doesn't match an OUT phrase but isn't
     obviously nothing either) -> UNRESOLVED (return None):
     Tier 1 deliberately does NOT guess at the ambiguous middle - that is
     exactly the job Tier 2's RAG + LLM read of the actual prose is for
     (see tier2.py). Returning ``None`` here is a real, meaningful signal:
     "Tier 1 has no confident opinion", not "assume fit".
"""

from __future__ import annotations

from agents.stats.schemas import Veto, VetoStatus

# Case-insensitive substrings that FPL's own `news` field uses for a
# definite, non-negotiable absence. Kept as a short, explicit list rather
# than a fuzzier heuristic - a false match here would wrongly veto a
# player, so only genuinely unambiguous phrasing belongs.
_OUT_PHRASES = (
    "ruled out",
    "suspended",
    "will not play",
    "out for the season",
    "long-term injury",
)


def check_tier1(player_id: int, chance_of_playing: float | None, news: str | None) -> Veto | None:
    """Return a ``Veto`` for a clear-cut case, or ``None`` if Tier 1 can't
    resolve this player and it needs to go to Tier 2.
    """
    news_lower = (news or "").strip().lower()

    if chance_of_playing == 0:
        return Veto(
            player_id=player_id, status=VetoStatus.OUT, confidence=1.0,
            grounding_snippet=news or "chance_of_playing_this_round = 0%",
        )

    if any(phrase in news_lower for phrase in _OUT_PHRASES):
        return Veto(
            player_id=player_id, status=VetoStatus.OUT, confidence=1.0,
            grounding_snippet=news,
        )

    no_news = news_lower == ""
    if chance_of_playing == 100 or (chance_of_playing is None and no_news):
        return Veto(
            player_id=player_id, status=VetoStatus.FIT, confidence=1.0,
            grounding_snippet=news or None,
        )

    return None
