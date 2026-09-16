"""Tier 2: RAG + LLM judgment for whatever Tier 1 couldn't resolve
(``resolve_availability``), plus a second, unrelated job Tier 1 has no way
to do at all - spotting notable positive coverage
(``find_notable_positive_coverage``). See agents/news/__init__.py for why
these two live in the same tier but produce different contract fields.

Both jobs follow the Stats agent's reasoning.py pattern exactly: retrieval
+ Prompt Shields happen BEFORE the LLM ever sees anything (deterministic,
testable), the LLM call is wrapped so ANY failure (network, bad JSON, empty
completion) falls back to something safe rather than propagating an
exception into the debate, and the LLM is never the thing that invents a
player or a number out of nothing - it only ever judges text it was
actually handed.
"""

from __future__ import annotations

import json
import logging
import os

from agents.stats.schemas import Recommendation, Veto, VetoStatus

from . import retrieval, severity_classifier
from .prompt_shields import screen_passages

logger = logging.getLogger("news_agent.tier2")

FOUNDRY_DEPLOYMENT = os.getenv("NEWS_AGENT_LLM_MODEL", "gpt-5.4-nano")
_ENDPOINT_ENV = "MICROSOFT_FOUNDRY_OPENAI_ENDPOINT"
_KEY_ENV = "MICROSOFT_FOUNDRY_KEY"

AVAILABILITY_SYSTEM_PROMPT = (
    "You are the availability specialist on a Fantasy Premier League selection committee. "
    "You are given one player and a handful of retrieved news passages about them. "
    "Classify the player's availability as exactly one of OUT, DOUBT, or FIT, based ONLY on "
    "the passages given - never assume or invent anything not in the text. "
    'Respond with ONLY a JSON object: {"status": "OUT"|"DOUBT"|"FIT", "confidence": 0.0-1.0, '
    '"grounding_snippet": "<the exact phrase from a passage that justifies this>"}. '
    "If the passages are inconclusive, prefer DOUBT with lower confidence over guessing OUT or FIT."
)

POSITIVE_COVERAGE_SYSTEM_PROMPT = (
    "You are the news specialist on a Fantasy Premier League selection committee, looking "
    "for GENUINELY NOTABLE positive coverage - a manager's press-conference praise, a "
    "breakout performance write-up, a clear signal a player is about to start. "
    "Be conservative: routine, neutral, or purely factual coverage (a price change, a generic "
    "team-news mention) is NOT notable and must not be included. Only include a player if the "
    "retrieved text itself would make a football fan sit up. "
    'Respond with ONLY a JSON object: {"notable": [{"player_id": <int>, "conviction": 0.0-1.0}], '
    '"reasoning": "<2-3 sentences citing the actual retrieved text for whichever picks you made, '
    'or a short note that nothing notable was found>"}.'
)


def _default_client():
    # Near-identical to reasoning.py's and embeddings.py's _default_client() -
    # see embeddings.py's module docstring for why this isn't factored out.
    from openai import OpenAI

    return OpenAI(api_key=os.environ[_KEY_ENV], base_url=os.environ[_ENDPOINT_ENV], timeout=12.0, max_retries=1)


def _llm_disabled() -> bool:
    return os.getenv("NEWS_AGENT_DISABLE_LLM") == "1"


def classify_availability_zero_shot(
    player_name: str, passages: list[str], *, client_factory=_default_client
) -> dict:
    """The raw zero-shot LLM call, given already-retrieved, already-screened
    passage text - no DB, no retrieval, no Prompt Shields in here.

    NOT the production path anymore: ``severity_finetune_eval.py`` measured
    this against the embedding-based classifier in ``severity_classifier.py``
    on the same real held-out test set and the classifier won by a wide,
    real margin (macro-F1 0.58 vs 0.192 - see docs/severity_classifier.md).
    ``resolve_availability`` below now calls the classifier. This function is
    kept as-is (not deleted) because the eval script calls it directly to
    reproduce that comparison, and because it's the honest zero-shot
    baseline that comparison is measured against - deleting it would make
    the comparison unreproducible.

    Returns ``{"status": VetoStatus, "confidence": float, "grounding_snippet": str | None}``.
    Raises on any failure (bad JSON, unexpected status value, network) -
    callers decide what the fallback should be.
    """
    client = client_factory()
    completion = client.chat.completions.create(
        model=FOUNDRY_DEPLOYMENT,
        messages=[
            {"role": "system", "content": AVAILABILITY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"player": player_name, "passages": passages})},
        ],
        max_completion_tokens=300,
        response_format={"type": "json_object"},
    )
    data = json.loads(completion.choices[0].message.content or "{}")
    return {
        "status": VetoStatus(data["status"]),  # raises if the model returns anything else
        "confidence": float(data.get("confidence", 0.5)),
        "grounding_snippet": data.get("grounding_snippet"),
    }


def resolve_availability(
    conn, player_id: int, player_name: str, fallback: Veto | None, *, client_factory=_default_client
) -> Veto | None:
    """The ambiguous middle Tier 1 left unresolved: retrieve team_news
    passages for this player, screen them, classify with the fine-tuned
    embedding classifier (see ``severity_classifier.py`` - chosen over
    zero-shot per the real evaluation in ``severity_finetune_eval.py``).

    ``fallback`` is whatever Tier 1 already determined (which is ``None``
    here by construction - this is only called when Tier 1 didn't resolve
    the player - but kept as an explicit parameter so the fallback path
    below and the caller's intent both read the same way: "if Tier 2 can't
    do better, fall back to Tier 1's answer", never "assume fit"). It's also
    what's returned if the classifier artifact hasn't been trained/deployed
    yet (``severity_classifier.MODEL_PATH`` missing) - never a silent guess.

    ``client_factory`` is unused by the classifier path; kept as a parameter
    for test call-site compatibility with the earlier zero-shot signature.
    """
    if _llm_disabled():
        return fallback
    try:
        passages = retrieval.retrieve_passages(
            conn, player_id, corpus="team_news", query_text=retrieval.AVAILABILITY_QUERY, top_k=5
        )
        safe_texts = screen_passages([p.chunk_text for p in passages])
        if not safe_texts:
            return fallback

        result = severity_classifier.classify_availability(player_name, safe_texts)
        return Veto(
            player_id=player_id, status=result["status"],
            confidence=result["confidence"],
            grounding_snippet=result["grounding_snippet"],
        )
    except Exception:
        logger.warning("Tier 2 availability call failed for player %s; falling back to Tier 1", player_id, exc_info=True)
        return fallback


def find_notable_positive_coverage(
    conn, players: list[tuple[int, str]], *, client_factory=_default_client
) -> tuple[list[Recommendation], str | None]:
    """Scan general_news passages across ``players`` (list of (player_id,
    name)); returns (recommendations, reasoning_text). Conservative by
    design - see POSITIVE_COVERAGE_SYSTEM_PROMPT: an empty list is the
    correct, expected result on a routine gameweek, not a failure.

    Returns ``([], None)`` on any failure or when the LLM is disabled -
    "no recommendation" rather than guessing, per the task's own
    instruction for this fallback path.
    """
    if _llm_disabled():
        return [], None
    try:
        candidates = []
        for player_id, name in players:
            passages = retrieval.retrieve_passages(
                conn, player_id, corpus="general_news", query_text=retrieval.POSITIVE_COVERAGE_QUERY, top_k=3
            )
            if not passages:
                continue
            safe = screen_passages([p.chunk_text for p in passages])
            if safe:
                candidates.append({"player_id": player_id, "name": name, "passages": safe})

        if not candidates:
            return [], None

        client = client_factory()
        completion = client.chat.completions.create(
            model=FOUNDRY_DEPLOYMENT,
            messages=[
                {"role": "system", "content": POSITIVE_COVERAGE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"candidates": candidates})},
            ],
            max_completion_tokens=500,
            response_format={"type": "json_object"},
        )
        data = json.loads(completion.choices[0].message.content or "{}")
        recs = [
            Recommendation(player_id=item["player_id"], conviction=float(item.get("conviction", 0.5)))
            for item in data.get("notable", [])
        ]
        return recs, data.get("reasoning")
    except Exception:
        logger.warning("Tier 2 positive-coverage call failed; returning no recommendations", exc_info=True)
        return [], None
