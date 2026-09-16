"""Entity linking: match player names/surnames appearing in scraped text to
``player_id``, at INGESTION time (see ``ingest.py``) - not left to semantic
search alone at query time.

Why not just rely on embedding similarity at query time? Retrieval already
filters by player_id first, then ranks by similarity (see
``retrieval.py``) - that filter has to come from somewhere, and a passage's
own text is the only place it can come from. Doing that match once, up
front, and storing it (``news_passage_players``) means retrieval is a plain
indexed lookup, not "hope the embedding for 'Salah' is close enough to a
passage that never happened to say his name explicitly" - and it means the
one tricky part (ambiguous surnames) is solved once per passage, not
re-guessed on every query.

AMBIGUOUS SURNAMES - THE RULE
------------------------------
FPL's ``web_name`` (what news text actually uses - "Silva", "Gomes") is not
unique; several current top-flight squads carry more than one player known
by the same surname. The rule, in order:

  1. Find every player whose ``web_name`` (case-insensitive, whole word)
     appears in the passage text.
  2. If exactly one such player exists league-wide, link the passage to
     that player_id. Done - most surnames are unique.
  3. If MORE than one player shares that surname, try to disambiguate using
     the player's TEAM name: if the passage text also mentions exactly one
     of the candidates' team names, link to that player only.
  4. If step 3 still leaves more than one candidate (the surname is shared
     AND no team name in the text narrows it down, or the team mention
     itself is ambiguous), the match is NOT made. It is recorded as
     "ambiguous" (surname + all candidate player_ids) rather than guessed -
     a wrong link would silently misattribute a real injury/praise to the
     wrong player, which is worse than not linking at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Player:
    player_id: int
    web_name: str
    team_name: str


@dataclass
class LinkResult:
    player_ids: list[int] = field(default_factory=list)
    # (surname, [candidate player_ids]) pairs that could not be resolved -
    # surfaced so ingest.py can log them, per the task's "don't fail
    # silently" spirit, rather than swallow the ambiguity.
    ambiguous: list[tuple[str, list[int]]] = field(default_factory=list)


def _word_in_text(word: str, text_lower: str) -> bool:
    # A plain `\b` boundary treats "." as non-word, so a bare surname like
    # "Sarr" would falsely match inside "M.Sarr" or "P.M.Sarr" - FPL's own
    # convention for disambiguating two players with the same surname is a
    # dotted-initial prefix, exactly the case this needs to reject. Found by
    # actually running entity linking against real scraped data (three
    # distinct real "Sarr" players all matched to one of them) and confirmed
    # against the real `players` table, not assumed. The left boundary
    # additionally excludes a preceding ".", the right boundary is unchanged.
    return re.search(rf"(?<![\w.]){re.escape(word.lower())}\b", text_lower) is not None


def link_players(text: str, players: list[Player]) -> LinkResult:
    """Return every player this passage can be confidently linked to, plus
    any surname matches that stayed ambiguous.
    """
    text_lower = text.lower()
    result = LinkResult()

    by_surname: dict[str, list[Player]] = {}
    for p in players:
        by_surname.setdefault(p.web_name.lower(), []).append(p)

    for surname, candidates in by_surname.items():
        if not _word_in_text(surname, text_lower):
            continue

        if len(candidates) == 1:
            result.player_ids.append(candidates[0].player_id)
            continue

        # Shared surname - try to narrow down by team name mentioned in the text.
        team_matches = [c for c in candidates if _word_in_text(c.team_name, text_lower)]
        if len(team_matches) == 1:
            result.player_ids.append(team_matches[0].player_id)
        else:
            result.ambiguous.append((surname, [c.player_id for c in candidates]))

    return result
