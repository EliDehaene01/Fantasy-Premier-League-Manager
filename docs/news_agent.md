# News agent

Availability veto + notable-coverage recommendations. See ARCHITECTURE.md
section 4 for the design rationale; this doc covers the concrete
implementation: corpus sources, entity linking, and the Tier 1/Tier 2 split.

## Status

Tier 1 + Tier 2, fully live against real data. Availability severity in
Tier 2 is classified by a fine-tuned embedding classifier
(`agents/news/severity_classifier.py`), not the zero-shot LLM call it was
originally built with - the classifier was evaluated against that zero-shot
baseline on real held-out data and won by a wide margin. See
docs/severity_classifier.md for the full comparison and deploy decision.

**Vector store**: pgvector 0.8.6 is installed and live on the Postgres
instance at `DATABASE_URL` (an earlier report in this project that it
wasn't installed was a stale check against the wrong moment in time, not a
real gap). 188 team_news + 89 general_news passages are ingested with real
embeddings and entity links. `ingestion/db.py::ensure_pgvector_schema` still
fails clearly (rather than silently) if pointed at a Postgres instance
without the extension, and `ingest.py`/`service.py` still degrade
gracefully (Tier 1 unaffected; Tier 2 falls back per-call) in that case -
kept as a real safety net, not the path actually running now.

## Tier 1 vs. Tier 2 - why split at all

Cost and reliability, not just code structure (see `agents/news/__init__.py`'s
docstring for the full reasoning): a plain threshold check on fields already
in silver is free and can't misread an unambiguous case; an LLM call is
neither. Tier 1 runs first and resolves the large majority of players
outright; Tier 2 only runs for the genuine leftover - the ambiguous middle,
plus a second job (notable positive coverage) Tier 1 has no way to do at all.

### Tier 1 (`agents/news/tier1.py`) - exact rules

Given `(chance_of_playing_this_round, news)`:

1. `chance_of_playing_this_round == 0` -> **OUT**, confidence 1.0
2. `news` matches an explicit ruled-out/suspension phrase (case-insensitive:
   "ruled out", "suspended", "will not play", "out for the season",
   "long-term injury") -> **OUT**, confidence 1.0
3. `chance_of_playing_this_round == 100`, or (chance is `None` **and** news
   is empty) -> **FIT**, confidence 1.0
4. Anything else (25/50/75%, or text that doesn't match an OUT phrase but
   isn't obviously nothing) -> **unresolved** (`None`) - handed to Tier 2,
   never guessed at.

### Tier 2 (`agents/news/tier2.py`) - two distinct jobs

- **Availability nuance** (`resolve_availability`): only for players Tier 1
  left unresolved. Retrieves team_news passages for that player, screens
  them through Prompt Shields, then classifies OUT/DOUBT/FIT with a
  confidence and a grounding snippet using the fine-tuned embedding
  classifier (`severity_classifier.classify_availability` - see
  docs/severity_classifier.md for why this replaced the original zero-shot
  LLM call as the production path; the zero-shot function,
  `classify_availability_zero_shot`, is kept in `tier2.py` as the evaluated
  baseline, not deleted). Falls back to Tier 1's answer (`None`, in practice
  - by construction this is only called on unresolved players) if retrieval
  finds nothing, Prompt Shields drops everything, or classification fails.
- **Notable positive coverage** (`find_notable_positive_coverage`): scans
  general_news passages across the whole player pool once per request (not
  once per player). Deliberately conservative - the system prompt tells the
  model routine/neutral coverage is NOT notable. Falls back to an empty
  list (no recommendation) on any failure, never a guess.

Both outputs land on the shared contract (`agents/stats/schemas.py`) but
mean different things to the Manager: `vetoes` is a hard constraint (CLAUDE.md);
`recommendations` is a normal vote, weighed like any other specialist's.

## Corpora

Both live on `fantasy.premierleague.com`, which is a **client-side-rendered
React SPA** - verified live during this build: a plain HTTP request (even to
`/robots.txt`) returns the same empty `<div id="root">` shell and JS bundle,
never real content. Playwright is required for both, not a plain HTTP
request. The platform's real robots.txt (`www.premierleague.com/robots.txt`)
permits general content browsing (only tracking query parameters are
disallowed).

- **Team News** (`agents/news/scraper_team_news.py`) -
  `fantasy.premierleague.com/en/the-scout/player-news` ("Availability" tab).
  A single page load returns a table (Player / Team / Position / News) for
  every player FPL currently has a fitness note on - one page covers
  everyone, not one request per player.
- **General News** (`agents/news/scraper_general_news.py`) -
  `fantasy.premierleague.com/en/the-scout` (headline list, reliably
  rendered) linking out to `www.premierleague.com/en/news/{id}` (full
  articles, a *different* platform). Headlines and metadata there are
  server-rendered, but the article body itself
  (`<div class="article__content" data-flatplan-body>`) is empty in the raw
  HTML and - confirmed live, including after accepting the cookie banner -
  stays empty after full client-side rendering too. This looks like a real
  gating mechanism (account/session state or bot mitigation) this scraper
  doesn't reproduce, not a bug in the selector. The scraper takes what IS
  reliably available (the headline) as its own chunk regardless, and
  attempts the full body as a bonus when it happens to succeed - it does
  not pad or fabricate content to compensate.

Politeness: a descriptive `User-Agent`, a delay before every page load
(`REQUEST_DELAY_SECONDS`), and a bound on how many article bodies are
fetched per run (`MAX_ARTICLES_TO_FETCH`) - the same conventions
`ingestion/fpl_client.py` already established for the official API.

## Entity linking (`agents/news/entity_linking.py`)

Done at ingestion time, not left to embedding similarity alone at query
time (common surnames make that unreliable). Rule:

1. A player's `web_name` (FPL's own display surname) appears as a whole
   word in the passage text -> candidate.
2. Exactly one candidate for that surname -> linked.
3. More than one player shares the surname -> try to disambiguate by
   whether the passage also names exactly one candidate's team.
4. Still ambiguous -> **not linked**. Recorded (`LinkResult.ambiguous`) and
   logged by `ingest.py` rather than guessed - a wrong link would silently
   misattribute a real injury or piece of praise to the wrong player, worse
   than not linking at all.

## Vector store & retrieval

pgvector on the existing Postgres instance (`ingestion/db.py`'s
`ensure_pgvector_schema`/`register_pgvector_adapter` - kept separate from
the main schema every connection bootstraps, since the extension isn't
guaranteed installed). `news_passages.embedding` is `VECTOR(1536)`
(text-embedding-3-small's native size, via Foundry -
`agents/news/embeddings.py`).

`retrieval.py` filters to passages already linked to the requested
`player_id` FIRST (an indexed join, not a similarity search over the whole
corpus), then ranks by cosine similarity to a fixed reference query -
`AVAILABILITY_QUERY` for Tier 2's availability job, `POSITIVE_COVERAGE_QUERY`
for the notable-coverage scan. `corpus` narrows a call to one corpus or
leaves it covering both, keeping team_news and general_news distinguishable
at retrieval time as required.

## Prompt Shields (`agents/news/prompt_shields.py`)

Every retrieved passage is screened by Content Safety's document-attack
detection (`FPL_CONTENT_SAFETY_ENDPOINT`/`_KEY`, the same credentials
already configured for this project) before it reaches the LLM. A flagged
passage is dropped, never forwarded with a warning label. Fails CLOSED: if
the Content Safety call itself errors, the whole batch is dropped rather
than forwarded unscreened. Every drop (flagged or failure) is logged.
