"""Scraper for FPL's "Team News" / availability corpus.

INVESTIGATION (per the task's own instruction to check before writing a
scraper): ``fantasy.premierleague.com`` is a client-side-rendered React SPA
- a plain HTTP GET (even to ``/robots.txt``) returns the same empty
``<div id="root"><p></p></div>`` shell and a JS bundle; there is no server-
rendered HTML to parse. Confirmed live with a real request during this
build. Playwright (a real browser) is therefore required, not optional -
this is exactly the case CLAUDE.md's ingestion conventions and this task's
instructions anticipated.

robots.txt: the fantasy subdomain has no dedicated robots.txt of its own
(any path falls through to the same SPA shell); the platform's real
robots.txt (``www.premierleague.com/robots.txt``) allows general content
browsing (it only disallows tracking query parameters). Politeness here is
enforced the same way ``ingestion/fpl_client.py`` enforces it for the
official API: a descriptive User-Agent and a delay before the request.

THE PAGE: ``fantasy.premierleague.com/en/the-scout/player-news`` ("Availability"
tab under The Scout) is a filterable table - Player / Team / Position /
News - already covering every player FPL currently has a fitness note on.
Verified structure: a plain ``<table><tbody><tr><td>...</td></tr></table>``;
each row's first cell contains the player's name/team/position as separate
lines, the second cell the free-text status. This overlaps with what
``bootstrap-static`` already gives Tier 1 (see ``tier1.py``) - the point of
scraping it too is to feed the SAME text into Tier 2's RAG pipeline
uniformly alongside the general-news corpus, and to catch phrasing this
page sometimes carries that the raw API fields don't (fuller context
sentences, not just a percentage).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

USER_AGENT = (
    "FPL-Agents-Manager/0.1 "
    "(+https://github.com/EliDehaene01/Fantasy-Premier-League-Manager; "
    "personal portfolio project, read-only content use)"
)
URL = "https://fantasy.premierleague.com/en/the-scout/player-news"

# Same "be a reasonable citizen" delay convention as ingestion/fpl_client.py.
REQUEST_DELAY_SECONDS = 0.75


@dataclass
class TeamNewsRow:
    name: str
    team: str
    position: str
    news_text: str


def scrape_team_news(*, headless: bool = True) -> list[TeamNewsRow]:
    """One Playwright page load, one table read - not one request per
    player, matching the same "one call covers everyone" politeness
    principle ``ingestion/backfill.py`` uses ``event/{gw}/live`` for.
    """
    from playwright.sync_api import sync_playwright

    time.sleep(REQUEST_DELAY_SECONDS)
    rows_out: list[TeamNewsRow] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(URL, wait_until="networkidle", timeout=30_000)
            for row in page.query_selector_all("table tbody tr"):
                cells = row.query_selector_all("td")
                if len(cells) < 2:
                    continue
                name_cell_lines = [
                    line.strip() for line in cells[0].inner_text().split("\n") if line.strip()
                ]
                # First line is a combined "Name, Team" accessibility label;
                # the real fields are the lines after it. Be defensive about
                # the exact count (site markup can shift) rather than assume
                # a fixed index blindly.
                fields = name_cell_lines[1:] if len(name_cell_lines) >= 4 else name_cell_lines
                if len(fields) < 3:
                    continue
                name, team, position = fields[0], fields[1], fields[2]
                news_text = cells[1].inner_text().strip()
                if news_text:
                    rows_out.append(TeamNewsRow(name=name, team=team, position=position, news_text=news_text))
        finally:
            browser.close()
    return rows_out
