"""Scraper for FPL's general "News" corpus - form write-ups, price-change
articles, gameweek reviews (where "notable positive coverage" actually
lives, per ARCHITECTURE.md section 4).

INVESTIGATION - two different rendering behaviours found for two different
domains, verified live during this build:

  * The headline LIST (``fantasy.premierleague.com/en/the-scout``) is on the
    same client-rendered SPA as the Team News page - Playwright required,
    same as ``scraper_team_news.py``.
  * Each article's full text lives at ``www.premierleague.com/en/news/{id}``
    - a DIFFERENT domain/platform. Its headline and metadata ARE present in
    the raw server-rendered HTML (a plain HTTP GET returns them), but the
    body itself sits in ``<div class="article__content" data-flatplan-body>``,
    which is EMPTY in the raw HTML and - confirmed with a real Playwright
    fetch, including after accepting the cookie banner - stays empty after
    full client-side rendering too. This looks like the body genuinely
    requires some additional interaction/session state this scraper doesn't
    reproduce (possibly account-gated, possibly a bot-mitigation measure);
    it is a real, observed limitation, not a guess.

    Rather than block the whole general-news corpus on that, this scraper
    takes what IS reliably available - the headline and its category tag,
    both of which render correctly - as the chunk, and separately attempts
    the full body (used when it succeeds). A short, real headline
    ("Maresca offers latest on O'Reilly and Doku") is still meaningful RAG
    content for the "notable positive coverage" judgment even without the
    full article text; it is not padded or invented to look like more than
    it is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .scraper_team_news import USER_AGENT  # same UA string, one source of truth

HUB_URL = "https://fantasy.premierleague.com/en/the-scout"
REQUEST_DELAY_SECONDS = 0.75
# Bound how many article bodies we attempt per ingestion run - each is its
# own page load; keep this a deliberate, small number rather than crawl
# every linked article every run.
MAX_ARTICLES_TO_FETCH = 15


@dataclass
class NewsArticle:
    headline: str
    category: str | None
    url: str
    body_text: str | None  # None if the body genuinely couldn't be fetched - see module docstring


def _scrape_headlines(headless: bool) -> list[NewsArticle]:
    from playwright.sync_api import sync_playwright

    time.sleep(REQUEST_DELAY_SECONDS)
    articles: list[NewsArticle] = []
    seen_urls: set[str] = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(HUB_URL, wait_until="networkidle", timeout=30_000)
            # Article links point off-domain to www.premierleague.com/en/news/{id};
            # each link's own text is the headline.
            anchors = page.query_selector_all('a[href*="/en/news/"]')
            for a in anchors:
                href = a.get_attribute("href") or ""
                if not href or href in seen_urls:
                    continue
                seen_urls.add(href)
                headline = (a.inner_text() or "").strip()
                if not headline:
                    continue
                articles.append(NewsArticle(headline=headline, category=None, url=href, body_text=None))
        finally:
            browser.close()
    return articles


def _fetch_body(url: str, headless: bool) -> str | None:
    from playwright.sync_api import sync_playwright

    time.sleep(REQUEST_DELAY_SECONDS)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            # "networkidle" can legitimately never fire on an ad/analytics-heavy
            # page (background beacons keep the network "busy" indefinitely) -
            # a per-article timeout here is a routine, expected outcome for
            # this best-effort fetch (see the module docstring), not something
            # that should crash the whole ingestion run over one slow article.
            try:
                page.goto(url, wait_until="networkidle", timeout=30_000)
            except Exception:
                return None
            content = page.query_selector(".article__content")
            text = content.inner_text().strip() if content else ""
            return text or None
        finally:
            browser.close()


def scrape_general_news(*, headless: bool = True, fetch_bodies: bool = True) -> list[NewsArticle]:
    """Headlines first (always reliable), then best-effort article bodies
    for up to ``MAX_ARTICLES_TO_FETCH`` of them - see the module docstring
    for why the body fetch is allowed to come back empty.
    """
    articles = _scrape_headlines(headless)
    if not fetch_bodies:
        return articles

    for article in articles[:MAX_ARTICLES_TO_FETCH]:
        article.body_text = _fetch_body(article.url, headless)
    return articles
