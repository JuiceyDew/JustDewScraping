"""Reddit threads found through a search engine, not through Reddit.

Reddit closed the anonymous doors: `www.reddit.com/….json` returns 403 on every
endpoint regardless of user-agent, `old.reddit.com` redirects to a login wall, and
fetching a thread page returns an 8 KB JavaScript shell that yields zero extractable
text. Verified 2026-09-14.

What still works is asking a search engine. `site:reddit.com <query>` returns
titles, permalinks and snippets, and the snippets carry real verbatim material:

    "I was so excited to get my Plunge, that I paid like $4,000 for. I can't get
     over how LOUD it is. It's like a full hot tub running."

That is exactly the kind of line a voice-of-customer report is built on. This
collector therefore trades depth for reach: no comment trees, no scores, roughly a
sentence or two per result -- but it never contacts Reddit, so it cannot be
throttled or blocked by them, and the search engine does the keyword matching that
Arctic Shift times out on.

Two consequences worth stating plainly:

  * Search results have no reliable publication date. These documents are stamped
    with collection time and use the platform `reddit-search`, which
    analyze/signals.py excludes from every time series. Counting them would put
    the whole corpus in the current week and inflate momentum, which is exactly
    the bug that undated web articles caused.
  * Engagement is unknown, so quote ranking cannot favour endorsed comments here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone

from ideafindr.collectors.base import is_on_topic, topic_terms
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

# /r/<sub>/comments/<id>/<slug>
_THREAD = re.compile(r"reddit\.com/r/([A-Za-z0-9_]+)/comments/([a-z0-9]+)")
MIN_SNIPPET_CHARS = 60


def _search(query: str, n: int) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as d:
        return list(d.text(query, max_results=n))


def _doc(hit: dict, now: datetime) -> Document | None:
    url = hit.get("href") or hit.get("url") or ""
    m = _THREAD.search(url)
    if not m:
        return None
    sub, native = m.group(1), m.group(2)
    title = (hit.get("title") or "").strip()
    body = " ".join((hit.get("body") or "").split())
    # Search engines repeat the title at the head of the snippet; it adds nothing
    # twice and skews the language bank.
    if title and body.startswith(title):
        body = body[len(title):].strip(" -–—:·")
    if len(body) < MIN_SNIPPET_CHARS and len(title) < MIN_SNIPPET_CHARS:
        return None

    return Document(
        # Deliberately NOT "reddit:<id>": if the same thread is also collected in
        # full by a Reddit collector, the full version must win rather than being
        # deduped away by a snippet that happens to share its id.
        id=f"redditsearch:{native}",
        platform="reddit-search",
        kind="post",
        url=url.split("?")[0],
        title=title or None,
        text=body,
        author=None,  # search results do not name the poster
        created_at=now,  # collection time; see the module docstring
        community=sub,
        engagement={},
        raw={"source": "ddg", "subreddit": sub, "thread_id": native},
    )


class DorkCollector:
    """Reddit discovery via `site:reddit.com` search."""

    name = "dork"

    def __init__(self, per_query: int = 12):
        self.per_query = per_query

    def queries(self, plan: RunPlan) -> list[str]:
        """Search-engine queries aimed at opinionated threads.

        The topic alone returns product pages and news. Pairing it with the
        planner's keywords, and with the words people use when they complain,
        surfaces the discussion instead.
        """
        base = plan.topic
        out = [f'site:reddit.com {base}']
        out += [f'site:reddit.com "{base}" {kw}' for kw in plan.keywords[:5]]
        out += [
            f'site:reddit.com "{base}" problems',
            f'site:reddit.com "{base}" worth it',
            f'site:reddit.com "{base}" vs',
        ]
        return out

    async def collect(self, plan: RunPlan, limit: int = 200) -> list[Document]:
        now = datetime.now(timezone.utc)
        docs: dict[str, Document] = {}
        terms = topic_terms(plan)

        for q in self.queries(plan):
            if len(docs) >= limit:
                break
            try:
                hits = await asyncio.to_thread(_search, q, self.per_query)
            except Exception as e:  # noqa: BLE001
                log.warning("dork search failed for %r: %s", q, e)
                continue
            found = 0
            for h in hits:
                d = _doc(h, now)
                if d and not is_on_topic(d, terms, min_terms=2):
                    continue
                if d and d.id not in docs:
                    docs[d.id] = d
                    found += 1
            log.info("dork %r -> %d thread(s)", q[:56], found)

        out = list(docs.values())[:limit]
        log.info("dork: %d reddit threads via search (snippets only, undated)", len(out))
        return out
