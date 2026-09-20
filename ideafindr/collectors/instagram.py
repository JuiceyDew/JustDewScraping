"""Instagram via instaloader. OPTIONAL, OFF BY DEFAULT.

Scraping Instagram is against Meta's Terms of Service -- see the README before
enabling this for client work.

Practical reality as of 2026: hashtag browsing is login-gated and anonymous
requests to the public JSON endpoints start returning HTTP 429 after a handful
of calls from one IP. Expect low yield regardless of effort, and never run this
from more than one account/IP at a time.

Two modes, in descending order of how well they actually work:

  websearch (default) -- find public post URLs via `site:instagram.com/p/ <kw>`
      on DuckDuckGo, then fetch those individually. No login, far fewer requests,
      and it degrades to "fewer results" rather than "banned account".
  hashtag             -- iterate a hashtag feed. Needs a logged-in session and
      will rate-limit; use a burner account, never a client's.

Session setup: enter `sessionid` (and `csrftoken` if you have it) under
Credentials in the web UI, or export them from the browser. They are stored in
the state dir, not the repo. The collector also accepts a legacy session file
written by `instaloader --login <user>`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import timezone

from ideafindr.config import settings
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

_SHORTCODE = re.compile(r"instagram\.com/(?:p|reel)/([A-Za-z0-9_-]+)")
MAX_COMMENTS = 15


def has_session() -> bool:
    """A usable logged-in session: a sessionid cookie, or a legacy session file."""
    if settings.instagram_sessionid:
        return True
    user = settings.instagram_session_user
    if user and (settings.sessions_dir / f"session-{user}").exists():
        return True
    return bool(user)


def _post_to_doc(post, community: str | None = None) -> Document | None:
    caption = (post.caption or "").strip()
    if not caption:
        return None
    return Document(
        id=f"instagram:{post.shortcode}",
        platform="instagram",
        kind="caption",
        url=f"https://www.instagram.com/p/{post.shortcode}/",
        title=None,
        text=caption,
        author=post.owner_username,
        created_at=post.date_utc.replace(tzinfo=timezone.utc),
        community=community,
        engagement={"likes": post.likes, "num_comments": post.comments},
        raw={"shortcode": post.shortcode, "is_video": post.is_video},
    )


def _loader():
    import instaloader

    L = instaloader.Instaloader(
        download_pictures=False, download_videos=False, download_comments=False,
        save_metadata=False, compress_json=False, quiet=True,
        dirname_pattern=str(settings.sessions_dir / "ig"),
    )

    # Preferred: a sessionid entered in the web UI. Install the cookies on
    # instaloader's HTTP session directly, so no `instaloader --login` step is
    # needed and nothing interactive ever runs.
    if settings.instagram_sessionid:
        try:
            cookies = {"sessionid": settings.instagram_sessionid}
            if settings.instagram_csrftoken:
                cookies["csrftoken"] = settings.instagram_csrftoken
            L.context._session.cookies.update(cookies)
            # Validate cheaply; a dead session should warn, not explode mid-run.
            username = L.test_login()
            if username:
                L.context.username = username
                log.info("instagram: using the stored sessionid (user %s)", username)
            else:
                log.warning(
                    "instagram: the stored sessionid was rejected (expired?); "
                    "falling back to public websearch"
                )
        except Exception as e:  # noqa: BLE001
            log.warning("instagram: could not load the stored sessionid: %s", e)

    # Fallback: a session file from `instaloader --login <user>`.
    user = settings.instagram_session_user
    if not L.context.username and user:
        try:
            L.load_session_from_file(user)
            log.info("instagram: loaded session file for %s", user)
        except FileNotFoundError:
            log.warning(
                "instagram: no session for %s; run `instaloader --login %s` or "
                "enter a sessionid in the web UI", user, user,
            )
    return L


def _collect_websearch(plan: RunPlan, limit: int) -> list[Document]:
    """Discover public post URLs via web search, then fetch each one."""
    import instaloader
    from ddgs import DDGS

    terms = plan.hashtags or plan.keywords or [plan.topic]
    codes: list[str] = []
    seen: set[str] = set()
    with DDGS() as d:
        for term in terms[:5]:
            try:
                for r in d.text(f"site:instagram.com/p/ {term}", max_results=12):
                    if m := _SHORTCODE.search(r.get("href") or ""):
                        if m.group(1) not in seen:
                            seen.add(m.group(1))
                            codes.append(m.group(1))
            except Exception as e:  # noqa: BLE001
                log.warning("instagram websearch failed for %r: %s", term, e)

    L = _loader()
    docs: list[Document] = []
    for code in codes[:limit]:
        try:
            post = instaloader.Post.from_shortcode(L.context, code)
            if d_ := _post_to_doc(post, community=plan.topic):
                docs.append(d_)
        except Exception as e:  # noqa: BLE001
            log.debug("instagram post %s failed: %s", code, e)
    return docs


def _collect_hashtag(plan: RunPlan, limit: int) -> list[Document]:
    import instaloader

    L = _loader()
    if not settings.instagram_session_user:
        log.warning("instagram hashtag mode needs a login session; expect HTTP 429 almost immediately")

    docs: list[Document] = []
    for tag in (plan.hashtags or [plan.topic.replace(" ", "")])[:4]:
        if len(docs) >= limit:
            break
        try:
            ht = instaloader.Hashtag.from_name(L.context, tag)
            for i, post in enumerate(ht.get_posts()):
                if i >= limit // 4:
                    break
                if d_ := _post_to_doc(post, community=f"#{tag}"):
                    docs.append(d_)
                    if post.comments > 3:
                        try:
                            for j, c in enumerate(post.get_comments()):
                                if j >= MAX_COMMENTS:
                                    break
                                if not (c.text or "").strip():
                                    continue
                                docs.append(
                                    Document(
                                        id=f"instagram:c{c.id}",
                                        platform="instagram",
                                        kind="comment",
                                        url=d_.url,
                                        text=c.text.strip(),
                                        author=c.owner.username,
                                        created_at=c.created_at_utc.replace(tzinfo=timezone.utc),
                                        parent_id=d_.id,
                                        community=f"#{tag}",
                                        engagement={"likes": getattr(c, "likes_count", 0)},
                                        raw={"id": str(c.id)},
                                    )
                                )
                        except Exception as e:  # noqa: BLE001
                            log.debug("instagram comments failed: %s", e)
        except Exception as e:  # noqa: BLE001
            log.warning("instagram hashtag #%s failed (%s) -- hashtag browsing is "
                        "login-gated and rate-limited", tag, e)
    return docs[:limit]


class InstagramCollector:
    name = "instagram"

    def __init__(self, mode: str = "websearch"):
        self.mode = mode

    async def collect(self, plan: RunPlan, limit: int = 100) -> list[Document]:
        fn = _collect_hashtag if self.mode == "hashtag" else _collect_websearch
        return await asyncio.to_thread(fn, plan, limit)
