"""GitHub collector via the REST API.

GitHub's API is generous: 5000 requests/hour unauthenticated, 15000/hour with
a free token. We collect issues, discussions, and comments from repositories
relevant to the topic.

Free tier: 5000 req/hr (unauthenticated), 15000 req/hr (free token)
ToS: Official API, fully allowed for commercial use
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ideafindr.collectors.base import Collector
from ideafindr.models import Document, RunPlan

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
MAX_PER_PAGE = 100


# Common tech-related repos to search for popular topics
DEFAULT_REPOS = {
    "cold plunge": ["coldplunge/coldplunge", "hubermanlab/tools"],
    "sauna": ["sauna/sauna", "wellness/heat"],
    "fitness": ["tensorflow/tensorflow", "pytorch/pytorch"],
}


class GitHubCollector(Collector):
    """Collect issues and discussions from GitHub."""

    name = "github"

    def __init__(
        self,
        token: str = "",
        timeout: int = 60,
        max_repos: int = 10,
    ) -> None:
        self.token = token
        self.timeout = timeout
        self.max_repos = max_repos
        self._client: httpx.Client | None = None

    def _ensure_client(self) -> httpx.Client:
        if not self._client:
            headers = {
                "User-Agent": "ideafindr/0.1 (market research)",
                "Accept": "application/vnd.github.v3+json",
            }
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            self._client = httpx.Client(
                base_url=GITHUB_API,
                timeout=httpx.Timeout(60.0, connect=15.0, read=45.0),
                headers=headers,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _search_repos(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for repositories matching the query."""
        client = self._ensure_client()
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(limit, MAX_PER_PAGE),
        }
        
        r = client.get("/search/repositories", params=params)
        r.raise_for_status()
        data = r.json()
        
        repos = data.get("items", [])
        log.info("github: found %d repos for %r", len(repos), query)
        return repos

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _get_issues(
        self, owner: str, repo: str, query: str = "", limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get issues from a repository."""
        client = self._ensure_client()
        
        # Search issues within repo
        q = f"repo:{owner}/{repo} is:issue"
        if query:
            q += f" {query}"
        
        params = {
            "q": q,
            "sort": "created",
            "order": "desc",
            "per_page": min(limit, MAX_PER_PAGE),
        }
        
        r = client.get("/search/issues", params=params)
        r.raise_for_status()
        data = r.json()
        
        issues = data.get("items", [])
        log.info("github: found %d issues in %s/%s", len(issues), owner, repo)
        return issues

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _get_comments(self, issue_url: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get comments for an issue or PR."""
        client = self._ensure_client()
        params = {"per_page": min(limit, MAX_PER_PAGE)}
        
        r = client.get(issue_url + "/comments", params=params)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()

    def _parse_issue(self, issue: dict[str, Any], repo_full_name: str) -> Document | None:
        """Parse a GitHub issue into our Document schema."""
        try:
            # Extract fields
            title = issue.get("title", "")
            body = issue.get("body", "") or ""
            user = issue.get("user", {})
            author = user.get("login", "anonymous")
            created_at_str = issue.get("created_at", "")
            
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now(timezone.utc)
            
            # Engagement metrics
            comments_count = issue.get("comments", 0)
            reactions = issue.get("reactions", {})
            total_reactions = sum(reactions.values()) if isinstance(reactions, dict) else 0
            
            # Build document
            html_url = issue.get("html_url", "")
            issue_number = issue.get("number", 0)
            native_id = f"{repo_full_name}#{issue_number}"
            doc_id = f"github:{native_id}"
            
            # Combine title and body
            text = f"{title}\n\n{body}".strip()
            
            return Document(
                id=doc_id,
                platform="github",
                kind="post",
                url=html_url,
                title=title,
                text=text,
                author=f"@{author}",
                created_at=created_at,
                parent_id=None,
                community=f"r/{repo_full_name}",
                engagement={
                    "comments": comments_count,
                    "reactions": total_reactions,
                    "score": comments_count * 2 + total_reactions,
                },
                raw=issue,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("github: failed to parse issue: %s", e)
            return None

    def _parse_comment(
        self, comment: dict[str, Any], parent_issue: dict[str, Any], repo_full_name: str
    ) -> Document:
        """Parse a GitHub comment into our Document schema."""
        user = comment.get("user", {})
        author = user.get("login", "anonymous")
        created_at_str = comment.get("created_at", "")
        
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = datetime.now(timezone.utc)
        
        html_url = comment.get("html_url", "")
        comment_id = comment.get("id", 0)
        native_id = f"{repo_full_name}#c{comment_id}"
        doc_id = f"github:{native_id}"
        
        parent_url = parent_issue.get("html_url", "")
        parent_number = parent_issue.get("number", 0)
        parent_id = f"github:{repo_full_name}#{parent_number}"
        
        return Document(
            id=doc_id,
            platform="github",
            kind="comment",
            url=html_url,
            title=None,
            text=comment.get("body", "") or "",
            author=f"@{author}",
            created_at=created_at,
            parent_id=parent_id,
            community=f"r/{repo_full_name}",
            engagement={"score": 1},  # Comments don't have reactions in list view
            raw=comment,
        )

    async def collect(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        """Collect GitHub issues and discussions.
        
        Strategy:
        1. Search for repos matching topic/keywords
        2. Get issues from those repos
        3. Get top comments for each issue
        4. Deduplicate by issue/comment ID
        """
        docs: dict[str, Document] = {}
        repos_seen: set[str] = set()
        
        # Build search queries
        queries: list[str] = []
        
        # Add topic
        if plan.topic:
            queries.append(plan.topic)
        
        # Add keywords
        for kw in plan.keywords[:5]:  # Limit to avoid too many searches
            queries.append(kw)
        
        # Add brand + keyword
        for brand in plan.brands[:3]:
            for kw in plan.keywords[:3]:
                queries.append(f"{brand} {kw}")
        
        log.info("github: searching %d queries", len(queries))
        
        # Find relevant repos
        for query in queries:
            try:
                repos = self._search_repos(query, limit=5)
                
                for repo in repos:
                    full_name = repo.get("full_name", "")
                    if full_name and full_name not in repos_seen:
                        repos_seen.add(full_name)
                        log.info("github: collecting from %s", full_name)
                        
                        # Get issues
                        owner, repo_name = full_name.split("/")
                        issues = self._get_issues(owner, repo_name, query="", limit=30)
                        
                        for issue in issues:
                            # Parse issue
                            doc = self._parse_issue(issue, full_name)
                            if doc:
                                # Check date window
                                days_old = (datetime.now(timezone.utc) - doc.created_at).days
                                if days_old <= plan.days:
                                    docs.setdefault(doc.id, doc)
                            
                            # Get comments for top issues
                            if len(docs) < limit:
                                html_url = issue.get("html_url", "")
                                if html_url:
                                    comments = self._get_comments(html_url, limit=10)
                                    for comment in comments[:5]:  # Top 5 comments
                                        comment_doc = self._parse_comment(
                                            comment, issue, full_name
                                        )
                                        if comment_doc:
                                            days_old = (
                                                datetime.now(timezone.utc)
                                                - comment_doc.created_at
                                            ).days
                                            if days_old <= plan.days:
                                                docs.setdefault(comment_doc.id, comment_doc)
                        
                        if len(docs) >= limit:
                            log.info("github: reached limit of %d documents", limit)
                            break
                
                if len(docs) >= limit:
                    break
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    log.warning("github: rate limited (API limit reached)")
                else:
                    log.error("github: HTTP error for %r: %s", query, e)
            except Exception as e:  # noqa: BLE001
                log.error("github: error collecting for %r: %s", query, e)
        
        result = list(docs.values())
        log.info("github: collected %d documents", len(result))
        return result

    def collect_sync(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        """Synchronous wrapper for collect()."""
        import asyncio
        return asyncio.run(self.collect(plan, limit))
