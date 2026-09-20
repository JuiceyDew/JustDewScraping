"""Product Hunt collector - new product launches.

Product Hunt is where new products launch daily. Great for:
- Early adopter feedback
- Product positioning insights
- Competitive launches
- Feature requests and complaints

Free tier: Official API (500 req/hour)
ToS: Allows commercial use with attribution
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

PH_API = "https://api.producthunt.com/v2/api/graphql"


class ProductHuntCollector(Collector):
    """Collect posts and comments from Product Hunt."""

    name = "producthunt"

    def __init__(self, token: str = "", timeout: int = 30) -> None:
        self.token = token
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _ensure_client(self) -> httpx.Client:
        if not self._client:
            headers = {
                "User-Agent": "ideafindr/0.1 (market research)",
                "Content-Type": "application/json",
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            
            self._client = httpx.Client(
                base_url=PH_API,
                timeout=self.timeout,
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
    def _search_posts(self, query: str, first: int = 20) -> list[dict[str, Any]]:
        """Search Product Hunt posts."""
        client = self._ensure_client()
        
        # GraphQL query
        graphql_query = """
        query Search($query: String!, $first: Int!) {
          search(query: $query, first: $first, type: POST) {
            edges {
              node {
                ... on Post {
                  id
                  name
                  tagline
                  description
                  url
                  votesCount
                  commentsCount
                  createdAt
                  maker {
                    name
                    username
                  }
                  topics {
                    name
                  }
                }
              }
            }
          }
        }
        """
        
        variables = {"query": query, "first": first}
        
        try:
            r = client.post("", json={"query": graphql_query, "variables": variables})
            r.raise_for_status()
            data = r.json()
            
            edges = data.get("data", {}).get("search", {}).get("edges", [])
            posts = [edge["node"] for edge in edges if "node" in edge]
            
            log.info("producthunt: found %d posts for %r", len(posts), query)
            return posts
            
        except httpx.HTTPError as e:
            log.warning("producthunt: search error: %s", e)
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _get_comments(self, post_id: str) -> list[dict[str, Any]]:
        """Get comments for a post."""
        client = self._ensure_client()
        
        graphql_query = """
        query Post($id: ID!) {
          post(id: $id) {
            comments {
              edges {
                node {
                  id
                  body
                  createdAt
                  user {
                    name
                    username
                  }
                  childComments {
                    edges {
                      node {
                        id
                        body
                        createdAt
                        user {
                          name
                          username
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        variables = {"id": post_id}
        
        try:
            r = client.post("", json={"query": graphql_query, "variables": variables})
            r.raise_for_status()
            data = r.json()
            
            edges = data.get("data", {}).get("post", {}).get("comments", {}).get("edges", [])
            comments = [edge["node"] for edge in edges if "node" in edge]
            
            # Flatten nested comments
            all_comments = []
            for comment in comments:
                all_comments.append(comment)
                child_edges = comment.get("childComments", {}).get("edges", [])
                for child in child_edges:
                    if "node" in child:
                        all_comments.append(child["node"])
            
            return all_comments
            
        except httpx.HTTPError as e:
            log.warning("producthunt: comments error: %s", e)
            return []

    def _parse_post(self, post: dict[str, Any]) -> Document | None:
        """Parse a Product Hunt post."""
        try:
            name = post.get("name", "")
            tagline = post.get("tagline", "")
            description = post.get("description", "")
            
            text = f"{name}: {tagline}\n\n{description}".strip()
            if not text:
                return None
            
            # Engagement
            votes = post.get("votesCount", 0)
            comments = post.get("commentsCount", 0)
            
            # Timestamp
            created_at_str = post.get("createdAt", "")
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, AttributeError):
                created_at = datetime.now(timezone.utc)
            
            # Author
            maker = post.get("maker", {})
            author = maker.get("name", "") or maker.get("username", "anonymous")
            
            # Topics
            topics = [t.get("name", "") for t in post.get("topics", []) if t.get("name")]
            community = ", ".join(topics[:3]) if topics else "producthunt"
            
            # IDs
            post_id = post.get("id", "")
            doc_id = f"producthunt:{post_id}"
            url = post.get("url", f"https://www.producthunt.com/posts/{post_id}")
            
            return Document(
                id=doc_id,
                platform="producthunt",
                kind="post",
                url=url,
                title=name,
                text=text,
                author=f"@{author}",
                created_at=created_at,
                parent_id=None,
                community=community,
                engagement={
                    "upvotes": votes,
                    "comments": comments,
                    "score": votes + 2 * comments,
                },
                raw=post,
            )
        except Exception as e:
            log.warning("producthunt: failed to parse post: %s", e)
            return None

    def _parse_comment(self, comment: dict[str, Any], parent_post: dict[str, Any]) -> Document:
        """Parse a Product Hunt comment."""
        user = comment.get("user", {})
        author = user.get("name", "") or user.get("username", "anonymous")
        
        created_at_str = comment.get("createdAt", "")
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except (ValueError, AttributeError):
            created_at = datetime.now(timezone.utc)
        
        comment_id = comment.get("id", "")
        doc_id = f"producthunt:c{comment_id}"
        post_id = parent_post.get("id", "")
        parent_id = f"producthunt:{post_id}"
        
        url = parent_post.get("url", "")
        
        return Document(
            id=doc_id,
            platform="producthunt",
            kind="comment",
            url=url,
            title=None,
            text=comment.get("body", "") or "",
            author=f"@{author}",
            created_at=created_at,
            parent_id=parent_id,
            community=parent_post.get("name", "producthunt"),
            engagement={"score": 1},
            raw=comment,
        )

    async def collect(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        """Collect Product Hunt posts and comments."""
        docs: dict[str, Document] = {}
        
        # Build queries
        queries = []
        if plan.topic:
            queries.append(plan.topic)
        for kw in plan.keywords[:5]:
            queries.append(kw)
        for brand in plan.brands[:3]:
            queries.append(brand)
        
        log.info("producthunt: searching %d queries", len(queries))
        
        for query in queries:
            try:
                posts = self._search_posts(query, first=20)
                
                for post in posts:
                    # Parse post
                    doc = self._parse_post(post)
                    if doc:
                        days_old = (datetime.now(timezone.utc) - doc.created_at).days
                        if days_old <= plan.days:
                            docs.setdefault(doc.id, doc)
                    
                    # Get comments
                    if len(docs) < limit:
                        post_id = post.get("id", "")
                        comments = self._get_comments(post_id)
                        for comment in comments[:10]:  # Top 10 comments
                            comment_doc = self._parse_comment(comment, post)
                            if comment_doc:
                                days_old = (datetime.now(timezone.utc) - comment_doc.created_at).days
                                if days_old <= plan.days:
                                    docs.setdefault(comment_doc.id, comment_doc)
                    
                    if len(docs) >= limit:
                        break
                        
            except Exception as e:
                log.error("producthunt: error for %r: %s", query, e)
            
            if len(docs) >= limit:
                break
        
        result = list(docs.values())
        log.info("producthunt: collected %d documents", len(result))
        return result

    def collect_sync(self, plan: RunPlan, limit: int = 600) -> list[Document]:
        """Synchronous wrapper."""
        import asyncio
        return asyncio.run(self.collect(plan, limit))
