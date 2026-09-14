"""Orchestration: plan -> collect -> analyse.

Kept separate from the CLI so the same flow can be driven from a notebook, a
scheduler, or a future API without dragging typer along.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone

from ideafindr.analyze.cluster import cluster_documents
from ideafindr.analyze.label import label_themes
from ideafindr.analyze.plan import build_plan, fallback_plan
from ideafindr.analyze.quotes import attach_quotes
from ideafindr.analyze.signals import compute_theme_signals
from ideafindr.collectors.base import safe_collect
from ideafindr.collectors.reddit_arctic import RedditArcticCollector
from ideafindr.collectors.web import WebCollector
from ideafindr.config import settings
from ideafindr.llm import LLMNotConfigured
from ideafindr.models import Document, RunPlan, Theme
from ideafindr.store import db

log = logging.getLogger(__name__)

ALL_SOURCES = ["reddit", "web", "tiktok", "instagram"]


def make_run_id(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40] or "run"
    return f"{slug}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def plan_for(topic: str, days: int) -> RunPlan:
    try:
        return build_plan(topic, days=days)
    except LLMNotConfigured:
        log.warning("no LLM key configured; using a keyword-only fallback plan")
        return fallback_plan(topic, days=days)


def build_collectors(sources: list[str]) -> list:
    """Optional scrapers are imported lazily: their dependencies are an extra, and
    an ImportError must not break a run that doesn't use them."""
    out: list = []
    if "reddit" in sources:
        out.append(RedditArcticCollector())
    if "web" in sources:
        out.append(WebCollector())
    if "tiktok" in sources and settings.enable_tiktok:
        try:
            from ideafindr.collectors.tiktok import TikTokCollector

            out.append(TikTokCollector())
        except Exception as e:  # noqa: BLE001
            log.warning("tiktok collector unavailable: %s", e)
    if "instagram" in sources and settings.enable_instagram:
        try:
            from ideafindr.collectors.instagram import InstagramCollector

            out.append(InstagramCollector())
        except Exception as e:  # noqa: BLE001
            log.warning("instagram collector unavailable: %s", e)
    return out


async def collect(plan: RunPlan, sources: list[str], limit: int = 600) -> list[Document]:
    collectors = build_collectors(sources)
    if not collectors:
        return []
    results = await asyncio.gather(
        *(safe_collect(c, plan, limit) for c in collectors)
    )
    docs: dict[str, Document] = {}
    for batch in results:
        for d in batch:
            docs.setdefault(d.id, d)
    return list(docs.values())


def run_collection(topic: str, sources: list[str], days: int, limit: int, plan: RunPlan | None = None) -> tuple[str, int]:
    plan = plan or plan_for(topic, days)
    run_id = make_run_id(topic)
    con = db.connect()
    try:
        db.create_run(con, run_id, plan, sources)
        docs = asyncio.run(collect(plan, sources, limit))
        n = db.save_documents(con, run_id, docs)
        return run_id, n
    finally:
        con.close()


def run_demand(
    run_id: str,
    topic: str | None = None,
    depth: int = 2,
    label: bool = True,
    reuse: bool = False,
    progress=None,
) -> list:
    """Harvest the query space for a run's topic, cluster it, and join it to the
    conversation themes already stored for that run.

    Safe to run on a run that has never been analysed: coverage simply comes back
    zero everywhere, and every gap collapses to the same value.
    """
    from ideafindr.demand import analyze as danalyze
    from ideafindr.demand import gap as dgap
    from ideafindr.demand.harvest import harvest

    con = db.connect()
    try:
        run = db.get_run(con, run_id)
        if not run and topic is None:
            raise ValueError(f"unknown run: {run_id}")
        subject = topic or (run.topic if run else "")

        if reuse:
            queries = db.load_queries(con, run_id)
            log.info("reusing %d stored query observations", len(queries))
        else:
            queries = asyncio.run(harvest(subject, depth=depth, progress=progress))
        if not queries:
            log.warning("no queries harvested for %r", subject)
            return []
        if not reuse:
            stored = db.save_queries(con, run_id, queries)
            log.info("stored %d query observations", stored)

        clusters = danalyze.cluster_queries(queries)
        if label and settings.ollama_api_key:
            clusters = danalyze.label_clusters(clusters, subject)
        elif label:
            log.warning("no LLM key configured; demand clusters keep keyword labels")
            for c in clusters:
                c.label = ", ".join(c.keywords[:4]) or c.label

        themes = db.load_themes(con, run_id)
        if themes:
            clusters = dgap.attach_coverage(clusters, run_id, themes, subject, con=con)
        else:
            log.warning("run %s has no themes; gaps will not reflect coverage", run_id)
        clusters = dgap.compute_gaps(clusters)
        db.save_demand_clusters(con, run_id, clusters)
        return clusters
    finally:
        con.close()


def run_analysis(run_id: str, n_clusters: int | None = None, label: bool = True) -> list[Theme]:
    con = db.connect()
    try:
        run = db.get_run(con, run_id)
        if not run:
            raise ValueError(f"unknown run: {run_id}")
        docs = list(db.iter_documents(con, run_id))
        if not docs:
            raise ValueError(f"run {run_id} has no documents")

        # `kept` is what M is aligned to -- cluster_documents drops content-free docs.
        themes, M, kept = cluster_documents(docs, n_clusters=n_clusters)

        # Decide about labelling once, up front. label_themes() catches Exception
        # per theme, so a missing key never propagates here -- it just burns the
        # full tenacity backoff (~6s) on every single theme before falling back to
        # exactly the keyword labels below. That is two minutes of nothing on a
        # 19-theme run.
        if label and not settings.ollama_api_key:
            log.warning("no LLM key configured; themes keep keyword labels")
            label = False

        if label:
            themes = label_themes(themes, kept, M, run.topic)
        else:
            # What --no-label advertises: "keyword labels only", not "Theme 3".
            for t in themes:
                t.label = ", ".join(t.keywords[:4]) or t.label
        themes = attach_quotes(themes, kept, M)
        themes = compute_theme_signals(themes, kept)
        db.save_themes(con, run_id, themes)
        # Hand the vectors to the bridge so a report doesn't re-embed the corpus.
        db.save_embeddings(con, run_id, [d.id for d in kept], M)
        return themes
    finally:
        con.close()
