"""Orchestration: plan -> collect -> analyse.

Kept separate from the CLI so the same flow can be driven from a notebook, a
scheduler, or a future API without dragging typer along.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from datetime import datetime, timezone

from ideafindr.analyze.cluster import cluster_documents
from ideafindr.analyze.label import label_themes
from ideafindr.analyze.plan import build_plan, fallback_plan
from ideafindr.analyze.quotes import attach_quotes
from ideafindr.analyze.signals import compute_theme_signals
from ideafindr.collectors.base import safe_collect
from ideafindr.collectors.reddit_arctic import RedditArcticCollector
from ideafindr.config import settings
from ideafindr.llm import LLMNotConfigured, usage_summary as llm_usage
from ideafindr.models import Document, RunPlan, Theme
from ideafindr.store import db

log = logging.getLogger(__name__)

ALL_SOURCES = ["reddit", "web", "dork", "hackernews", "lemmy", "stackexchange",
               "x", "bluesky", "instagram", "tiktok",
               "github", "mastodon", "producthunt"]


def make_run_id(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40] or "run"
    return f"{slug}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def plan_for(topic: str, days: int) -> RunPlan:
    try:
        return build_plan(topic, days=days)
    except LLMNotConfigured:
        log.warning("no LLM key configured; using a keyword-only fallback plan")
        return fallback_plan(topic, days=days)


# Collectors that need no extra to be installed. Each entry is
# (module, class). Imported lazily so an optional dependency missing for one
# source never breaks a run that does not use it.
_KEYLESS = {
    "web": ("ideafindr.collectors.web", "WebCollector"),
    "dork": ("ideafindr.collectors.dork", "DorkCollector"),
    "hackernews": ("ideafindr.collectors.hackernews", "HackerNewsCollector"),
    "lemmy": ("ideafindr.collectors.lemmy", "LemmyCollector"),
    "stackexchange": ("ideafindr.collectors.stackexchange", "StackExchangeCollector"),
    "bluesky": ("ideafindr.collectors.bluesky", "BlueskyCollector"),
    "x": ("ideafindr.collectors.x_twitter", "XCollector"),
}

# Collectors that come from an optional extra and may raise on import.
_OPTIONAL = {
    "tiktok": ("ideafindr.collectors.tiktok", "TikTokCollector"),
    "instagram": ("ideafindr.collectors.instagram", "InstagramCollector"),
    "github": ("ideafindr.collectors.github", "GitHubCollector"),
    "mastodon": ("ideafindr.collectors.mastodon", "MastodonCollector"),
    "producthunt": ("ideafindr.collectors.producthunt", "ProductHuntCollector"),
}


def _instantiate(module: str, cls: str) -> object | None:
    try:
        mod = __import__(module, fromlist=[cls])
        return getattr(mod, cls)()
    except Exception as e:  # noqa: BLE001 - a missing extra must not kill a run
        log.warning("%s unavailable: %s", cls, e)
        return None


def build_collectors(sources: list[str]) -> list:
    """Resolve source names to collector instances.

    TikTok and Instagram stay off unless their enable flag is set, matching the
    README: both scrape against platform ToS and should be a deliberate choice.
    """
    out: list = []
    for source in sources:
        if source == "reddit":
            if settings.reddit_client_id and settings.reddit_client_secret:
                from ideafindr.collectors.reddit_api import RedditAPICollector
                log.info("reddit: using the official API (credentials found)")
                out.append(RedditAPICollector())
            else:
                log.info("reddit: using Arctic Shift (no credentials); expect some "
                         "keyword queries to time out")
                out.append(RedditArcticCollector())
            continue

        if source in ("tiktok", "instagram"):
            enabled = settings.enable_tiktok if source == "tiktok" else settings.enable_instagram
            if not enabled:
                log.info("%s: skipped (disabled; enable it in settings)", source)
                continue

        spec = _KEYLESS.get(source) or _OPTIONAL.get(source)
        if not spec:
            continue
        if (c := _instantiate(*spec)) is not None:
            log.info("%s: collector enabled", source)
            out.append(c)
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
        summary = llm_usage()
        if summary["calls"] or summary["cached"]:
            log.info(
                "llm: %d calls, %d cached, %d prompt + %d completion tokens",
                summary["calls"], summary["cached"],
                summary["prompt_tokens"], summary["completion_tokens"],
            )
        return themes
    finally:
        con.close()


def delete_run(run_id: str) -> str:
    """Remove a run: its rows in every table, plus its rendered report directory.

    Lives here rather than in a UI module so the CLI, the web UI and any future
    caller delete identically. Returns a short human summary.
    """
    con = db.connect()
    try:
        if not db.get_run(con, run_id):
            raise ValueError(f"unknown run: {run_id}")
        counts = db.delete_run(con, run_id)
    finally:
        con.close()

    reports = settings.reports_dir / run_id
    if reports.exists():
        shutil.rmtree(reports, ignore_errors=True)
    return f"deleted {counts['documents']} documents, {counts['themes']} themes"


def render_report(run_id: str, summary: str = "") -> tuple:
    """Render the Markdown + self-contained HTML for an analysed run.

    Shared by the CLI, the web job and simple.py so all three produce identical
    output. `summary` is the optional LLM-written executive summary. Returns
    (md_path, html_path).
    """
    from ideafindr.analyze.signals import corpus_stats, language_bank, share_of_voice
    from ideafindr.report.render import render

    con = db.connect()
    try:
        run = db.get_run(con, run_id)
        if not run:
            raise ValueError(f"unknown run: {run_id}")
        themes = db.load_themes(con, run_id)
        if not themes:
            raise ValueError(f"run {run_id} has no themes; analyse it first")
        docs = list(db.iter_documents(con, run_id))
        demand = db.load_demand_clusters(con, run_id)
    finally:
        con.close()

    return render(
        run=run, themes=themes, stats=corpus_stats(docs),
        language=language_bank(docs),
        sov=share_of_voice(docs, run.plan.brands),
        summary=summary, demand=demand,
    )
