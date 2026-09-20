#!/usr/bin/env python3
"""Simple ideafindr: topic in, HTML report out.

A scriptable one-shot version of the web UI's research flow, for cron or a
terminal where the server is the wrong shape.

Usage:
    python simple.py "cold plunge tubs"
    python simple.py "sauna tents" --days 90 --sources reddit,web,bluesky
"""

from __future__ import annotations

import argparse

from ideafindr import pipeline
from ideafindr.config import settings


def research(topic: str, days: int, sources: list[str]) -> str | None:
    print(f"Researching: {topic}")
    print(f"Window: last {days} days · sources: {', '.join(sources)}\n")

    print("1. Planning…")
    plan = pipeline.plan_for(topic, days)
    print(f"   {len(plan.subreddits)} communities, {len(plan.keywords)} keywords")

    print("2. Collecting…")
    run_id, n = pipeline.run_collection(topic, sources, days, 600, plan=plan)
    if not n:
        print("   No conversations found. Try a broader topic or different sources.")
        return None
    print(f"   {n} documents  ·  run {run_id}")

    print("3. Analysing…")
    themes = pipeline.run_analysis(run_id)
    print(f"   {len(themes)} themes")

    print("4. Writing report…")
    md, html = pipeline.render_report(run_id)
    print(f"   {md}\n   {html}\n")

    print("Key themes:")
    for i, t in enumerate(themes[:5], 1):
        arrow = "↑" if t.momentum > 1.2 else "↓" if t.momentum < 0.8 else "→"
        print(f"  {i}. {t.label} {arrow} ({t.volume})")
    print(f"\nOpen: file://{html}")
    return str(html)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-shot social listening report.")
    ap.add_argument("topic")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument(
        "--sources", default="reddit,web,bluesky",
        help="Comma-separated sources (default: reddit,web,bluesky). "
             "x and instagram need cookies entered in the web UI.",
    )
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in pipeline.ALL_SOURCES]
    if unknown:
        ap.error(f"unknown source(s): {', '.join(unknown)}; valid: {', '.join(pipeline.ALL_SOURCES)}")

    settings.ensure_dirs()
    research(args.topic, args.days, sources)


if __name__ == "__main__":
    main()
