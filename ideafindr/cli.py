"""ideafindr CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ideafindr import pipeline
from ideafindr.analyze.signals import corpus_stats, language_bank, share_of_voice
from ideafindr.config import settings
from ideafindr.report.render import momentum_tag, render
from ideafindr.store import db

app = typer.Typer(add_completion=False, help="Find out what people are actually saying about any topic.")
console = Console()


def _setup(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings.ensure_dirs()


def _resolve_run(run_id: str | None) -> str:
    con = db.connect()
    try:
        rid = run_id or db.latest_run_id(con)
        if not rid:
            console.print("[red]No runs yet.[/] Start one with: ideafindr research \"<topic>\"")
            raise typer.Exit(1)
        return rid
    finally:
        con.close()


def _show_plan(plan) -> None:
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column(style="dim", width=12)
    t.add_column()
    t.add_row("topic", plan.topic)
    t.add_row("window", f"last {plan.days} days")
    t.add_row("subreddits", ", ".join(plan.subreddits) or "[dim]none[/]")
    t.add_row("keywords", ", ".join(plan.keywords) or "[dim]none[/]")
    t.add_row("brands", ", ".join(plan.brands) or "[dim]none[/]")
    t.add_row("web", "\n".join(plan.web_queries) or "[dim]none[/]")
    console.print(t)


def _show_themes(themes) -> None:
    table = Table(title=None, header_style="bold")
    table.add_column("#", width=3, justify="right")
    table.add_column("Theme", overflow="fold")
    table.add_column("Docs", justify="right", width=5)
    table.add_column("Mom.", justify="right", width=6)
    table.add_column("Stance", width=11)
    for i, t in enumerate(themes, 1):
        cls, _ = momentum_tag(t.momentum)
        color = {"rising": "green", "fading": "yellow"}.get(cls, "white")
        table.add_row(str(i), t.label, str(t.volume),
                      f"[{color}]{t.momentum:.2f}[/]", t.stance.replace("_", " "))
    console.print(table)


@app.command()
def plan(
    topic: str,
    days: int = typer.Option(180, help="Look back this many days."),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Show the collection plan for a topic without collecting anything."""
    _setup(verbose)
    p = pipeline.plan_for(topic, days)
    _show_plan(p)


@app.command()
def collect(
    topic: str,
    days: int = typer.Option(180),
    sources: str = typer.Option("reddit,web", help="Comma-separated: reddit,web,tiktok,instagram"),
    limit: int = typer.Option(600, help="Rough cap on documents collected."),
    yes: bool = typer.Option(False, "-y", help="Skip plan confirmation."),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Collect a corpus for a topic."""
    _setup(verbose)
    srcs = [s.strip() for s in sources.split(",") if s.strip()]
    p = pipeline.plan_for(topic, days)
    _show_plan(p)
    if not yes and not typer.confirm("\nCollect with this plan?", default=True):
        raise typer.Abort()

    with console.status("Collecting…"):
        run_id, n = pipeline.run_collection(topic, srcs, days, limit, plan=p)
    console.print(f"\n[green]{n}[/] documents collected  ·  run [bold]{run_id}[/]")


@app.command()
def analyze(
    run_id: str = typer.Argument(None, help="Defaults to the most recent run."),
    clusters: int = typer.Option(None, "--clusters", "-k", help="Force a theme count."),
    no_label: bool = typer.Option(False, "--no-label", help="Skip LLM naming (keyword labels only)."),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Cluster a corpus into named themes with quotes and signals."""
    _setup(verbose)
    rid = _resolve_run(run_id)
    with console.status("Clustering and labelling…"):
        themes = pipeline.run_analysis(rid, n_clusters=clusters, label=not no_label)
    _show_themes(themes)
    console.print(f"\n[dim]run {rid} · {len(themes)} themes[/]")


@app.command()
def report(
    run_id: str = typer.Argument(None),
    engine: str = typer.Option("none", help="Executive summary engine: gpt-researcher | none"),
    no_web: bool = typer.Option(False, "--no-web", help="Corpus only, no live web search."),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Render a run to Markdown + self-contained HTML."""
    _setup(verbose)
    rid = _resolve_run(run_id)
    con = db.connect()
    try:
        run = db.get_run(con, rid)
        themes = db.load_themes(con, rid)
        docs = list(db.iter_documents(con, rid))
    finally:
        con.close()

    if not run:
        console.print(f"[red]Unknown run:[/] {rid}")
        raise typer.Exit(1)
    if not themes:
        console.print("[yellow]No themes yet.[/] Run: ideafindr analyze " + rid)
        raise typer.Exit(1)

    summary = ""
    if engine == "gpt-researcher":
        from ideafindr.engines import researcher

        with console.status("Writing executive summary (gpt-researcher over your corpus)…"):
            try:
                summary = researcher.write_report(run.topic, rid, include_web=not no_web)
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]Summary engine failed:[/] {e}")
                console.print("[dim]Rendering the report without it.[/]")

    md, html = render(
        run=run, themes=themes, stats=corpus_stats(docs),
        language=language_bank(docs), sov=share_of_voice(docs, run.plan.brands),
        summary=summary,
    )
    console.print(f"\n[green]Report written[/]\n  {md}\n  {html}")


@app.command()
def research(
    topic: str,
    days: int = typer.Option(180),
    sources: str = typer.Option("reddit,web"),
    limit: int = typer.Option(600),
    clusters: int = typer.Option(None, "--clusters", "-k"),
    engine: str = typer.Option("none", help="Executive summary engine: gpt-researcher | none"),
    yes: bool = typer.Option(False, "-y"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Collect, analyse and report on a topic, end to end."""
    _setup(verbose)
    srcs = [s.strip() for s in sources.split(",") if s.strip()]
    p = pipeline.plan_for(topic, days)
    _show_plan(p)
    if not yes and not typer.confirm("\nRun with this plan?", default=True):
        raise typer.Abort()

    with console.status("Collecting…"):
        rid, n = pipeline.run_collection(topic, srcs, days, limit, plan=p)
    console.print(f"[green]{n}[/] documents  ·  run [bold]{rid}[/]")
    if n == 0:
        console.print("[yellow]Nothing collected.[/] Try different subreddits or a longer window.")
        raise typer.Exit(1)

    with console.status("Clustering and labelling…"):
        themes = pipeline.run_analysis(rid, n_clusters=clusters)
    _show_themes(themes)

    report(run_id=rid, engine=engine, no_web=False, verbose=verbose)


@app.command()
def ask(
    question: str,
    run_id: str = typer.Option(None, "--run", help="Defaults to the most recent run."),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Ask a question of an already-collected corpus (deep-searcher)."""
    _setup(verbose)
    rid = _resolve_run(run_id)
    from ideafindr.engines import searcher

    with console.status("Searching the corpus…"):
        answer = searcher.ask(rid, question)
    console.print(answer)


@app.command()
def runs() -> None:
    """List past runs."""
    settings.ensure_dirs()
    con = db.connect()
    try:
        rows = db.list_runs(con)
    finally:
        con.close()
    if not rows:
        console.print("[dim]No runs yet.[/]")
        return
    t = Table(header_style="bold")
    t.add_column("Run"); t.add_column("Topic"); t.add_column("Docs", justify="right")
    t.add_column("Sources"); t.add_column("When", style="dim")
    for r in rows:
        t.add_row(r.id, r.topic, str(r.doc_count), ",".join(r.sources),
                  r.created_at.strftime("%Y-%m-%d %H:%M"))
    console.print(t)


@app.command()
def bridge(verbose: bool = typer.Option(False, "-v")) -> None:
    """Run the retriever bridge standalone (for debugging or external agents)."""
    _setup(verbose)
    from ideafindr.bridge.retriever_api import serve

    console.print(f"Bridge on [bold]{settings.bridge_url}/retrieve[/]  (Ctrl-C to stop)")
    serve()


@app.command()
def doctor() -> None:
    """Check external dependencies (same probes as scripts/preflight.py)."""
    import subprocess, sys

    subprocess.run([sys.executable, "scripts/preflight.py"], check=False)


if __name__ == "__main__":
    app()
