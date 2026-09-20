"""ideafindr CLI."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ideafindr import pipeline
from ideafindr.config import settings
from ideafindr.report.render import momentum_tag
from ideafindr.store import db

app = typer.Typer(
    add_completion=False,
    help="Find out what people are actually saying -- and searching for -- about any topic.",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main(ctx: typer.Context) -> None:
    """Launch the web UI when no subcommand is given.

    The web UI is the primary interface; the subcommands remain for scripting and
    for anywhere a full-screen app is the wrong shape (CI, pipes, ssh one-liners).
    """
    if ctx.invoked_subcommand is None:
        web(host=None, port=None)


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


def _parse_sources(sources: str) -> list[str]:
    """Split and validate --sources.

    Unvalidated, `--sources reddt` silently builds no collectors and the run
    ends with 0 documents and no explanation of why.
    """
    srcs = [s.strip().lower() for s in sources.split(",") if s.strip()]
    unknown = [s for s in srcs if s not in pipeline.ALL_SOURCES]
    if unknown:
        console.print(f"[red]Unknown source(s):[/] {', '.join(unknown)}")
        console.print(f"[dim]Valid sources: {', '.join(pipeline.ALL_SOURCES)}[/]")
        raise typer.Exit(1)
    if not srcs:
        console.print("[red]No sources given.[/]")
        raise typer.Exit(1)
    return srcs


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
    srcs = _parse_sources(sources)
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

    md, html = pipeline.render_report(rid, summary=summary)
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
    srcs = _parse_sources(sources)
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


def _show_demand(clusters) -> None:
    t = Table(header_style="bold")
    t.add_column("#", width=3, justify="right")
    t.add_column("Search intent", overflow="fold")
    t.add_column("Queries", justify="right", width=7)
    t.add_column("Demand", justify="right", width=7)
    t.add_column("Corpus", justify="right", width=6)
    t.add_column("Gap", justify="right", width=6)
    for i, c in enumerate(clusters, 1):
        colour = "green" if c.gap > 0.25 else ("dim" if c.gap < -0.25 else "white")
        t.add_row(str(i), c.label, str(len(c.unique_texts())),
                  f"{c.demand_score:.1f}", str(c.coverage),
                  f"[{colour}]{c.gap:+.2f}[/]")
    console.print(t)
    console.print(
        "[dim]Gap = demand percentile − corpus-coverage percentile. Positive means "
        "people search it more than your corpus discusses it.[/]"
    )


@app.command()
def demand(
    topic: str = typer.Argument(None, help="Defaults to the most recent run's topic."),
    run_id: str = typer.Option(None, "--run", help="Attach to this run instead of the newest."),
    depth: int = typer.Option(2, help="1 = expand the topic only. 2 = also expand what it finds."),
    no_label: bool = typer.Option(False, "--no-label", help="Skip LLM intent naming."),
    reuse: bool = typer.Option(False, "--reuse", help="Re-cluster stored queries without re-harvesting."),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Harvest what people SEARCH for, and find the gaps your corpus doesn't cover."""
    _setup(verbose)
    rid = _resolve_run(run_id)
    msg = "Re-clustering stored queries…" if reuse else "Harvesting the query space…"
    with console.status(msg):
        clusters = pipeline.run_demand(rid, topic=topic, depth=depth,
                                       label=not no_label, reuse=reuse)
    if not clusters:
        console.print("[yellow]No queries harvested.[/] The suggest endpoints returned nothing.")
        raise typer.Exit(1)
    _show_demand(clusters)
    n = len({q.key for c in clusters for q in c.queries})
    console.print(f"\n[dim]run {rid} · {n} distinct queries · {len(clusters)} intents[/]")


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
def trends(
    run_id: str = typer.Argument(None, help="Defaults to the most recent run."),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Validate a run's momentum against Google Trends (needs the trends extra)."""
    _setup(verbose)
    rid = _resolve_run(run_id)
    con = db.connect()
    try:
        run = db.get_run(con, rid)
        themes = db.load_themes(con, rid) if run else []
    finally:
        con.close()
    if not run:
        console.print(f"[red]Unknown run:[/] {rid}")
        raise typer.Exit(1)

    from ideafindr.analyze.trends import PYTRENDS_AVAILABLE, validate_momentum

    if not PYTRENDS_AVAILABLE:
        console.print(
            "[yellow]pytrends is not installed.[/] Install it with:\n"
            "  uv sync --extra trends\n"
            "[dim]This command is optional; the report does not need it.[/]"
        )
        raise typer.Exit(1)

    # The run's overall momentum: mean across coherent themes, so the search-side
    # number is compared against what the corpus actually shows.
    coherent = [t for t in themes if t.coherent] or themes
    social = sum(t.momentum for t in coherent) / len(coherent) if coherent else 1.0
    res = validate_momentum(social, [run.topic, *run.plan.keywords[:3]], run.plan.days)
    console.print(f"topic: [bold]{run.topic}[/]")
    console.print(
        f"social momentum {res.get('social_momentum')} · "
        f"search momentum {res.get('search_momentum')} · "
        f"alignment {res.get('alignment')} ({res.get('confidence')})"
    )
    if note := res.get("note"):
        console.print(f"[dim]{note}[/]")


@app.command()
def delete(
    run_id: str = typer.Argument(..., help="The run to delete."),
    yes: bool = typer.Option(False, "-y", help="Skip the confirmation."),
) -> None:
    """Delete a run: its documents, themes, search demand and rendered reports."""
    settings.ensure_dirs()
    con = db.connect()
    try:
        run = db.get_run(con, run_id)
    finally:
        con.close()
    if not run:
        console.print(f"[red]Unknown run:[/] {run_id}")
        raise typer.Exit(1)
    console.print(f"[bold]{run.topic}[/]  ·  {run.doc_count} documents  ·  {run_id}")
    if not yes and not typer.confirm("Delete this run? It cannot be undone.", default=False):
        raise typer.Abort()

    console.print(f"[green]Deleted.[/] {pipeline.delete_run(run_id)}")


@app.command()
def web(
    host: str = typer.Option(None, help="Bind address (default 0.0.0.0)."),
    port: int = typer.Option(None, help="Port (default 8000)."),
) -> None:
    """Open the web UI (same as running `ideafindr` bare)."""
    settings.ensure_dirs()
    from ideafindr.web.app import serve

    shown = host or settings.web_host
    # 0.0.0.0 is a bind address, not a destination; show a navigable URL.
    if shown in ("0.0.0.0", "::"):
        shown = "127.0.0.1"
    console.print(f"ideafindr on [bold]http://{shown}:{port or settings.web_port}[/]  (Ctrl-C to stop)")
    serve(host=host, port=port)


@app.command()
def doctor() -> None:
    """Check external dependencies (same probes as scripts/preflight.py)."""
    import subprocess, sys

    from ideafindr.config import ROOT

    # Resolved from the package, not the cwd: `ideafindr doctor` must work from
    # anywhere, the same as every other command.
    subprocess.run([sys.executable, str(ROOT / "scripts" / "preflight.py")], check=False)


if __name__ == "__main__":
    app()
