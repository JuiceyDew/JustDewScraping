"""The Ideafindr TUI -- the primary interface.

Why this exists rather than more CLI: every interesting thing this tool does takes
minutes and produces more output than a terminal table can hold. `collect` showed
a spinner reading "Collecting…" for several minutes with no indication of which
subreddit it was on or how much it had found; the corpus had a full-text index
with no way to search it; and comparing two runs of the same topic meant reading
two 55KB HTML files side by side.

Theming: the app runs with `ansi_color = True` and an ANSI theme, so every colour
resolves to one of the terminal's own 16 palette entries rather than a hardcoded
RGB value. It therefore matches whatever colour scheme the terminal already uses,
light or dark, instead of imposing its own.

Threading: the pipeline is synchronous and calls asyncio.run() internally
(pipeline.run_collection), which raises inside Textual's running event loop. Every
pipeline call therefore goes through a threaded worker, and progress comes back
through a logging handler rather than by refactoring every collector to take a
callback.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from ideafindr import pipeline
from ideafindr.analyze.signals import corpus_stats
from ideafindr.store import db

log = logging.getLogger(__name__)


class TuiLogHandler(logging.Handler):
    """Pipe the pipeline's own log records into the activity panel.

    The collectors already log the things worth watching -- which subreddits
    resolved, how many posts came back, what got dropped as off-topic, when Arctic
    Shift asked us to slow down. Capturing those gives a live view of a run
    without threading a progress callback through every collector.
    """

    def __init__(self, app: "IdeafindrApp") -> None:
        super().__init__(level=logging.INFO)
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001
            return
        style = {
            logging.WARNING: "yellow",
            logging.ERROR: "red",
            logging.CRITICAL: "red",
        }.get(record.levelno, "")
        # Log records arrive on worker threads; the UI must be touched on the
        # app's own thread.
        try:
            self._app.call_from_thread(self._app.write_activity, msg, style)
        except Exception:  # noqa: BLE001 - app shutting down
            pass


class IdeafindrApp(App):
    """Browse runs, search the corpus, and drive the pipeline with live status."""

    # Use the terminal's own palette instead of hardcoded RGB.
    ansi_color = True

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #runs-pane { width: 34%; border: round $foreground 30%; }
    #detail-pane { width: 1fr; border: round $foreground 30%; }
    #activity { height: 22%; border: round $foreground 30%; }
    .pane-title { padding: 0 1; text-style: bold; }
    #status { padding: 0 1; height: 1; }
    DataTable { height: 1fr; }
    Input { border: round $foreground 30%; }
    #search-results { height: 1fr; }
    """

    BINDINGS = [
        ("n", "new_run", "New run"),
        ("a", "analyze", "Analyze"),
        ("d", "demand", "Demand"),
        ("r", "report", "Report"),
        ("s", "focus_search", "Search"),
        ("c", "compare", "Compare"),
        ("f5", "refresh", "Refresh"),
        ("ctrl+t", "cycle_theme", "Theme"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.run_ids: list[str] = []
        self.busy = False
        self._awaiting_topic = False

    # --- layout ---------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Ready", id="status")
        with Horizontal(id="body"):
            with Vertical(id="runs-pane"):
                yield Label("Runs", classes="pane-title")
                yield DataTable(id="runs", cursor_type="row", zebra_stripes=True)
            with Vertical(id="detail-pane"):
                with TabbedContent(id="tabs"):
                    with TabPane("Themes", id="tab-themes"):
                        yield DataTable(id="themes", cursor_type="row")
                    with TabPane("Demand", id="tab-demand"):
                        yield DataTable(id="demand", cursor_type="row")
                    with TabPane("Search", id="tab-search"):
                        yield Input(
                            placeholder="Search this run's corpus…", id="search-input"
                        )
                        yield DataTable(id="search-results", cursor_type="row")
                    with TabPane("Compare", id="tab-compare"):
                        yield DataTable(id="compare", cursor_type="row")
        yield RichLog(id="activity", highlight=False, markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "ansi-dark"
        self.title = "Ideafindr"
        self.sub_title = "say · search · gaps"

        # Widths are explicit because auto-sizing spends the space on the widest
        # column and truncates the rest -- "Stance" came out as "Sta"/"neu".
        runs = self.query_one("#runs", DataTable)
        runs.add_column("Topic", width=18)
        runs.add_column("Docs", width=5)
        runs.add_column("When", width=11)

        themes = self.query_one("#themes", DataTable)
        themes.add_column("#", width=3)
        themes.add_column("Theme", width=44)
        themes.add_column("Docs", width=5)
        themes.add_column("Mom.", width=5)
        themes.add_column("Stance", width=10)

        demand = self.query_one("#demand", DataTable)
        demand.add_column("#", width=3)
        demand.add_column("Search intent", width=40)
        demand.add_column("Qs", width=4)
        demand.add_column("Demand", width=7)
        demand.add_column("Corpus", width=6)
        demand.add_column("Gap", width=6)

        search = self.query_one("#search-results", DataTable)
        search.add_column("Where", width=16)
        search.add_column("Text", width=64)
        search.add_column("Date", width=11)

        comp = self.query_one("#compare", DataTable)
        comp.add_column("Metric", width=16)
        comp.add_column("A", width=26)
        comp.add_column("B", width=26)

        handler = TuiLogHandler(self)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger("ideafindr")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.write_activity("Ideafindr ready. n = new run, s = search, ? = keys.")
        self.action_refresh()

    # --- helpers --------------------------------------------------------------

    def write_activity(self, message: str, style: str = "") -> None:
        panel = self.query_one("#activity", RichLog)
        stamp = datetime.now().strftime("%H:%M:%S")
        body = f"[{style}]{message}[/{style}]" if style else message
        panel.write(f"[dim]{stamp}[/dim] {body}")

    def set_status(self, text: str, working: bool = False) -> None:
        marker = "⠿ " if working else ""
        self.query_one("#status", Static).update(f"{marker}{text}")

    def selected_run(self) -> str | None:
        table = self.query_one("#runs", DataTable)
        if not self.run_ids or table.cursor_row is None:
            return None
        if not (0 <= table.cursor_row < len(self.run_ids)):
            return None
        return self.run_ids[table.cursor_row]

    def guard_busy(self) -> bool:
        """True when a job is already running, so actions don't stack."""
        if self.busy:
            self.write_activity("A job is already running; wait for it.", "yellow")
            return True
        return False

    # --- loading --------------------------------------------------------------

    def action_refresh(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.clear()
        self.run_ids = []
        con = db.connect()
        try:
            rows = db.list_runs(con)
        finally:
            con.close()
        for r in rows:
            self.run_ids.append(r.id)
            table.add_row(
                r.topic[:18], str(r.doc_count), r.created_at.strftime("%m-%d %H:%M")
            )
        if rows:
            self.load_detail(rows[0].id)
            self.set_status(f"{len(rows)} run(s)")
        else:
            self.set_status("No runs yet — press n to start one")

    def load_detail(self, run_id: str) -> None:
        con = db.connect()
        try:
            themes = db.load_themes(con, run_id)
            clusters = db.load_demand_clusters(con, run_id)
        finally:
            con.close()

        tt = self.query_one("#themes", DataTable)
        tt.clear()
        for i, t in enumerate(themes, 1):
            tt.add_row(str(i), t.label[:44], str(t.volume),
                       f"{t.momentum:.2f}", t.stance.replace("_", " "))

        dt = self.query_one("#demand", DataTable)
        dt.clear()
        for i, c in enumerate(clusters, 1):
            dt.add_row(str(i), c.label[:40], str(len(c.unique_texts())),
                       f"{c.demand_score:.1f}", str(c.coverage), f"{c.gap:+.2f}")

        if not themes:
            self.write_activity(f"{run_id} has no themes yet — press a to analyze.")
        if not clusters:
            self.write_activity(f"{run_id} has no demand data — press d to harvest.")

    @on(DataTable.RowHighlighted, "#runs")
    def on_run_highlighted(self, event: DataTable.RowHighlighted) -> None:
        rid = self.selected_run()
        if rid:
            self.load_detail(rid)
            self.set_status(rid)

    # --- search ---------------------------------------------------------------

    def action_focus_search(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-search"
        self.query_one("#search-input", Input).focus()

    @on(Input.Submitted, "#search-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """One handler, two modes -- the input doubles as the topic prompt.

        Textual dispatches an event to every matching handler, so these cannot be
        two separate @on methods on the same selector: both would fire and a
        search would also start a collection run.
        """
        if self._awaiting_topic:
            self._awaiting_topic = False
            inp = self.query_one("#search-input", Input)
            inp.placeholder = "Search this run's corpus…"
            inp.value = ""
            topic = event.value.strip()
            if topic:
                self.collect_worker(topic)
            return
        self._do_search(event.value)

    def _do_search(self, value: str) -> None:
        rid = self.selected_run()
        if not rid:
            self.write_activity("No run selected.", "yellow")
            return
        query = value.strip()
        results = self.query_one("#search-results", DataTable)
        results.clear()
        if not query:
            return
        con = db.connect()
        try:
            doc_ids = db.search_fts(con, rid, query, limit=200)
            docs = db.get_documents(con, rid, doc_ids)
        finally:
            con.close()
        order = {d: i for i, d in enumerate(doc_ids)}
        docs.sort(key=lambda d: order.get(d.id, 9999))
        for d in docs:
            text = " ".join(d.searchable_text.split())[:64]
            where = f"{d.platform}/{d.community}" if d.community else d.platform
            results.add_row(where[:16], text, d.created_at.strftime("%Y-%m-%d"))
        self.write_activity(f"search {query!r}: {len(docs)} document(s)")
        self.set_status(f"{len(docs)} match(es) for {query!r}")

    # --- compare --------------------------------------------------------------

    def action_compare(self) -> None:
        """Compare the selected run against the next one in the list."""
        rid = self.selected_run()
        if not rid or len(self.run_ids) < 2:
            self.write_activity("Need two runs to compare.", "yellow")
            return
        i = self.run_ids.index(rid)
        other = self.run_ids[(i + 1) % len(self.run_ids)]

        con = db.connect()
        try:
            stats = {}
            for key in (rid, other):
                docs = list(db.iter_documents(con, key))
                themes = db.load_themes(con, key)
                clusters = db.load_demand_clusters(con, key)
                cs = corpus_stats(docs)
                stats[key] = {
                    "Documents": str(cs.get("documents", 0)),
                    "Posts": str(cs.get("posts", 0)),
                    "Comments": str(cs.get("comments", 0)),
                    "Articles": str(cs.get("articles", 0)),
                    "Authors": str(cs.get("authors", 0)),
                    "Date range": f"{cs.get('earliest','—')} → {cs.get('latest','—')}",
                    "Themes": str(len(themes)),
                    "Top theme": themes[0].label[:34] if themes else "—",
                    "Demand intents": str(len(clusters)),
                    "Biggest gap": f"{clusters[0].gap:+.2f}" if clusters else "—",
                }
        finally:
            con.close()

        table = self.query_one("#compare", DataTable)
        table.clear()
        table.add_row("run", rid[:22], other[:22])
        for metric in stats[rid]:
            table.add_row(metric, stats[rid][metric], stats[other][metric])
        self.query_one("#tabs", TabbedContent).active = "tab-compare"
        self.write_activity(f"comparing {rid} against {other}")

    # --- pipeline actions -----------------------------------------------------

    def action_new_run(self) -> None:
        if self.guard_busy():
            return
        self.query_one("#tabs", TabbedContent).active = "tab-search"
        inp = self.query_one("#search-input", Input)
        inp.placeholder = "Topic to research, then Enter (Esc to cancel)…"
        inp.value = ""
        inp.focus()
        self._awaiting_topic = True

    def action_analyze(self) -> None:
        rid = self.selected_run()
        if rid and not self.guard_busy():
            self.analyze_worker(rid)

    def action_demand(self) -> None:
        rid = self.selected_run()
        if rid and not self.guard_busy():
            self.demand_worker(rid)

    def action_report(self) -> None:
        rid = self.selected_run()
        if rid and not self.guard_busy():
            self.report_worker(rid)

    def action_cycle_theme(self) -> None:
        self.theme = "ansi-light" if self.theme == "ansi-dark" else "ansi-dark"
        self.write_activity(f"theme: {self.theme}")

    # Workers run the synchronous pipeline off the event loop. thread=True is
    # required, not stylistic: pipeline.run_collection calls asyncio.run(), which
    # raises if there is already a loop running on this thread.

    @work(thread=True, exclusive=True)
    def collect_worker(self, topic: str) -> None:
        self.app.call_from_thread(self._begin, f"Collecting “{topic}”…")
        try:
            plan = pipeline.plan_for(topic, 180)
            self.app.call_from_thread(
                self.write_activity,
                f"plan: {len(plan.subreddits)} subreddit(s), "
                f"{len(plan.keywords)} keyword(s), {len(plan.brands)} brand(s)",
            )
            rid, n = pipeline.run_collection(topic, ["reddit", "web"], 180, 600, plan=plan)
            self.app.call_from_thread(
                self.write_activity, f"collected {n} document(s) into {rid}", "green"
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.write_activity, f"collect failed: {e}", "red")
        finally:
            self.app.call_from_thread(self._end, "Collection finished")

    @work(thread=True, exclusive=True)
    def analyze_worker(self, run_id: str) -> None:
        self.app.call_from_thread(self._begin, f"Analyzing {run_id}…")
        try:
            themes = pipeline.run_analysis(run_id)
            self.app.call_from_thread(
                self.write_activity, f"{len(themes)} theme(s) found", "green"
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.write_activity, f"analyze failed: {e}", "red")
        finally:
            self.app.call_from_thread(self._end, "Analysis finished")

    @work(thread=True, exclusive=True)
    def demand_worker(self, run_id: str) -> None:
        self.app.call_from_thread(self._begin, f"Harvesting demand for {run_id}…")

        def progress(done: int, total: int, note: str) -> None:
            if done % 10 == 0 or done == total:
                self.app.call_from_thread(
                    self.set_status, f"Harvesting {done}/{total} — {note}", True
                )

        try:
            clusters = pipeline.run_demand(run_id, depth=1, progress=progress)
            self.app.call_from_thread(
                self.write_activity, f"{len(clusters)} search intent(s)", "green"
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.write_activity, f"demand failed: {e}", "red")
        finally:
            self.app.call_from_thread(self._end, "Demand harvest finished")

    @work(thread=True, exclusive=True)
    def report_worker(self, run_id: str) -> None:
        self.app.call_from_thread(self._begin, f"Rendering {run_id}…")
        try:
            from ideafindr.analyze.signals import language_bank, share_of_voice
            from ideafindr.report.render import render

            con = db.connect()
            try:
                run = db.get_run(con, run_id)
                themes = db.load_themes(con, run_id)
                docs = list(db.iter_documents(con, run_id))
                clusters = db.load_demand_clusters(con, run_id)
            finally:
                con.close()
            if not run or not themes:
                raise ValueError("nothing to render -- analyze the run first")
            md, html = render(
                run=run, themes=themes, stats=corpus_stats(docs),
                language=language_bank(docs),
                sov=share_of_voice(docs, run.plan.brands),
                summary="", demand=clusters,
            )
            self.app.call_from_thread(self.write_activity, f"wrote {md}", "green")
            self.app.call_from_thread(self.write_activity, f"wrote {html}", "green")
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.write_activity, f"report failed: {e}", "red")
        finally:
            self.app.call_from_thread(self._end, "Report finished")

    def _begin(self, message: str) -> None:
        self.busy = True
        self.set_status(message, working=True)
        self.write_activity(message)

    def _end(self, message: str) -> None:
        self.busy = False
        self.set_status(message)
        self.action_refresh()


def run() -> None:
    IdeafindrApp().run()
