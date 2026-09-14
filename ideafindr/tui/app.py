"""The Ideafindr TUI.

Three screens, each doing one thing, rather than one dashboard doing all of them:

    Home      a prompt and a list of past projects. Nothing else.
    Research  one topic being worked, stage by stage, with live detail.
    Project   one past project, opened as its own page.

The first version put runs, themes, demand, search, compare and a log panel on a
single screen. Everything was visible and nothing was legible: the thing you
actually came to do -- research a topic -- was a keystroke buried among six
others. Home is now almost empty on purpose, and the deep research flow is the
default action rather than one option among many.

Theming: `ansi_color = True` with an ANSI theme, so every colour resolves to one
of the terminal's own 16 palette entries and the app matches whatever scheme the
terminal already uses.

Threading: the pipeline is synchronous and calls asyncio.run() internally
(pipeline.run_collection), which raises inside Textual's running event loop, so
every pipeline call goes through a threaded worker. Live detail comes from
capturing the pipeline's own log records rather than threading a progress
callback through every collector.
"""

from __future__ import annotations

import logging
from datetime import datetime

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from ideafindr import pipeline
from ideafindr.analyze.signals import corpus_stats, language_bank, share_of_voice
from ideafindr.report.render import render
from ideafindr.store import db

log = logging.getLogger(__name__)

DAYS = 180


class TuiLogHandler(logging.Handler):
    """Route the pipeline's own log records to whichever screen is watching.

    The collectors already log what is worth seeing -- which subreddits resolved,
    how many posts came back, what was dropped as off-topic, when Arctic Shift
    asks us to slow down. Only the research screen displays them; elsewhere they
    are dropped, because a log panel on every page is exactly the clutter this
    layout removes.
    """

    def __init__(self, app: "IdeafindrApp") -> None:
        super().__init__(level=logging.INFO)
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001
            return
        style = "yellow" if record.levelno >= logging.WARNING else ""
        try:
            self._app.call_from_thread(self._app.relay_log, msg, style)
        except RuntimeError:
            # call_from_thread refuses to run on the app's own thread. Anything
            # logged from the UI thread rather than a worker lands here, and
            # would otherwise be dropped silently.
            try:
                self._app.relay_log(msg, style)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001 - app shutting down
            pass


def _project_rows() -> list[tuple[str, str, str]]:
    """(run_id, topic, one-line summary) for every stored run, newest first."""
    con = db.connect()
    try:
        out = []
        for r in db.list_runs(con):
            themes = len(db.load_themes(con, r.id))
            demand = len(db.load_demand_clusters(con, r.id))
            bits = [f"{r.doc_count} docs"]
            if themes:
                bits.append(f"{themes} themes")
            if demand:
                bits.append(f"{demand} intents")
            bits.append(r.created_at.strftime("%b %d"))
            out.append((r.id, r.topic, "  ·  ".join(bits)))
        return out
    finally:
        con.close()


# --- home ---------------------------------------------------------------------


class HomeScreen(Screen):
    """A prompt and your past projects. Deliberately almost empty."""

    BINDINGS = [
        ("escape", "clear", "Clear"),
        ("f5", "refresh", "Refresh"),
        ("ctrl+t", "cycle_theme", "Theme"),
        ("q", "quit", "Quit"),
    ]

    CSS = """
    HomeScreen { align: center top; }
    #hero { width: 72; margin-top: 2; }
    #wordmark { text-style: bold; width: 100%; content-align: center middle; }
    #tagline { color: $foreground 55%; width: 100%; content-align: center middle;
               margin-bottom: 2; }
    #topic { width: 100%; }
    #hint { color: $foreground 45%; width: 100%; content-align: center middle;
            margin-bottom: 2; }
    #projects-label { text-style: bold; margin-bottom: 1; }
    #projects { height: auto; max-height: 14; background: transparent; }
    /* Without explicit heights the inner Horizontal takes 1fr and each row
       becomes as tall as the list, hiding every project but the first. */
    #projects ListItem { height: 1; padding: 0 1; background: transparent; }
    #projects ListItem Static { height: 1; }
    #empty { color: $foreground 45%; }
    """

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="hero"):
                yield Static("IDEAFINDR", id="wordmark")
                yield Static("what people say · what people search", id="tagline")
                yield Input(placeholder="Research a topic…", id="topic")
                yield Static("enter to research  ·  ↑↓ then enter to open a project",
                             id="hint")
                yield Label("Projects", id="projects-label")
                yield ListView(id="projects")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#topic", Input).focus()
        self.action_refresh()

    def action_refresh(self) -> None:
        listing = self.query_one("#projects", ListView)
        listing.clear()
        self.run_ids: list[str] = []
        rows = _project_rows()
        if not rows:
            listing.append(ListItem(Static("No projects yet — type a topic above.",
                                           id="empty")))
            return
        for rid, topic, meta in rows:
            self.run_ids.append(rid)
            # One Static per row, not a Horizontal of two: side-by-side widgets
            # each defaulted to the full width, so the first pushed the second
            # off the row entirely. Markup gives the same two-tone look with one
            # widget and no layout arithmetic.
            listing.append(
                ListItem(Static(f"[b]{topic[:28]:<28}[/b] [dim]{meta}[/dim]"))
            )

    def action_clear(self) -> None:
        self.query_one("#topic", Input).value = ""

    def action_cycle_theme(self) -> None:
        self.app.action_cycle_theme()

    @on(Input.Submitted, "#topic")
    def start(self, event: Input.Submitted) -> None:
        topic = event.value.strip()
        if not topic:
            return
        self.query_one("#topic", Input).value = ""
        # A slash command is someone reaching for a way out, not a topic. Without
        # this, typing "/exit" scraped the web for the phrase "/exit" and left a
        # zero-document project behind.
        if topic.startswith("/"):
            if topic.lower() in ("/exit", "/quit", "/q"):
                self.app.exit()
            else:
                self.notify(f"Unknown command {topic}. Type a topic, or q to quit.",
                            severity="warning")
            return
        self.app.push_screen(ResearchScreen(topic))

    @on(ListView.Selected, "#projects")
    def open_project(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(getattr(self, "run_ids", [])):
            self.app.push_screen(ProjectScreen(self.run_ids[idx]))


# --- research -----------------------------------------------------------------

STAGES = [
    ("plan", "Plan the sweep"),
    ("collect", "Collect the corpus"),
    ("analyze", "Cluster into themes"),
    ("demand", "Harvest search demand"),
    ("report", "Write the report"),
]


class ResearchScreen(Screen):
    """One topic, worked end to end, with the stages visible."""

    BINDINGS = [
        ("escape", "leave", "Back"),
        ("ctrl+t", "cycle_theme", "Theme"),
    ]

    CSS = """
    ResearchScreen { align: center top; }
    #work { width: 82; margin-top: 2; }
    #heading { text-style: bold; margin-bottom: 1; }
    .stage { height: 1; }
    .stage-done { color: green; }
    .stage-active { text-style: bold; }
    .stage-todo { color: $foreground 40%; }
    .stage-failed { color: red; }
    #detail { height: 14; border: round $foreground 25%; margin-top: 1; }
    """

    def __init__(self, topic: str) -> None:
        super().__init__()
        self.topic = topic
        self.run_id: str | None = None
        self.finished = False

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="work"):
                yield Static(f"Researching “{self.topic}”", id="heading")
                for key, title in STAGES:
                    yield Static(f"  ·  {title}", id=f"stage-{key}", classes="stage stage-todo")
                yield RichLog(id="detail", markup=True, wrap=True, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        self.app.log_sink = self
        self.deep_research()

    def on_unmount(self) -> None:
        if getattr(self.app, "log_sink", None) is self:
            self.app.log_sink = None

    def write_detail(self, message: str, style: str = "") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        body = f"[{style}]{message}[/{style}]" if style else message
        self.query_one("#detail", RichLog).write(f"[dim]{stamp}[/dim] {body}")

    def stage(self, key: str, state: str, note: str = "") -> None:
        marks = {"active": "⠿", "done": "✓", "failed": "✗", "todo": "·"}
        title = dict(STAGES)[key]
        widget = self.query_one(f"#stage-{key}", Static)
        widget.set_classes(f"stage stage-{state}")
        suffix = f"   {note}" if note else ""
        widget.update(f"  {marks[state]}  {title}{suffix}")

    def action_leave(self) -> None:
        """Leave the screen. The worker keeps running; its results are stored."""
        if self.finished and self.run_id:
            self.app.switch_screen(ProjectScreen(self.run_id))
        else:
            self.app.pop_screen()

    def action_cycle_theme(self) -> None:
        self.app.action_cycle_theme()

    @work(thread=True, exclusive=True)
    def deep_research(self) -> None:
        """Plan, collect, analyse, harvest demand and render, in one pass.

        Threaded because pipeline.run_collection calls asyncio.run() internally,
        which raises inside Textual's event loop.
        """
        app = self.app

        def ui(fn, *a) -> None:
            app.call_from_thread(fn, *a)

        try:
            ui(self.stage, "plan", "active")
            plan = pipeline.plan_for(self.topic, DAYS)
            ui(self.stage, "plan", "done",
               f"{len(plan.subreddits)} subreddits, {len(plan.keywords)} keywords")

            ui(self.stage, "collect", "active")
            rid, n = pipeline.run_collection(
                self.topic, ["reddit", "web"], DAYS, 600, plan=plan
            )
            self.run_id = rid
            if not n:
                ui(self.stage, "collect", "failed", "nothing collected")
                ui(self.write_detail,
                   "Nothing collected. Try a broader topic or a longer window.", "yellow")
                return
            ui(self.stage, "collect", "done", f"{n} documents")

            ui(self.stage, "analyze", "active")
            themes = pipeline.run_analysis(rid)
            ui(self.stage, "analyze", "done", f"{len(themes)} themes")

            ui(self.stage, "demand", "active")
            try:
                clusters = pipeline.run_demand(rid, depth=1)
                ui(self.stage, "demand", "done", f"{len(clusters)} search intents")
            except Exception as e:  # noqa: BLE001 - demand is not worth losing a corpus over
                clusters = []
                ui(self.stage, "demand", "failed", str(e)[:40])

            ui(self.stage, "report", "active")
            con = db.connect()
            try:
                run = db.get_run(con, rid)
                docs = list(db.iter_documents(con, rid))
            finally:
                con.close()
            md, _ = render(
                run=run, themes=themes, stats=corpus_stats(docs),
                language=language_bank(docs),
                sov=share_of_voice(docs, run.plan.brands),
                summary="", demand=clusters,
            )
            ui(self.stage, "report", "done", str(md.parent))
            self.finished = True
            ui(self.write_detail, "Done. Press esc to open the project.", "green")
        except Exception as e:  # noqa: BLE001
            ui(self.write_detail, f"Failed: {e}", "red")


# --- project ------------------------------------------------------------------


class ProjectScreen(Screen):
    """One past project, on its own page."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("s", "focus_search", "Search"),
        ("a", "analyze", "Re-analyze"),
        ("d", "demand", "Demand"),
        ("r", "report", "Report"),
        ("ctrl+t", "cycle_theme", "Theme"),
    ]

    CSS = """
    #proj-head { text-style: bold; padding: 0 1; height: 1; }
    #proj-sub { color: $foreground 55%; padding: 0 1; height: 1; margin-bottom: 1; }
    #proj-status { color: $foreground 55%; padding: 0 1; height: 1; }
    /* Without this the tabs claim the whole screen height rather than what is
       left after the heading rows, and the heading scrolls off the top. */
    #tabs { height: 1fr; }
    DataTable { height: 1fr; }
    """

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.topic = run_id

    def compose(self) -> ComposeResult:
        yield Static("", id="proj-head")
        yield Static("", id="proj-sub")
        with TabbedContent(id="tabs"):
            with TabPane("Themes", id="tab-themes"):
                yield DataTable(id="themes", cursor_type="row")
            with TabPane("Demand", id="tab-demand"):
                yield DataTable(id="demand", cursor_type="row")
            with TabPane("Search", id="tab-search"):
                yield Input(placeholder="Search this corpus…", id="search")
                yield DataTable(id="results", cursor_type="row")
        yield Static("", id="proj-status")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#themes", DataTable)
        t.add_column("#", width=3)
        t.add_column("Theme", width=52)
        t.add_column("Docs", width=6)
        t.add_column("Mom.", width=6)
        t.add_column("Stance", width=11)

        d = self.query_one("#demand", DataTable)
        d.add_column("#", width=3)
        d.add_column("Search intent", width=46)
        d.add_column("Queries", width=8)
        d.add_column("Corpus", width=7)
        d.add_column("Gap", width=7)

        r = self.query_one("#results", DataTable)
        r.add_column("Where", width=18)
        r.add_column("Text", width=72)
        r.add_column("Date", width=11)

        self.app.log_sink = None
        self.load()

    def load(self) -> None:
        con = db.connect()
        try:
            run = db.get_run(con, self.run_id)
            themes = db.load_themes(con, self.run_id)
            clusters = db.load_demand_clusters(con, self.run_id)
            docs = list(db.iter_documents(con, self.run_id))
        finally:
            con.close()
        if not run:
            self.query_one("#proj-head", Static).update("Unknown project")
            return

        self.topic = run.topic
        stats = corpus_stats(docs)
        self.query_one("#proj-head", Static).update(run.topic)
        self.query_one("#proj-sub", Static).update(
            f"{stats.get('documents', 0)} documents  ·  "
            f"{stats.get('earliest', '—')} → {stats.get('latest', '—')}  ·  {self.run_id}"
        )

        tt = self.query_one("#themes", DataTable)
        tt.clear()
        for i, t in enumerate(themes, 1):
            tt.add_row(str(i), t.label[:52], str(t.volume),
                       f"{t.momentum:.2f}", t.stance.replace("_", " "))

        dt = self.query_one("#demand", DataTable)
        dt.clear()
        for i, c in enumerate(clusters, 1):
            dt.add_row(str(i), c.label[:46], str(len(c.unique_texts())),
                       str(c.coverage), f"{c.gap:+.2f}")

        hints = []
        if not themes:
            hints.append("press a to analyze")
        if not clusters:
            hints.append("press d to harvest demand")
        self.set_status("  ·  ".join(hints))

    def set_status(self, text: str) -> None:
        self.query_one("#proj-status", Static).update(text)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_cycle_theme(self) -> None:
        self.app.action_cycle_theme()

    def action_focus_search(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-search"
        self.query_one("#search", Input).focus()

    @on(Input.Submitted, "#search")
    def search(self, event: Input.Submitted) -> None:
        self.do_search(event.value)

    def do_search(self, value: str) -> None:
        results = self.query_one("#results", DataTable)
        results.clear()
        query = value.strip()
        if not query:
            return
        con = db.connect()
        try:
            ids = db.search_fts(con, self.run_id, query, limit=200)
            docs = db.get_documents(con, self.run_id, ids)
        finally:
            con.close()
        order = {d: i for i, d in enumerate(ids)}
        docs.sort(key=lambda d: order.get(d.id, 9999))
        for d in docs:
            where = f"{d.platform}/{d.community}" if d.community else d.platform
            results.add_row(where[:18],
                            " ".join(d.searchable_text.split())[:72],
                            d.created_at.strftime("%Y-%m-%d"))
        self.set_status(f"{len(docs)} match(es) for “{query}”")

    def action_analyze(self) -> None:
        self.set_status("Re-analyzing…")
        self.job("analyze")

    def action_demand(self) -> None:
        self.set_status("Harvesting search demand… (about a minute)")
        self.job("demand")

    def action_report(self) -> None:
        self.set_status("Rendering…")
        self.job("report")

    @work(thread=True, exclusive=True)
    def job(self, kind: str) -> None:
        app = self.app
        try:
            if kind == "analyze":
                n = len(pipeline.run_analysis(self.run_id))
                msg = f"{n} themes"
            elif kind == "demand":
                n = len(pipeline.run_demand(self.run_id, depth=1))
                msg = f"{n} search intents"
            else:
                con = db.connect()
                try:
                    run = db.get_run(con, self.run_id)
                    themes = db.load_themes(con, self.run_id)
                    docs = list(db.iter_documents(con, self.run_id))
                    clusters = db.load_demand_clusters(con, self.run_id)
                finally:
                    con.close()
                if not themes:
                    raise ValueError("analyze the project first")
                md, _ = render(
                    run=run, themes=themes, stats=corpus_stats(docs),
                    language=language_bank(docs),
                    sov=share_of_voice(docs, run.plan.brands),
                    summary="", demand=clusters,
                )
                msg = f"written to {md.parent}"
            app.call_from_thread(self.set_status, msg)
            app.call_from_thread(self.load)
        except Exception as e:  # noqa: BLE001
            app.call_from_thread(self.set_status, f"Failed: {e}")


# --- app ----------------------------------------------------------------------


class IdeafindrApp(App):
    """Research a topic, then read the project it produced."""

    ansi_color = True
    TITLE = "Ideafindr"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.log_sink: ResearchScreen | None = None

    def on_mount(self) -> None:
        self.theme = "ansi-dark"
        handler = TuiLogHandler(self)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger("ideafindr")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.push_screen(HomeScreen())

    def relay_log(self, message: str, style: str = "") -> None:
        """Only the research screen shows pipeline chatter."""
        sink = self.log_sink
        if sink is not None and sink.is_mounted:
            sink.write_detail(message, style)

    def action_cycle_theme(self) -> None:
        self.theme = "ansi-light" if self.theme == "ansi-dark" else "ansi-dark"


def run() -> None:
    IdeafindrApp().run()
