"""The TUI, driven headlessly.

Textual apps are testable without a terminal, so things that would otherwise only
break in front of a user -- a widget id that does not exist, a screen that never
loads its data, a prompt wired to the wrong action -- fail here instead.
"""

from __future__ import annotations

import pytest

import ideafindr.config as cfg
from ideafindr.models import RunPlan
from ideafindr.store import db
from ideafindr.tui.app import (
    ConfirmScreen,
    TopicScreen,
    HomeScreen,
    IdeafindrApp,
    ProjectScreen,
    ResearchScreen,
    delete_project,
)
from tests.conftest import make_doc


@pytest.fixture
def seeded(tmp_path, plan, monkeypatch):
    """Two stored projects, so the home list and project pages have real data."""
    monkeypatch.setattr(cfg.settings, "db_path", tmp_path / "t.db")
    con = db.connect(tmp_path / "t.db")
    db.create_run(con, "run-a", plan, ["reddit"])
    db.save_documents(con, "run-a", [
        make_doc(1, "the chiller is far too loud at night and wakes everyone up",
                 kind="post", title="Chiller noise"),
        make_doc(2, "water goes cloudy after a week even with ozone", kind="post"),
    ])
    db.create_run(con, "run-b", RunPlan(topic="sauna tents", days=90), ["reddit"])
    db.save_documents(con, "run-b", [make_doc(3, "sauna tent condensation problems")])
    con.close()
    return tmp_path


# --- home ---------------------------------------------------------------------


async def test_home_is_one_list_with_new_research_first(seeded):
    """Everything is a row, so the arrows always move and enter always activates.
    The previous layout had an Input holding focus beside a list, which meant
    plain keys went into the text box and the list needed a special binding to
    reach at all."""
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        home = app.screen
        assert isinstance(home, HomeScreen)
        rows = home.query_one("#rows")
        assert rows.has_focus, "the list, not a text box, holds focus"
        assert len(rows.children) == 3, "one New row plus two projects"
        assert rows.index == 0
        assert not home.query("Input"), "Home must not own an input any more"


async def test_home_stays_uncluttered(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        home = app.screen
        assert not home.query("DataTable"), "no tables on the home screen"
        assert not home.query("RichLog"), "no log panel on the home screen"
        assert not home.query("TabbedContent"), "no tabs on the home screen"


async def test_enter_on_the_first_row_asks_for_a_topic(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, TopicScreen)
        assert app.screen.query_one("#topic").has_focus


async def test_topic_prompt_starts_research_and_escape_cancels(seeded):
    app = IdeafindrApp()
    started: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        ResearchScreen.deep_research = lambda self: started.append(self.topic)  # type: ignore[assignment]

        await pilot.press("n")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen), "esc returns home"
        assert started == [], "cancelling must not start anything"

        await pilot.press("n")
        await pilot.pause()
        app.screen.query_one("#topic").value = "sauna tents"
        await pilot.press("enter")
        await pilot.pause()
        assert started == ["sauna tents"]
        assert isinstance(app.screen, ResearchScreen)


async def test_arrow_keys_move_through_the_list(seeded):
    """No focus juggling: down from the New row lands on the first project."""
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        home = app.screen
        assert home._selected_run() is None, "row 0 is New research, not a project"
        await pilot.press("down")
        await pilot.pause()
        assert home._selected_run() == home.run_ids[0]


async def test_every_row_is_visible(seeded):
    app = IdeafindrApp()
    async with app.run_test(size=(110, 34)) as pilot:
        await pilot.pause()
        await pilot.pause()
        items = app.screen.query_one("#rows").children
        assert all(i.size.height == 1 for i in items), \
            f"rows should be one line each, got {[i.size.height for i in items]}"


async def test_selecting_a_project_opens_its_own_page(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        home = app.screen
        rows = home.query_one("#rows")
        rows.index = 1 + home.run_ids.index("run-a")
        rows.action_select_cursor()
        await pilot.pause()
        assert isinstance(app.screen, ProjectScreen)
        assert app.screen.run_id == "run-a"


# --- research -----------------------------------------------------------------


async def test_research_screen_shows_every_stage(seeded, monkeypatch):
    monkeypatch.setattr(ResearchScreen, "deep_research", lambda self: None)
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = ResearchScreen("cold plunge tubs")
        await app.push_screen(screen)
        await pilot.pause()
        for key in ("plan", "collect", "analyze", "demand", "report"):
            assert screen.query_one(f"#stage-{key}")


async def test_stage_marks_progress_and_failure(seeded, monkeypatch):
    monkeypatch.setattr(ResearchScreen, "deep_research", lambda self: None)
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = ResearchScreen("cold plunge tubs")
        await app.push_screen(screen)
        await pilot.pause()

        screen.stage("plan", "done", "12 subreddits")
        assert "stage-done" in screen.query_one("#stage-plan").classes
        screen.stage("collect", "failed", "nothing collected")
        assert "stage-failed" in screen.query_one("#stage-collect").classes


async def test_pipeline_logs_reach_the_research_screen_only(seeded, monkeypatch):
    """Live detail belongs on the screen doing the work; a log panel on every
    page is the clutter this layout removes."""
    import logging

    monkeypatch.setattr(ResearchScreen, "deep_research", lambda self: None)
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Home is showing: pipeline chatter must go nowhere and must not raise.
        assert app.log_sink is None
        logging.getLogger("ideafindr.collectors.reddit_arctic").info("discovered r/x")
        await pilot.pause()

        screen = ResearchScreen("cold plunge tubs")
        await app.push_screen(screen)
        await pilot.pause()
        assert app.log_sink is screen
        logging.getLogger("ideafindr.collectors.reddit_arctic").info(
            "discovered r/coldplunge(13,000)"
        )
        await pilot.pause()
        assert screen.query_one("#detail").lines


# --- project ------------------------------------------------------------------


async def test_project_page_loads_its_own_data(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = ProjectScreen("run-a")
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.topic == "cold plunge tubs"
        assert "2 documents" in str(screen.query_one("#proj-sub").content)
        assert "press a to analyze" in str(screen.query_one("#proj-status").content)


async def test_project_heading_stays_on_screen(seeded):
    """TabbedContent claimed the full screen height rather than the space left
    after the heading rows, scrolling the project name off the top."""
    app = IdeafindrApp()
    async with app.run_test(size=(104, 28)) as pilot:
        await pilot.pause()
        screen = ProjectScreen("run-a")
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one("#proj-head").region.y >= 0, "project name scrolled away"
        assert screen.query_one("#proj-sub").region.y >= 0


async def test_number_keys_switch_project_tabs(seeded):
    """TabbedContent ships with no keybindings, so the tabs were mouse-only."""
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = ProjectScreen("run-a")
        await app.push_screen(screen)
        await pilot.pause()
        tabs = screen.query_one("#tabs")

        await pilot.press("2")
        await pilot.pause()
        assert tabs.active == "tab-demand"
        await pilot.press("3")
        await pilot.pause()
        assert tabs.active == "tab-search"
        assert screen.query_one("#search").has_focus, "search tab focuses its input"


async def test_tab_key_cycles_through_every_pane(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = ProjectScreen("run-a")
        await app.push_screen(screen)
        await pilot.pause()
        seen = []
        for _ in range(4):
            screen.action_next_tab()
            await pilot.pause()
            seen.append(screen.query_one("#tabs").active)
        assert set(seen) == set(ProjectScreen.TABS), "every pane reachable"
        assert seen[0] == seen[3], "and it wraps"


async def test_project_search_is_scoped_to_that_project(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = ProjectScreen("run-a")
        await app.push_screen(screen)
        await pilot.pause()

        screen.do_search("chiller loud")
        await pilot.pause()
        assert screen.query_one("#results").row_count == 1

        screen.do_search("sauna condensation")
        await pilot.pause()
        assert screen.query_one("#results").row_count == 0, \
            "another project's documents must not leak in"


async def test_escape_returns_home_from_a_project(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.push_screen(ProjectScreen("run-a"))
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)


# --- app ----------------------------------------------------------------------


async def test_uses_the_terminals_own_palette(seeded):
    """No hardcoded RGB, so the app takes on the terminal's existing scheme."""
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.ansi_color is True
        assert app.theme == "ansi-dark"
        app.action_cycle_theme()
        assert app.theme == "ansi-light"


# --- deleting -----------------------------------------------------------------


async def test_delete_asks_before_removing_anything(seeded):
    """Deletion is irreversible, so it must never happen on a single keystroke."""
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        home = app.screen
        home.query_one("#rows").index = 1 + home.run_ids.index("run-a")
        await pilot.press("x")
        await pilot.pause()

        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("escape")
        await pilot.pause()

        con = db.connect()
        try:
            assert db.get_run(con, "run-a") is not None, "cancelling must keep the run"
        finally:
            con.close()


async def test_confirming_removes_the_project_and_leaves_the_others(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        home = app.screen
        home.query_one("#rows").index = 1 + home.run_ids.index("run-a")
        await pilot.press("x")
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()

        con = db.connect()
        try:
            assert db.get_run(con, "run-a") is None
            assert db.get_run(con, "run-b") is not None, "only the chosen run goes"
            assert db.search_fts(con, "run-a", "chiller") == []
            assert db.search_fts(con, "run-b", "sauna") != [], "survivor stays searchable"
        finally:
            con.close()
        assert set(app.screen.run_ids) == {"run-b"}, "the list should refresh"


async def test_delete_on_the_new_row_does_nothing(seeded):
    """Row 0 is not a project. Pressing delete there must be inert, not delete
    whatever happens to be first."""
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query_one("#rows").index == 0
        await pilot.press("x")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen), "no confirmation should appear"


def test_delete_project_also_removes_rendered_reports(seeded, monkeypatch):
    """Reports live outside the database; leaving them orphans a report citing
    documents that no longer exist."""
    import ideafindr.config as cfg

    reports = seeded / "reports"
    monkeypatch.setattr(cfg.settings, "reports_dir", reports)
    (reports / "run-a").mkdir(parents=True)
    (reports / "run-a" / "report.md").write_text("# stale")
    (reports / "run-b").mkdir(parents=True)

    delete_project("run-a")
    assert not (reports / "run-a").exists()
    assert (reports / "run-b").exists(), "another project's reports must survive"
