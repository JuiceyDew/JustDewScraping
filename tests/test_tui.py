"""The TUI, driven headlessly.

Textual apps are testable without a terminal, so the things that would otherwise
only break in front of a user -- a widget id that does not exist, a handler bound
twice, a table whose columns never got added -- fail here instead.
"""

from __future__ import annotations

import pytest

import ideafindr.config as cfg
from ideafindr.models import RunPlan
from ideafindr.store import db
from ideafindr.tui.app import IdeafindrApp
from tests.conftest import make_doc


@pytest.fixture
def seeded(tmp_path, plan, monkeypatch):
    """A database with two runs, so run selection and comparison have something
    to work on."""
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


async def test_app_starts_and_lists_runs(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#runs")
        assert table.row_count == 2, "both runs should be listed"
        assert set(app.run_ids) == {"run-a", "run-b"}


async def test_uses_the_terminals_own_palette(seeded):
    """The whole point of the theming choice: no hardcoded RGB, so the app takes
    on whatever colours the user's terminal already uses."""
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.ansi_color is True
        assert app.theme.startswith("ansi-")


async def test_search_finds_documents_via_fts(seeded):
    """The corpus has had a full-text index all along with no way to reach it."""
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Runs list newest-first, so select run-a explicitly rather than relying
        # on which one happens to be under the cursor.
        app.query_one("#runs").move_cursor(row=app.run_ids.index("run-a"))
        await pilot.pause()
        assert app.selected_run() == "run-a"

        app._do_search("chiller loud")
        await pilot.pause()
        assert app.query_one("#search-results").row_count == 1

        app._do_search("sauna condensation")
        await pilot.pause()
        assert app.query_one("#search-results").row_count == 0, \
            "search must be scoped to the selected run"


async def test_search_does_not_start_a_collection_run(seeded):
    """The input doubles as the topic prompt. Two @on handlers on one selector
    would both fire, and searching would silently start scraping."""
    app = IdeafindrApp()
    started: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app.collect_worker = lambda topic: started.append(topic)  # type: ignore[method-assign]
        await pilot.press("s")
        for ch in "chiller":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause()
        assert started == [], "a search must never launch a run"


async def test_new_run_prompt_routes_the_input_to_collection(seeded):
    app = IdeafindrApp()
    started: list[str] = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app.collect_worker = lambda topic: started.append(topic)  # type: ignore[method-assign]
        await pilot.press("n")
        await pilot.pause()
        app.query_one("#search-input").value = "sauna tents"
        await pilot.press("enter")
        await pilot.pause()
        assert started == ["sauna tents"]
        assert app._awaiting_topic is False, "the prompt must be one-shot"


async def test_compare_shows_both_runs_side_by_side(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_compare()
        await pilot.pause()
        table = app.query_one("#compare")
        assert table.row_count > 3
        assert app.query_one("#tabs").active == "tab-compare"


async def test_actions_do_not_stack_while_a_job_is_running(seeded):
    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.busy = True
        assert app.guard_busy() is True


async def test_pipeline_log_records_reach_the_activity_panel(seeded):
    """Live status comes from the pipeline's own logging, so a collector that
    logs 'discovered r/coldplunge' shows up without any extra plumbing."""
    import logging

    app = IdeafindrApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        logging.getLogger("ideafindr.collectors.reddit_arctic").info(
            "discovered r/coldplunge(13,000)"
        )
        await pilot.pause()
        rendered = app.query_one("#activity").lines
        assert rendered, "activity panel should have content"
