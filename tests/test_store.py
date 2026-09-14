from ideafindr.models import RunPlan
from ideafindr.store import db
from tests.conftest import make_doc


def _con(tmp_path):
    return db.connect(tmp_path / "t.db")


def test_documents_dedupe_within_a_run(tmp_path, plan):
    con = _con(tmp_path)
    db.create_run(con, "r1", plan, ["reddit"])
    docs = [make_doc(i, f"text {i}") for i in range(5)]
    assert db.save_documents(con, "r1", docs) == 5
    assert db.save_documents(con, "r1", docs) == 0
    assert db.get_run(con, "r1").doc_count == 5


def test_fts_finds_by_keyword_and_ignores_operators(tmp_path, plan):
    con = _con(tmp_path)
    db.create_run(con, "r1", plan, ["reddit"])
    db.save_documents(con, "r1", [
        make_doc(1, "the chiller is extremely loud"),
        make_doc(2, "water quality problems again"),
    ])
    assert "reddit:1" in db.search_fts(con, "r1", "chiller loud")
    # FTS5 syntax in a natural-language query must not raise
    assert db.search_fts(con, "r1", 'chiller AND "unclosed') is not None


def test_children_of_returns_comments_ranked_by_engagement(tmp_path, plan):
    con = _con(tmp_path)
    db.create_run(con, "r1", plan, ["reddit"])
    db.save_documents(con, "r1", [
        make_doc(1, "parent post", kind="post"),
        make_doc(2, "meh reply", kind="comment", parent_id="reddit:1", engagement={"score": 1}),
        make_doc(3, "great reply", kind="comment", parent_id="reddit:1", engagement={"score": 99}),
    ])
    kids = db.children_of(con, "r1", "reddit:1")
    assert [k.id for k in kids] == ["reddit:3", "reddit:2"]


def test_get_documents_handles_more_than_sqlite_variable_limit(tmp_path, plan):
    con = _con(tmp_path)
    db.create_run(con, "r1", plan, ["reddit"])
    docs = [make_doc(i, f"text {i}") for i in range(1200)]
    db.save_documents(con, "r1", docs)
    got = db.get_documents(con, "r1", [d.id for d in docs])
    assert len(got) == 1200
