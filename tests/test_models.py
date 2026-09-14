from datetime import datetime

from ideafindr.models import Document


def test_naive_datetime_is_made_utc_aware():
    """Naive datetimes would silently corrupt every time-series signal."""
    d = Document(id="x:1", platform="reddit", kind="post", url="https://x",
                 text="hi", created_at=datetime(2026, 1, 1, 12, 0))
    assert d.created_at.tzinfo is not None
    assert d.created_at.utcoffset().total_seconds() == 0


def test_searchable_text_joins_title_and_body():
    d = Document(id="x:1", platform="reddit", kind="post", url="https://x",
                 title="Noisy chiller", text="it hums", created_at=datetime.now())
    assert "Noisy chiller" in d.searchable_text and "it hums" in d.searchable_text


def test_engagement_score_weights_comments_above_score():
    """A post with replies signals more than a post with silent upvotes."""
    quiet = Document(id="a", platform="reddit", kind="post", url="u", text="t",
                     created_at=datetime.now(), engagement={"score": 10})
    talked = Document(id="b", platform="reddit", kind="post", url="u", text="t",
                      created_at=datetime.now(), engagement={"score": 10, "num_comments": 5})
    assert talked.engagement_score() > quiet.engagement_score()
