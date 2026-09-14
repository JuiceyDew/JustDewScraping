from datetime import datetime, timedelta, timezone

from ideafindr.analyze.cluster import auto_k, cluster_documents, keywords_for
from ideafindr.analyze.quotes import attach_quotes
from ideafindr.analyze.signals import (
    compute_theme_signals,
    language_bank,
    momentum,
    share_of_voice,
    strip_noise,
    week_key,
)
from ideafindr.embed import TfidfEmbedder, _l2
from tests.conftest import make_doc

import numpy as np


def test_zero_vectors_get_distinct_directions():
    """All-stopword documents produce zero rows; sklearn's cosine metric raises on
    them, and collapsing them to one shared vector would fake a large theme."""
    m = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2(m)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert not np.allclose(out[1], out[2])


def test_content_free_documents_are_excluded_from_clustering(docs):
    docs = docs + [make_doc(999, "lol", kind="comment")]
    _, M, kept = cluster_documents(docs, embedder=TfidfEmbedder())
    assert "reddit:999" not in {d.id for d in kept}
    assert M.shape[0] == len(kept), "M must stay aligned to the returned documents"


def test_clustering_separates_distinct_topics(docs):
    themes, M, kept = cluster_documents(docs, embedder=TfidfEmbedder(), n_clusters=3)
    assert 2 <= len(themes) <= 3
    assert sum(t.volume or len(t.doc_ids) for t in themes) <= len(kept)
    # every returned doc id must exist in the kept set
    ids = {d.id for d in kept}
    for t in themes:
        assert set(t.doc_ids) <= ids


def test_auto_k_scales_sublinearly():
    assert auto_k(20) < auto_k(500) < auto_k(5000)
    assert auto_k(10) >= 4 and auto_k(100000) <= 28


def test_momentum_is_relative_to_corpus_not_raw_recency():
    now = datetime.now(timezone.utc)
    def d(i, days):
        x = make_doc(i, f"text {i}")
        x.created_at = now - timedelta(days=days)
        return x
    corpus = [d(i, i * 3) for i in range(40)]          # spread over 120 days
    rising = [x for x in corpus if (now - x.created_at).days < 25]
    fading = [x for x in corpus if (now - x.created_at).days > 90]
    assert momentum(rising, corpus) > 1.2
    assert momentum(fading, corpus) < 0.5


def test_web_documents_do_not_inflate_momentum():
    """Web articles carry their COLLECTION date, not their publication date, so
    they all land in the current week. Counted in the time series they inflate
    the recent share of whatever theme they join -- which is what made the
    highest-momentum theme in a real report 94% web articles."""
    from ideafindr.models import Theme

    now = datetime.now(timezone.utc)
    def d(i, days, platform="reddit"):
        x = make_doc(i, f"some text about chillers {i}", platform=platform)
        x.created_at = now - timedelta(days=days)
        return x

    corpus = [d(i, i * 3) for i in range(40)]                     # spread over 120 days
    theme_ids = [x.id for x in corpus if (now - x.created_at).days > 90]
    theme = Theme(id=0, label="old theme", doc_ids=list(theme_ids))
    before = compute_theme_signals([theme], corpus)[0].momentum

    # Same theme, now padded with web articles stamped `now`.
    web = [d(1000 + i, 0, platform="web") for i in range(15)]
    theme2 = Theme(id=0, label="old theme", doc_ids=theme_ids + [w.id for w in web])
    after = compute_theme_signals([theme2], corpus + web)[0]

    assert after.momentum == before, "undated web docs must not move momentum"
    assert after.volume == len(theme_ids) + len(web), "but they still count as documents"
    assert all(wk != week_key(now) for wk, _ in after.trend), "and never enter the trend"


def test_share_of_voice_trend_excludes_undated_web_documents():
    docs = [make_doc(1, "the Plunge brand tub is good", days_ago=40),
            make_doc(2, "Plunge shipping was slow", platform="web", kind="article")]
    sov = share_of_voice(docs, ["Plunge"])
    assert sov["Plunge"]["mentions"] == 2, "web mentions still count towards share"
    assert len(sov["Plunge"]["trend"]) == 1, "but only the dated one shapes the trend"


def test_share_of_voice_uses_word_boundaries():
    docs = [make_doc(1, "the Plunge brand tub is good"),
            make_doc(2, "I plunged into the lake"),          # must NOT match "Plunge"
            make_doc(3, "Ice Barrel shipping was slow")]
    sov = share_of_voice(docs, ["Plunge", "Ice Barrel", "Renu"])
    assert sov["Plunge"]["mentions"] == 1
    assert sov["Ice Barrel"]["mentions"] == 1
    assert "Renu" not in sov, "brands with zero mentions must be omitted"


def test_strip_noise_removes_urls_and_quoted_text():
    out = strip_noise("see [this](https://x.com/a) \n&gt; they said that\nmy own words")
    assert "http" not in out and "they said that" not in out
    assert "my own words" in out and "this" in out


def test_language_bank_prefers_phrases_and_excludes_urls():
    docs = [make_doc(i, "cold plunge water quality https://spam.example.com") for i in range(6)]
    bank = dict(language_bank(docs))
    assert any(" " in w for w in bank), "bigrams should surface"
    assert not any("http" in w or "www" in w for w in bank)


def test_quotes_carry_permalinks_and_skip_fragments(docs):
    themes, M, kept = cluster_documents(docs, embedder=TfidfEmbedder(), n_clusters=3)
    themes = attach_quotes(themes, kept, M, n=3)
    themes = compute_theme_signals(themes, kept)
    quoted = [q for t in themes for q in t.quotes]
    assert quoted, "expected at least one quote"
    for q in quoted:
        assert q.url.startswith("https://")
        assert len(q.text) >= 40


def test_keywords_ignore_stopwords():
    kw = keywords_for(["the chiller is very loud and it is really annoying"] * 3)
    assert "chiller" in kw
    assert not ({"the", "is", "and", "it"} & set(kw))


def test_incoherent_clusters_rank_last_regardless_of_size():
    """Clustering returns k groups whether or not k real themes exist. The
    leftovers bucket must never outrank a real theme just by being big."""
    from ideafindr.models import Theme

    themes = [
        Theme(id=0, label="leftovers", coherent=False, doc_ids=["a"] * 80),
        Theme(id=1, label="real theme", coherent=True, doc_ids=["b"] * 10),
    ]
    ranked = compute_theme_signals(themes, [])
    assert [t.label for t in ranked] == ["real theme", "leftovers"]
