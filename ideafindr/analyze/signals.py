"""The quantitative layer: volume, trend, momentum, share of voice.

This is the part none of the eight upstream repos do. They all write prose;
"what are people talking about" is a counting question first.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from ideafindr.models import Document, Theme


# Platforms whose created_at is collection time rather than publication time.
# WebCollector stamps `now` on every article it extracts, because trafilatura's
# date extraction is wrong often enough that trusting it would corrupt the trend
# charts (see collectors/web.py). Those documents are real and count towards
# volume, themes and the language bank -- they just cannot carry a date, so they
# must be kept out of every time-series calculation. Left in, they all land in
# the current week and inflate the recent share of whatever theme they join.
UNDATED_PLATFORMS = {"web"}


def timed(docs: list[Document]) -> list[Document]:
    """Only the documents whose created_at is a real publication date."""
    return [d for d in docs if d.platform not in UNDATED_PLATFORMS]


def week_key(dt: datetime) -> str:
    d = dt.astimezone(timezone.utc)
    return (d - timedelta(days=d.weekday())).date().isoformat()


def theme_trend(docs: list[Document]) -> list[tuple[str, int]]:
    c: Counter[str] = Counter(week_key(d.created_at) for d in docs)
    return sorted(c.items())


def momentum(docs: list[Document], all_docs: list[Document], recent_days: int = 30) -> float:
    """Is this theme rising or fading, controlling for overall corpus volume?

    Returns the theme's share of recent chatter divided by its share overall.
    >1 means rising faster than the topic as a whole; ~1 is flat. This is the
    number a marketing client actually pays for -- raw recent volume just tracks
    how much was collected, not what's growing.
    """
    if not docs or not all_docs:
        return 0.0
    cutoff = datetime.now(timezone.utc) - timedelta(days=recent_days)
    t_recent = sum(1 for d in docs if d.created_at >= cutoff)
    a_recent = sum(1 for d in all_docs if d.created_at >= cutoff)
    if not a_recent or not t_recent:
        return 0.0
    recent_share = t_recent / a_recent
    overall_share = len(docs) / len(all_docs)
    return round(recent_share / overall_share, 2) if overall_share else 0.0


def compute_theme_signals(themes: list[Theme], docs: list[Document]) -> list[Theme]:
    by_id = {d.id: d for d in docs}
    # Both sides of the momentum ratio are filtered: dropping undated documents
    # from the theme but not the corpus baseline skews the ratio the other way.
    dated_corpus = timed(docs)
    for t in themes:
        tdocs = [by_id[i] for i in t.doc_ids if i in by_id]
        t.volume = len(tdocs)
        t.engagement_weighted = round(sum(d.engagement_score() for d in tdocs), 1)
        t.trend = theme_trend(timed(tdocs))
        t.momentum = momentum(timed(tdocs), dated_corpus)
    # Rank by what a marketer should look at first: size, weighted by growth.
    # Incoherent clusters sort last regardless of size -- a big pile of unrelated
    # posts is the least actionable thing in the report, not the fourth most.
    themes.sort(key=lambda t: (t.coherent, t.volume * max(t.momentum, 0.35)), reverse=True)
    return themes


def share_of_voice(
    docs: list[Document], brands: list[str]
) -> dict[str, dict[str, object]]:
    """Brand mention counts over time.

    Word-boundary matching, so 'Ice' doesn't match 'service' and 'Plunge' doesn't
    match 'plunged'. Multi-word brands are matched as phrases.
    """
    if not brands:
        return {}
    pats = {
        b: re.compile(r"\b" + r"\s+".join(re.escape(w) for w in b.split()) + r"\b", re.I)
        for b in brands
    }
    totals: Counter[str] = Counter()
    weekly: dict[str, Counter[str]] = defaultdict(Counter)
    engagement: dict[str, float] = defaultdict(float)

    for d in docs:
        text = d.searchable_text
        dated = d.platform not in UNDATED_PLATFORMS
        for b, p in pats.items():
            if p.search(text):
                totals[b] += 1
                # Mentions count wherever they appear, but only dated documents
                # may shape the trend line -- same reasoning as theme momentum.
                if dated:
                    weekly[b][week_key(d.created_at)] += 1
                engagement[b] += d.engagement_score()

    total_mentions = sum(totals.values()) or 1
    return {
        b: {
            "mentions": totals[b],
            "share": round(100 * totals[b] / total_mentions, 1),
            "engagement": round(engagement[b], 1),
            "trend": sorted(weekly[b].items()),
        }
        for b in sorted(brands, key=lambda x: -totals[x])
        if totals[b]
    }


_URL = re.compile(r"https?://\S+|www\.\S+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_QUOTE_LINE = re.compile(r"^&gt;.*$|^>.*$", re.M)


def strip_noise(text: str) -> str:
    """Remove URLs, markdown link targets and quoted parent text.

    Without this the language bank fills up with 'https', 'www' and the domain
    names of whatever gets linked, and quoted text double-counts other people's
    words as if the commenter had said them.
    """
    text = _QUOTE_LINE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    return _URL.sub(" ", text)


_WORD = re.compile(r"[a-z][a-z'-]{2,}")
_GENERIC = set(
    """the a an and or but if then this that these those it its is are was were be been have
    has had do does did will would can could should you your they them their we our i me my
    for with from about into over just like get got really very much more most some any all
    what when where who how why there here not no yes too also even still back way one two
    thing things know think want need make made going time day days people good bad better
    best new old right now well lot""".split()
)


def language_bank(docs: list[Document], n: int = 40) -> list[tuple[str, int]]:
    """The actual words and phrases people use -- raw material for ad copy.

    Bigrams are included because customer language is phrasal ("ice bath",
    "water quality", "worth the money") and single words lose the meaning.
    """
    uni: Counter[str] = Counter()
    bi: Counter[str] = Counter()
    for d in docs:
        words = [w for w in _WORD.findall(strip_noise(d.searchable_text).lower())]
        keep = [w for w in words if w not in _GENERIC]
        uni.update(keep)
        for a, b in zip(words, words[1:]):
            if a not in _GENERIC and b not in _GENERIC:
                bi.update([f"{a} {b}"])
    merged = Counter()
    merged.update({w: c for w, c in uni.items() if c >= 3})
    merged.update({w: int(c * 1.8) for w, c in bi.items() if c >= 3})  # favour phrases
    return merged.most_common(n)


def corpus_stats(docs: list[Document]) -> dict[str, object]:
    if not docs:
        return {"documents": 0}
    dates = [d.created_at for d in docs]
    return {
        "documents": len(docs),
        "posts": sum(1 for d in docs if d.kind == "post"),
        "comments": sum(1 for d in docs if d.kind == "comment"),
        "articles": sum(1 for d in docs if d.kind == "article"),
        "platforms": dict(Counter(d.platform for d in docs)),
        "communities": dict(Counter(d.community for d in docs if d.community).most_common(15)),
        "authors": len({d.author for d in docs if d.author}),
        "earliest": min(dates).date().isoformat(),
        "latest": max(dates).date().isoformat(),
    }
