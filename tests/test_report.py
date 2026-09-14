from datetime import datetime, timezone

from ideafindr.models import Quote, Run, RunPlan, Theme
from ideafindr.report.render import _md_to_html, bar, momentum_tag, render, sparkline


def test_sparkline_needs_two_points():
    assert sparkline([("2026-01-01", 5)]) == ""
    assert "<svg" in sparkline([("2026-01-01", 5), ("2026-01-08", 9)])


def test_bar_clamps_out_of_range_shares():
    assert "<svg" in bar(150.0) and "<svg" in bar(-10.0)


def test_momentum_tags():
    assert momentum_tag(1.5)[0] == "rising"
    assert momentum_tag(0.4)[0] == "fading"
    assert momentum_tag(1.0)[0] == "steady"


def test_md_to_html_escapes_injected_markup():
    out = _md_to_html("<script>alert(1)</script> **bold**")
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<strong>bold</strong>" in out


def test_html_escapes_scraped_content_everywhere_not_just_the_summary(tmp_path):
    """Theme labels, descriptions, keywords and quote text all come from scraped
    Reddit or from an LLM. These reports are emailed and opened from disk, so
    injected markup would run. Only the summary used to be escaped."""
    plan = RunPlan(topic="cold plunge tubs", days=90)
    run = Run(id="r1", topic="cold plunge tubs", created_at=datetime.now(timezone.utc),
              plan=plan, sources=["reddit"])
    evil = "<script>alert(1)</script>"
    theme = Theme(
        id=0, label=f"Chiller noise {evil}", description=f"Owners say {evil}",
        stance="pain_point", doc_ids=["reddit:1"], keywords=[evil, "noise"],
        volume=12, momentum=1.4, trend=[("2026-01-01", 3), ("2026-01-08", 9)],
        quotes=[Quote(text=f"The chiller is unbearable at 6am {evil} and my neighbour complained",
                      url="https://www.reddit.com/r/coldplunge/comments/x/y/",
                      author=evil, community=evil,
                      created_at=datetime.now(timezone.utc))],
    )
    stats = {"documents": 12, "posts": 4, "comments": 8, "articles": 0,
             "platforms": {"reddit": 12}, "communities": {evil: 12},
             "authors": 9, "earliest": "2026-01-01", "latest": "2026-03-01"}
    _, html_path = render(run, [theme], stats, [(evil, 9)], {}, out_dir=tmp_path)

    page = html_path.read_text()
    assert "<script>" not in page, "scraped markup must never reach the page raw"
    assert "&lt;script&gt;" in page, "it should appear escaped instead"
    # The report's own markup must survive escaping.
    assert "<svg" in page and 'class="spark"' in page, "sparklines are intentional markup"


def test_report_is_self_contained_and_cites_sources(tmp_path):
    plan = RunPlan(topic="cold plunge tubs", days=90, brands=["Ice Barrel"])
    run = Run(id="r1", topic="cold plunge tubs", created_at=datetime.now(timezone.utc),
              plan=plan, sources=["reddit"])
    theme = Theme(
        id=0, label="Chiller noise makes daily use antisocial",
        description="Owners report the compressor is loud enough to annoy neighbours.",
        stance="pain_point", doc_ids=["reddit:1"], keywords=["chiller", "noise"],
        volume=12, momentum=1.4, trend=[("2026-01-01", 3), ("2026-01-08", 9)],
        quotes=[Quote(text="The chiller noise is unbearable at 6am, my neighbour complained twice.",
                      url="https://www.reddit.com/r/coldplunge/comments/x/y/",
                      author="someone", community="coldplunge",
                      created_at=datetime.now(timezone.utc))],
    )
    stats = {"documents": 12, "posts": 4, "comments": 8, "articles": 0,
             "platforms": {"reddit": 12}, "communities": {"coldplunge": 12},
             "authors": 9, "earliest": "2026-01-01", "latest": "2026-03-01"}
    md, html = render(run, [theme], stats, [("chiller", 9)],
                      {"Ice Barrel": {"mentions": 3, "share": 100.0, "engagement": 5.0, "trend": []}},
                      summary="**Headline** finding.", out_dir=tmp_path)

    h = md.read_text()
    assert "Chiller noise makes daily use antisocial" in h
    assert "https://www.reddit.com/r/coldplunge/comments/x/y/" in h
    assert "over-represents" in h, "methodology caveats must survive into the report"

    page = html.read_text()
    # Nothing may be fetched from the network: these get emailed and opened from disk.
    import re
    assert not re.findall(r'(?:src|href)="https?://(?!www\.reddit)', page)
    assert "<svg" in page and "prefers-color-scheme" in page
