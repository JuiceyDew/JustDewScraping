"""Render a run into Markdown and a self-contained HTML report.

Charts are inline SVG generated here rather than a JS charting library: the HTML
must render with no network access, because these reports get emailed around and
opened from disk by people who will never run a web server.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment

from ideafindr.models import Quote, Run, Theme

STANCE_LABEL = {
    "pain_point": "Pain point",
    "desire": "Desire",
    "objection": "Objection",
    "praise": "Praise",
    "question": "Question",
    "neutral": "Neutral",
}


def sparkline(points: list[tuple[str, int]], w: int = 180, h: int = 30) -> str:
    """Tiny inline SVG trend line."""
    if len(points) < 2:
        return ""
    vals = [v for _, v in points]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1)
    step = w / (len(vals) - 1)
    pts = " ".join(
        f"{i * step:.1f},{h - 2 - (v - lo) / span * (h - 6):.1f}" for i, v in enumerate(vals)
    )
    return (
        f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'preserveAspectRatio="none" role="img" aria-label="trend">'
        f'<polyline points="{pts}" fill="none" stroke="currentColor" '
        f'stroke-width="1.5" stroke-linejoin="round"/></svg>'
    )


def bar(pct: float, w: int = 160, h: int = 8) -> str:
    fill = max(0.0, min(100.0, pct)) / 100 * w
    return (
        f'<svg class="bar" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">'
        f'<rect width="{w}" height="{h}" rx="4" fill="currentColor" opacity="0.12"/>'
        f'<rect width="{fill:.1f}" height="{h}" rx="4" fill="currentColor"/></svg>'
    )


def momentum_tag(m: float) -> tuple[str, str]:
    if m >= 1.3:
        return "rising", "&#9650;"
    if m <= 0.7:
        return "fading", "&#9660;"
    return "steady", "&#9644;"


# --- markdown -----------------------------------------------------------------

MD = """# {{ run.topic }}

*Social listening report — generated {{ now }}*

{{ stats.documents }} documents from {{ stats.platforms | length }} platform(s),
{{ stats.earliest }} to {{ stats.latest }}, {{ stats.authors }} distinct authors.

{% if summary %}
## Executive summary

{{ summary }}
{% endif %}

## Theme map

| # | Theme | Docs | Momentum | Stance |
|---|-------|------|----------|--------|
{% for t in themes -%}
| {{ loop.index }} | {{ t.label }} | {{ t.volume }} | {{ "%.2f"|format(t.momentum) }} | {{ stance[t.stance] }} |
{% endfor %}

## Voice of the customer

{% for t in themes %}
### {{ loop.index }}. {{ t.label }}

{{ t.description }}

*{{ t.volume }} documents · momentum {{ "%.2f"|format(t.momentum) }} · {{ stance[t.stance] }}*
{% if t.keywords %}
Keywords: {{ t.keywords | join(", ") }}
{% endif %}
{% for q in t.quotes %}
> {{ q.text }}
>
> — [{{ q.author or "anon" }} in {{ q.community or "web" }}]({{ q.url }}), {{ q.created_at.date() }}
{% endfor %}
{% endfor %}

## Language bank

The words and phrases people actually use — raw material for copy.

{% for w, c in language %}`{{ w }}` ({{ c }}){% if not loop.last %} · {% endif %}{% endfor %}

{% if sov %}
## Share of voice

| Brand | Mentions | Share | Engagement |
|-------|----------|-------|------------|
{% for b, d in sov.items() -%}
| {{ b }} | {{ d.mentions }} | {{ d.share }}% | {{ d.engagement }} |
{% endfor %}
{% endif %}

## Methodology

- **Sources:** {{ run.sources | join(", ") }}
- **Window:** last {{ run.plan.days }} days ({{ stats.earliest }} to {{ stats.latest }})
- **Corpus:** {{ stats.documents }} documents ({{ stats.posts }} posts, {{ stats.comments }} comments, {{ stats.articles }} articles)
- **Communities:** {% for c, n in stats.communities.items() %}{{ c }} ({{ n }}){% if not loop.last %}, {% endif %}{% endfor %}
- **Themes:** {{ themes | length }}, found by embedding + k-means clustering, named by LLM
{% if unthemed %}- **Unthemed:** {{ unthemed }} documents fell into clusters the labeller judged incoherent; they count in the corpus and language bank but have no theme section{% endif %}

**Caveats.** Reddit over-represents some demographics (younger, more male,
more English-speaking, more technical) and under-represents others. Treat these
themes as a picture of what an engaged online subset says, not of the whole
market. Web articles carry their collection date, not their publication date, so
they are excluded from trend and momentum figures. Momentum compares a theme's
share of the last 30 days against its share of the whole window; it measures
relative change within this corpus, not absolute market growth.
"""

# --- html ---------------------------------------------------------------------

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ run.topic }} — social listening</title>
<style>
  :root{
    --bg:#fbfaf8; --fg:#1a1a18; --muted:#6b6862; --line:#e3ded6;
    --card:#fff; --accent:#b4532a; --rise:#2f7a4f; --fade:#9a6b2f;
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#16161a; --fg:#e8e6e1; --muted:#9a968e; --line:#2c2c33;
    --card:#1d1d22; --accent:#e08a5e; --rise:#68c091; --fade:#d0a35f;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:40px 20px 80px}
  h1{font-size:2rem;line-height:1.2;margin:0 0 .2em}
  h2{font-size:1.3rem;margin:2.4em 0 .8em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
  h3{font-size:1.05rem;margin:0 0 .3em}
  .sub{color:var(--muted);font-size:.9rem;margin-bottom:2em}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;
    padding:18px 20px;margin:14px 0}
  .meta{color:var(--muted);font-size:.82rem;margin:.4em 0 .8em}
  .pill{display:inline-block;font-size:.72rem;padding:2px 8px;border-radius:99px;
    border:1px solid var(--line);color:var(--muted);margin-right:6px}
  .rising{color:var(--rise);border-color:currentColor}
  .fading{color:var(--fade);border-color:currentColor}
  blockquote{margin:12px 0;padding:10px 16px;border-left:3px solid var(--accent);
    background:color-mix(in srgb,var(--accent) 5%,transparent);border-radius:0 6px 6px 0}
  blockquote p{margin:0 0 6px}
  blockquote cite{font-size:.8rem;color:var(--muted);font-style:normal}
  a{color:var(--accent)}
  table{border-collapse:collapse;width:100%;font-size:.9rem}
  th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
  .spark{color:var(--accent);vertical-align:middle}
  .bar{color:var(--accent);vertical-align:middle}
  .lang span{display:inline-block;margin:0 6px 6px 0;padding:3px 9px;border:1px solid var(--line);
    border-radius:6px;font-size:.85rem;background:var(--card)}
  .lang small{color:var(--muted)}
  .tbl-wrap{overflow-x:auto}
  .caveat{font-size:.88rem;color:var(--muted);background:var(--card);
    border:1px solid var(--line);border-left:3px solid var(--fade);border-radius:0 8px 8px 0;padding:14px 18px}
  @media print{body{background:#fff}.card{break-inside:avoid}}
</style></head><body><div class="wrap">

<h1>{{ run.topic }}</h1>
<div class="sub">Social listening report · generated {{ now }}<br>
{{ stats.documents }} documents · {{ stats.earliest }} to {{ stats.latest }} · {{ stats.authors }} distinct authors</div>

{% if summary %}
<h2>Executive summary</h2>
<div class="card">{{ summary_html | safe }}</div>
{% endif %}

<h2>Theme map</h2>
<div class="tbl-wrap"><table>
<tr><th>Theme</th><th>Docs</th><th>Trend</th><th>Momentum</th><th>Stance</th></tr>
{% for t in themes %}
<tr>
  <td>{{ t.label }}</td>
  <td>{{ t.volume }}</td>
  <td>{{ t.spark | safe }}</td>
  <td><span class="pill {{ t.mom_cls }}">{{ t.mom_icon | safe }} {{ "%.2f"|format(t.momentum) }}</span></td>
  <td>{{ stance[t.stance] }}</td>
</tr>
{% endfor %}
</table></div>

<h2>Voice of the customer</h2>
{% for t in themes %}
<div class="card">
  <h3>{{ loop.index }}. {{ t.label }}</h3>
  <div class="meta">
    <span class="pill">{{ stance[t.stance] }}</span>
    <span class="pill {{ t.mom_cls }}">{{ t.mom_icon | safe }} momentum {{ "%.2f"|format(t.momentum) }}</span>
    <span class="pill">{{ t.volume }} docs</span>
    {{ t.spark | safe }}
  </div>
  {% if t.description %}<p>{{ t.description }}</p>{% endif %}
  {% for q in t.quotes %}
  <blockquote><p>{{ q.text }}</p>
    <cite>— {{ q.author or "anon" }} in {{ q.community or "web" }},
    {{ q.created_at.date() }} · <a href="{{ q.url }}" rel="noopener">source</a></cite>
  </blockquote>
  {% endfor %}
  {% if t.keywords %}<div class="meta">Keywords: {{ t.keywords | join(", ") }}</div>{% endif %}
</div>
{% endfor %}

<h2>Language bank</h2>
<p class="meta">The words and phrases people actually use — raw material for copy.</p>
<div class="lang">{% for w, c in language %}<span>{{ w }} <small>{{ c }}</small></span>{% endfor %}</div>

{% if sov %}
<h2>Share of voice</h2>
<div class="tbl-wrap"><table>
<tr><th>Brand</th><th>Mentions</th><th>Share</th><th></th><th>Engagement</th></tr>
{% for b, d in sov.items() %}
<tr><td>{{ b }}</td><td>{{ d.mentions }}</td><td>{{ d.share }}%</td>
<td>{{ d.bar | safe }}</td><td>{{ d.engagement }}</td></tr>
{% endfor %}
</table></div>
{% endif %}

<h2>Methodology</h2>
<div class="card">
<table>
<tr><th>Sources</th><td>{{ run.sources | join(", ") }}</td></tr>
<tr><th>Window</th><td>last {{ run.plan.days }} days ({{ stats.earliest }} → {{ stats.latest }})</td></tr>
<tr><th>Corpus</th><td>{{ stats.documents }} docs — {{ stats.posts }} posts, {{ stats.comments }} comments, {{ stats.articles }} articles</td></tr>
<tr><th>Communities</th><td>{% for c, n in stats.communities.items() %}{{ c }} ({{ n }}){% if not loop.last %}, {% endif %}{% endfor %}</td></tr>
<tr><th>Themes</th><td>{{ themes | length }} — embedding + k-means clustering, named by LLM</td></tr>
{% if unthemed %}<tr><th>Unthemed</th><td>{{ unthemed }} documents in clusters the labeller judged incoherent — counted in the corpus and language bank, but given no theme section</td></tr>{% endif %}
</table>
</div>
<p class="caveat"><strong>Caveats.</strong> Reddit over-represents some demographics
(younger, more male, more English-speaking, more technical) and under-represents
others — treat these themes as what an engaged online subset says, not the whole
market. Web articles carry their collection date, not publication date, so they
are excluded from trend and momentum figures. Momentum compares a theme's share
of the last 30 days against its share of the full window: it measures relative
change <em>within this corpus</em>, not absolute market growth.</p>

</div></body></html>
"""


def _md_to_html(text: str) -> str:
    """Minimal markdown for the LLM's summary: paragraphs, bold, headings, bullets."""
    import re

    out: list[str] = []
    for block in text.split("\n\n"):
        b = html.escape(block.strip())
        if not b:
            continue
        b = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", b)
        b = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" rel="noopener">\1</a>', b)
        if b.startswith("#"):
            lvl = len(b) - len(b.lstrip("#"))
            out.append(f"<h{min(lvl+2,5)}>{b.lstrip('# ')}</h{min(lvl+2,5)}>")
        elif b.lstrip().startswith(("- ", "* ")):
            items = "".join(
                f"<li>{ln.lstrip('-* ').strip()}</li>" for ln in b.split("\n") if ln.strip()
            )
            out.append(f"<ul>{items}</ul>")
        else:
            out.append(f"<p>{b.replace(chr(10), '<br>')}</p>")
    return "\n".join(out)


def render(
    run: Run,
    themes: list[Theme],
    stats: dict,
    language: list[tuple[str, int]],
    sov: dict,
    summary: str = "",
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    # Two environments, deliberately. Every value interpolated below -- theme
    # labels, descriptions, keywords, quote text, authors, communities -- is
    # scraped from Reddit or written by an LLM, so the HTML must escape it: these
    # reports get emailed around and opened from disk, where injected markup runs.
    # Markdown must NOT be escaped: &, < and > are ordinary prose there, and
    # escaping them would corrupt the document. The HTML template marks its
    # intentional markup (`spark`, `mom_icon`, `bar`, `summary_html`) `| safe`.
    env_md = Environment(autoescape=False)
    env_html = Environment(autoescape=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # The documents in an incoherent cluster are real, so they still count in the
    # corpus stats and the language bank -- they just don't get a section that
    # implies they mean something together.
    incoherent = [t for t in themes if not t.coherent]
    themes = [t for t in themes if t.coherent]

    enriched = []
    for t in themes:
        d = t.model_dump()
        cls, icon = momentum_tag(t.momentum)
        d["spark"] = sparkline(t.trend)
        d["mom_cls"], d["mom_icon"] = cls, icon
        d["quotes"] = t.quotes
        enriched.append(d)

    sov_h = {b: {**v, "bar": bar(float(v["share"]))} for b, v in sov.items()}

    ctx = dict(
        run=run, themes=enriched, stats=stats, language=language,
        sov=sov, now=now, stance=STANCE_LABEL, summary=summary,
        unthemed=sum(t.volume for t in incoherent),
    )

    out = out_dir or Path("data/reports") / run.id
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "report.md"
    md_path.write_text(env_md.from_string(MD).render(**ctx), encoding="utf-8")

    html_ctx = {**ctx, "sov": sov_h, "summary_html": _md_to_html(summary)}
    html_path = out / "report.html"
    html_path.write_text(env_html.from_string(HTML).render(**html_ctx), encoding="utf-8")

    return md_path, html_path
