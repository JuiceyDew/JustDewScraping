from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ideafindr.models import Document, RunPlan


@pytest.fixture
def plan() -> RunPlan:
    return RunPlan(
        topic="cold plunge tubs",
        subreddits=["coldplunge"],
        keywords=["chiller", "water"],
        brands=["Ice Barrel", "Plunge"],
        days=90,
    )


def make_doc(i: int, text: str, *, kind: str = "post", days_ago: int = 10, **kw) -> Document:
    return Document(
        id=kw.pop("id", f"reddit:{i}"),
        platform=kw.pop("platform", "reddit"),
        kind=kind,
        url=kw.pop("url", f"https://www.reddit.com/r/coldplunge/comments/{i}/x/"),
        title=kw.pop("title", None),
        text=text,
        author=kw.pop("author", f"user{i}"),
        created_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        community=kw.pop("community", "coldplunge"),
        engagement=kw.pop("engagement", {"score": i, "num_comments": 0}),
        **kw,
    )


@pytest.fixture
def docs() -> list[Document]:
    themes = {
        "noise": ["the chiller is far too loud at night and wakes everyone up",
                  "my chiller noise is unbearable, neighbours complained twice",
                  "loud humming from the chiller unit all night long",
                  "chiller noise level is brutal, anyone fixed this"],
        "water": ["water gets cloudy after about a week even with ozone",
                  "how often do you change the water in your plunge",
                  "filter keeps clogging with debris and hair",
                  "water quality is a constant battle with daily use"],
        "cost":  ["electricity bill went up a lot after buying the tub",
                  "running cost per month is honestly brutal",
                  "power consumption of the chiller is higher than advertised",
                  "is a cold plunge really worth the money"],
    }
    out = []
    i = 0
    for name, texts in themes.items():
        for t in texts * 3:
            i += 1
            out.append(make_doc(i, t, kind="comment", days_ago=(i % 60) + 1))
    return out
