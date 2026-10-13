"""Per-platform export profiles and caption/hashtag text."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Platform:
    key: str
    name: str
    max_duration: float     # longest upload the short-form feed accepts
    fps_cap: float
    crf: int
    maxrate: str            # peak video bitrate
    audio_bitrate: str
    caption_margin_v: int   # px from bottom (of 1920) to keep captions clear of the app's UI
    base_tags: tuple[str, ...]
    max_hashtags: int
    title_limit: int | None


PLATFORMS = {
    # TikTok re-encodes aggressively, so upload high quality; its caption/description UI covers ~the bottom third.
    "tiktok": Platform("tiktok", "TikTok", 600, 60, 18, "15M", "192k", 640, ("fyp", "foryou"), 5, None),
    # Instagram recommends 30 fps; Reels overlay is similar to TikTok.
    "instagram": Platform("instagram", "Instagram Reels", 180, 30, 19, "8M", "128k", 600, ("reels",), 5, None),
    # Shorts: up to 3 min; bottom UI is a little shorter. Title max 100 chars.
    "youtube": Platform("youtube", "YouTube Shorts", 180, 60, 18, "20M", "192k", 520, ("shorts",), 3, 100),
}

_STOP = set("""
a about above after again against all also am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few
for from further get got gonna had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just know let's like
me more most mustn't my myself no nor not now of off on once one only or other ought our ours ourselves out over
own really right same say said she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've thing things think this
those through to too under until up us very want was wasn't way we we'd we'll we're we've well were weren't what
what's when when's where where's which while who who's whom why why's will with won't would wouldn't yeah yes
you you'd you'll you're you've your yours yourself yourselves okay ok oh um uh go going gonna kind lot actually
mean see even back make made much many still something stuff guys without within around
every never always another anything everything nothing today time times part video channel
first last next new old good great little big better best two three four five
""".split())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9']+", text.lower()) if w not in _STOP and len(w) > 2]


def hashtags(clip_text: str, clip_title: str, video_title: str, video_tags: list[str],
             extra: list[str], limit: int) -> list[str]:
    counts = Counter(_words(clip_text))
    # Title words only count if the clip actually talks about them.
    for w in _words(clip_title) + _words(video_title):
        if w in counts:
            counts[w] += 3
    for t in video_tags[:15]:
        tag = re.sub(r"[^A-Za-z0-9]", "", t)
        if tag and len(tag) <= 24:
            counts[tag.lower()] += 2
    tags = [re.sub(r"[^A-Za-z0-9_]", "", t.lstrip("#")) for t in extra]
    for w, _ in counts.most_common():
        if len(tags) >= limit:
            break
        w = w.replace("'", "")
        if w not in (t.lower() for t in tags):
            tags.append(w)
    return [t for t in tags if t]


def caption_text(p: Platform, title: str, hook: str, tags: list[str], url: str | None) -> str:
    all_tags = " ".join(f"#{t}" for t in [*tags, *p.base_tags])
    if p.key == "youtube":
        yt_title = title if len(title) + 8 <= p.title_limit else title[: p.title_limit - 9].rstrip() + "…"
        lines = [f"TITLE:\n{yt_title} #shorts", "", "DESCRIPTION:", hook]
        if url:
            lines += ["", f"Full video: {url}"]
        lines += ["", all_tags]
    else:
        lines = ["CAPTION:", f"{title}\n\n{hook}" if hook and hook != title else title, "", all_tags]
        if url and p.key == "tiktok":
            lines += ["", f"(source: {url})"]
    return "\n".join(lines).strip() + "\n"
