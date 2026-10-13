"""Decide which parts of the video become clips: intro/outro removal, chapters, splitting, picking."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .source import Chapter, Source
from .transcribe import Sentence

Range = tuple[float, float]
SPILL = 2.5

# Chapter titles that mean "not content". Matched against the whole (normalised) title so that
# e.g. "Introduction to Rust" is kept but "Intro" / "0:00 Intro" is skipped.
_SKIP_CHAPTER = re.compile(
    r"^(the )?(intro(duction)?|outro|opening|ending|end ?screen|end ?card|credits|sponsor(ed)?( segment| message| read| spot)?"
    r"|(the )?end|ad( read| break)?|preview|teaser|trailer|starting soon|stream starting|brb|be right back|thanks for watching"
    r"|patreon( shoutouts?)?|shout ?outs?|subscribe)$"
)
_INTRO_PHRASES = re.compile(
    r"welcome (back )?to (the|my|this|our) (channel|video|stream|show|podcast)|what'?s up,? (guys|everyone|everybody|y'?all)"
    r"|^(hey|hi|hello),? (guys|everyone|everybody|y'?all|friends)|this (video|episode) is sponsored|today'?s sponsor"
    r"|thanks? to .{0,40} for sponsoring|before we (get started|begin|start|dive in)",
    re.I,
)
_OUTRO_PHRASES = re.compile(
    r"thanks? (you )?(so much )?for watching|see (you|ya) (guys )?(all )?(in the |on the )?next (time|one|video|week|stream)"
    r"|(like|hit) (and|&) subscribe|smash (that|the) like|hit (that|the) (bell|like|subscribe)"
    r"|don'?t forget to (like|subscribe)|that'?s (it|all) for (today|this video|now|this week)|until next time"
    r"|catch you (guys )?(in the )?next|leave a comment (below|down below)|peace out",
    re.I,
)


@dataclass
class Clip:
    start: float
    end: float
    title: str
    chapter: str | None = None
    score: float | None = None
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Settings:
    min_len: float = 15.0
    target_len: float = 45.0
    max_len: float = 60.0
    max_clips: int | None = None
    per_chapter: int = 1
    skip_intro: str = "auto"  # "auto" | seconds
    skip_outro: str = "auto"


def _norm_title(t: str) -> str:
    t = re.sub(r"^\s*[\d:.\-–—)]+\s*", "", t.lower())
    return re.sub(r"[^a-z0-9' ]+", " ", t).strip()


def subtract(ranges: list[Range], cuts: list[Range]) -> list[Range]:
    out = list(ranges)
    for cs, ce in cuts:
        nxt = []
        for s, e in out:
            if ce <= s or cs >= e:
                nxt.append((s, e))
                continue
            if cs > s:
                nxt.append((s, cs))
            if ce < e:
                nxt.append((ce, e))
        out = nxt
    return [r for r in out if r[1] - r[0] > 0.5]


def find_exclusions(src: Source, sentences: list[Sentence], cfg: Settings) -> list[tuple[float, float, str]]:
    ex = list(src.skip_ranges)
    for ch in src.chapters:
        if _SKIP_CHAPTER.match(_norm_title(ch.title)):
            ex.append((ch.start, ch.end, f"chapter:{ch.title}"))
    have_intro = any(s <= 1.0 for s, _, _ in ex)
    have_outro = any(e >= src.duration - 1.0 for _, e, _ in ex)

    # Intro
    if cfg.skip_intro != "auto":
        if float(cfg.skip_intro) > 0:
            ex.append((0.0, float(cfg.skip_intro), "manual intro"))
    elif not have_intro and sentences:
        cut = max(0.0, sentences[0].start - 0.3)  # dead air / music before anyone talks
        reason = "no speech"
        for s in sentences:
            if s.start > 90:
                break
            if _INTRO_PHRASES.search(s.text):
                cut, reason = s.end, "intro phrase"
        if cut > 2.0:
            ex.append((0.0, cut, f"auto intro ({reason})"))

    # Outro
    if cfg.skip_outro != "auto":
        if float(cfg.skip_outro) > 0:
            ex.append((src.duration - float(cfg.skip_outro), src.duration, "manual outro"))
    elif not have_outro and sentences:
        window_start = src.duration - min(180.0, src.duration * 0.25)
        cut, reason = sentences[-1].end + 0.5, "no speech"
        for s in sentences:
            if s.start >= window_start and _OUTRO_PHRASES.search(s.text):
                cut, reason = s.start, "outro phrase"
                break
        if src.duration - cut > 2.0:
            ex.append((cut, src.duration, f"auto outro ({reason})"))
    return ex


def _snap(s: float, e: float, sentences: list[Sentence], region: Range) -> tuple[float, float, str]:
    """Expand to whole sentences and pad slightly so first/last words aren't clipped.

    Chapter markers are often a second or two off, so a clip may spill up to SPILL seconds past its
    region to finish a sentence.
    """
    inside = [x for x in sentences if s <= (x.start + x.end) / 2 <= e]
    if inside:
        s, e = inside[0].start, inside[-1].end
    s = max(region[0] - SPILL, s - 0.15, 0.0)
    e = min(region[1] + SPILL, e + 0.4)
    return s, e, " ".join(x.text for x in inside)


def _sentences_in(region: Range, sentences: list[Sentence]) -> list[Sentence]:
    return [x for x in sentences if region[0] <= (x.start + x.end) / 2 <= region[1]]


def split_region(region: Range, sentences: list[Sentence], cfg: Settings) -> list[Range]:
    """Cut a region into consecutive clips at sentence boundaries."""
    sents = _sentences_in(region, sentences)
    if not sents:  # no speech (music, gameplay): fixed-length chunks
        out, t = [], region[0]
        while region[1] - t >= cfg.min_len:
            out.append((t, min(t + cfg.target_len, region[1])))
            t += cfg.target_len
        return out

    out: list[Range] = []
    cs = ce = None
    for sent in sents:
        if cs is None:
            cs, ce = sent.start, sent.end
        elif sent.end - cs > cfg.max_len:
            if ce - cs >= cfg.min_len:
                out.append((cs, ce))
            cs, ce = sent.start, sent.end
        else:
            ce = sent.end
        if ce - cs >= cfg.target_len:
            out.append((cs, ce))
            cs = None
    if cs is not None:
        if ce - cs >= cfg.min_len:
            out.append((cs, ce))
        elif out and ce - out[-1][0] <= cfg.max_len:
            out[-1] = (out[-1][0], ce)
    return out


def candidate_windows(region: Range, sentences: list[Sentence], cfg: Settings, stride: float = 4.0) -> list[Range]:
    """Overlapping sentence-aligned windows, used when we need to *pick* the best part."""
    sents = _sentences_in(region, sentences)
    if not sents:
        out, t = [], region[0]
        while region[1] - t >= cfg.min_len:
            out.append((t, min(t + cfg.target_len, region[1])))
            t += stride
        return out
    out, last = [], -1e9
    for i, first in enumerate(sents):
        if first.start - last < stride:
            continue
        end = None
        for sent in sents[i:]:
            if sent.end - first.start > cfg.max_len:
                break
            end = sent.end
            if end - first.start >= cfg.target_len:
                break
        if end is not None and end - first.start >= cfg.min_len:
            out.append((first.start, end))
            last = first.start
    return out


def pick_best(windows: list[Range], scores: list[float], n: int) -> list[tuple[Range, float]]:
    chosen: list[tuple[Range, float]] = []
    for w, sc in sorted(zip(windows, scores), key=lambda x: -x[1]):
        if len(chosen) >= n:
            break
        if all(w[1] <= c[0] + 1.0 or w[0] >= c[1] - 1.0 for c, _ in chosen):
            chosen.append((w, sc))
    return sorted(chosen, key=lambda x: x[0][0])


def first_sentence(text: str, limit: int = 70) -> str:
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return first if len(first) <= limit else first[: limit - 1].rsplit(" ", 1)[0] + "…"


def plan(mode: str, src: Source, sentences: list[Sentence], exclusions, cfg: Settings,
         span: Range, scorer=None) -> tuple[str, list[Clip]]:
    usable = subtract([span], [(s, e) for s, e, _ in exclusions])
    chapters = [c for c in src.chapters if c.end > span[0] and c.start < span[1]]
    if mode == "auto":
        mode = "chapters" if len(chapters) >= 2 else "full"
    if mode == "chapters" and not chapters:
        print("  no chapters found; falling back to --mode full")
        mode = "full"

    # Regions never straddle a chapter boundary (chapters are natural topic breaks).
    regions: list[tuple[Range, Chapter | None]] = []
    for r in usable:
        if chapters:
            for c in chapters:
                s, e = max(r[0], c.start), min(r[1], c.end)
                if e - s > 0.5:
                    regions.append(((s, e), c))
        else:
            regions.append((r, None))

    picked: list[tuple[Range, Range, Chapter | None, float | None]] = []  # (window, region, chapter, score)
    if mode == "full":
        for region, ch in regions:
            picked += [(w, region, ch, None) for w in split_region(region, sentences, cfg)]
        if cfg.max_clips:
            picked = picked[: cfg.max_clips]

    elif mode == "chapters":
        by_chapter: dict[int, list[Range]] = {}
        for region, ch in regions:
            by_chapter.setdefault(id(ch), []).append(region)
        for ch in chapters:
            pieces = [p for p in by_chapter.get(id(ch), []) if p[1] - p[0] >= cfg.min_len]
            if not pieces:
                continue
            if len(pieces) == 1 and pieces[0][1] - pieces[0][0] <= cfg.max_len:
                picked.append((pieces[0], pieces[0], ch, None))  # whole chapter fits in one clip
                continue
            wins = [(w, p) for p in pieces for w in candidate_windows(p, sentences, cfg)]
            if not wins:
                continue
            scores = scorer([w for w, _ in wins]) if scorer else [0.0] * len(wins)
            best = pick_best([w for w, _ in wins], scores, cfg.per_chapter)
            region_of = dict(wins)
            picked += [(w, region_of[w], ch, sc) for w, sc in best]
        if cfg.max_clips:
            picked = picked[: cfg.max_clips]

    elif mode == "best":
        wins = [(w, region, ch) for region, ch in regions for w in candidate_windows(region, sentences, cfg)]
        if not wins:
            return mode, []
        scores = scorer([w for w, _, _ in wins])
        best = pick_best([w for w, _, _ in wins], scores, cfg.max_clips or 5)
        lookup = {w: (region, ch) for w, region, ch in wins}
        picked = [(w, *lookup[w], sc) for w, sc in best]
    else:
        raise ValueError(f"unknown mode {mode}")

    clips: list[Clip] = []
    parts: dict[str, int] = {}
    for (s, e), region, ch, sc in picked:
        s, e, text = _snap(s, e, sentences, region)
        title = ch.title if ch else first_sentence(text) or src.title
        clips.append(Clip(s, e, title, ch.title if ch else None, sc, text))
        parts[title] = parts.get(title, 0) + 1
    # "Chapter (Part 2)" when a chapter produced several clips
    seen: dict[str, int] = {}
    for c in clips:
        if parts[c.title] > 1:
            seen[c.title] = seen.get(c.title, 0) + 1
            c.title = f"{c.title} (Part {seen[c.title]})"
    return mode, clips
