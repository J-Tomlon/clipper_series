"""Word-by-word captions as an ASS subtitle file (burned in by ffmpeg/libass)."""
from __future__ import annotations

import re
from pathlib import Path

from .reframe import OUT_H, OUT_W
from .transcribe import Word

YELLOW = r"&H0000FFFF&"  # ASS colours are &HAABBGGRR


def _ts(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _clean(text: str, upper: bool) -> str:
    text = re.sub(r"[{}\\]", "", text).strip(",;:")
    if not text.endswith("..."):
        text = text.rstrip(".")
    return text.upper() if upper else text


def build_ass(words: list[Word], clip_start: float, clip_end: float, out: Path, *, style: str = "highlight",
              margin_v: int = 600, font: str = "Arial", upper: bool = True, words_per_line: int = 3) -> Path:
    raw = [w for w in words if w.end > clip_start and w.start < clip_end and _clean(w.text, upper)]
    ws = [Word(w.start - clip_start, w.end - clip_start, _clean(w.text, upper)) for w in raw]
    dur = clip_end - clip_start
    size = 110 if style == "word" else 84

    # When does each word stop being shown? Until the next word, unless there's a pause.
    ends = []
    for i, w in enumerate(ws):
        nxt = ws[i + 1].start if i + 1 < len(ws) else dur
        ends.append(min(nxt, dur) if nxt - w.end < 0.6 else w.end + 0.25)

    events = []
    if style == "word":
        for w, e in zip(ws, ends):
            events.append((w.start, e, r"{\fscx125\fscy125\t(0,90,\fscx100\fscy100)}" + w.text))
    else:
        # Groups of N words; the word being spoken is highlighted.
        groups: list[list[int]] = [[]]
        for i, w in enumerate(ws):
            g = groups[-1]
            if g and (len(g) >= words_per_line or w.start - ws[g[-1]].end > 0.6
                      or re.search(r"[.!?]$", raw[g[-1]].text)):
                groups.append([])
            groups[-1].append(i)
        for g in groups:
            for i in g:
                parts = [(r"{\c" + YELLOW + r"\fscx108\fscy108}" + ws[j].text + r"{\r}") if j == i else ws[j].text
                         for j in g]
                events.append((ws[i].start, ends[i], " ".join(parts)))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,7,3,2,90,90,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [f"Dialogue: 0,{_ts(s)},{_ts(e)},Default,,0,0,0,,{t}" for s, e, t in events if e > s]
    out.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return out
