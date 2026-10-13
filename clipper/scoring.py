"""Score candidate windows to find the most clip-worthy sections.

heuristic: audio loudness peaks + speech density (no extra dependencies)
ai:        heuristic + a free Hugging Face zero-shot classifier reading the transcript
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .transcribe import Sentence

DEFAULT_HF_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
LABELS = [
    "funny or hilarious",
    "surprising or shocking",
    "emotional or heartfelt",
    "a useful tip or insight",
    "intense or exciting",
    "a bold or controversial opinion",
]
HOP = 0.5  # seconds per loudness frame


def ai_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def loudness_track(wav: Path) -> np.ndarray:
    """RMS level in dB per HOP seconds, streamed so long VODs don't need the whole file in memory."""
    levels = []
    with wave.open(str(wav), "rb") as f:
        n = int(f.getframerate() * HOP)
        while True:
            buf = f.readframes(n)
            if not buf:
                break
            x = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            levels.append(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-6))
    return np.asarray(levels)


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.full_like(x, 0.5) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


class Scorer:
    def __init__(self, kind: str, wav: Path, offset: float, sentences: list[Sentence],
                 hf_model: str = DEFAULT_HF_MODEL):
        self.kind = kind
        self.offset = offset
        self.sentences = sentences
        self.levels = loudness_track(wav)
        self.hf_model = hf_model
        self._units: list[tuple[float, float, float]] | None = None  # (start, end, text score)

    def _energy(self, s: float, e: float) -> float:
        a, b = int((s - self.offset) / HOP), int((e - self.offset) / HOP)
        seg = self.levels[max(a, 0):max(b, a + 1)]
        if seg.size == 0:
            return 0.0
        top = np.sort(seg)[-max(1, seg.size // 4):]  # loudest quarter: laughs, shouting, hype
        return float(np.mean(top) - np.median(self.levels))

    def _density(self, s: float, e: float) -> float:
        n = sum(len(x.words) for x in self.sentences if x.start >= s and x.end <= e)
        return n / max(e - s, 1.0)

    def _text_units(self) -> list[tuple[float, float, float]]:
        """Classify ~12 s blocks of transcript once; windows then average the overlapping blocks."""
        if self._units is not None:
            return self._units
        import torch
        from transformers import pipeline

        blocks: list[list[Sentence]] = []
        for sent in self.sentences:
            if blocks and blocks[-1][-1].end - blocks[-1][0].start < 12.0:
                blocks[-1].append(sent)
            else:
                blocks.append([sent])
        print(f"  scoring {len(blocks)} transcript blocks with {self.hf_model}")
        clf = pipeline("zero-shot-classification", model=self.hf_model,
                       device=0 if torch.cuda.is_available() else -1)
        texts = [" ".join(s.text for s in b) for b in blocks]
        results = clf(texts, candidate_labels=LABELS, multi_label=True,
                      hypothesis_template="This passage is {}.", batch_size=16)
        if isinstance(results, dict):
            results = [results]
        self._units = [(b[0].start, b[-1].end, max(r["scores"])) for b, r in zip(blocks, results)]
        return self._units

    def _text(self, s: float, e: float) -> float:
        units = [(us, ue, sc) for us, ue, sc in self._text_units() if ue > s and us < e]
        if not units:
            return 0.0
        weights = np.array([min(ue, e) - max(us, s) for us, ue, _ in units])
        scores = np.array([sc for _, _, sc in units])
        # Weighted mean, plus a bonus for a strong opening (the hook matters most on shorts).
        return float(0.7 * np.average(scores, weights=weights) + 0.3 * scores[0])

    def __call__(self, windows: list[tuple[float, float]]) -> list[float]:
        energy = _norm(np.array([self._energy(s, e) for s, e in windows]))
        density = _norm(np.array([self._density(s, e) for s, e in windows]))
        if self.kind == "ai" and self.sentences:
            text = _norm(np.array([self._text(s, e) for s, e in windows]))
            total = 0.6 * text + 0.25 * energy + 0.15 * density
        else:
            total = 0.6 * energy + 0.4 * density
        return [round(float(x), 4) for x in total]
