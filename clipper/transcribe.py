"""Speech-to-text with word-level timestamps (faster-whisper, models pulled from Hugging Face)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Sentence:
    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def extract_audio(src: Path, wav: Path, start: float, end: float) -> Path:
    """16 kHz mono WAV of [start, end]; reused by whisper and the loudness scorer."""
    if not wav.exists():
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
             "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
            check=True,
        )
    return wav


def _add_cuda_dlls() -> None:
    """Expose CUDA libraries installed via requirements-gpu.txt (nvidia-cublas-cu12, nvidia-cudnn-cu12)."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for bin_dir in Path(base).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def _run_whisper(wav: Path, model_size: str, language: str | None, device: str) -> list[Word]:
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    # Hand over raw samples: we already decoded with ffmpeg, so PyAV (and its version quirks) isn't needed.
    with wave.open(str(wav), "rb") as f:
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model.transcribe(audio, word_timestamps=True, vad_filter=True, language=language)
    print(f"  language: {info.language}  (audio {info.duration / 60:.1f} min, device {device})")
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(float(w.start), float(w.end), text))
        print(f"\r  transcribed {seg.end / 60:6.1f} / {info.duration / 60:.1f} min", end="", flush=True)
    print()
    return words


def transcribe(wav: Path, cache: Path, offset: float, model_size: str,
               language: str | None, device: str) -> list[Word]:
    key = {"model": model_size, "language": language, "offset": offset, "wav": wav.name}
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("key") == key:
            return [Word(**w) for w in data["words"]]

    _add_cuda_dlls()
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    try:
        words = _run_whisper(wav, model_size, language, device)
    except Exception as e:  # usually missing cuBLAS/cuDNN DLLs on Windows
        if device == "cpu":
            raise
        print(f"\n  GPU transcription failed ({e}); falling back to CPU")
        words = _run_whisper(wav, model_size, language, "cpu")

    for w in words:
        w.start += offset
        w.end += offset
    cache.write_text(json.dumps({"key": key, "words": [asdict(w) for w in words]}), encoding="utf-8")
    return words


_END = re.compile(r"[.!?…]['\")\]]*$")


def build_sentences(words: list[Word], max_len: float = 15.0) -> list[Sentence]:
    """Group words into sentences; long run-ons are broken at commas or forcibly at max_len."""
    sentences: list[Sentence] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        dur = w.end - cur[0].start
        if (nxt is None or _END.search(w.text) or nxt.start - w.end > 1.0
                or (dur > max_len / 2 and w.text.endswith(",")) or dur > max_len):
            sentences.append(Sentence(cur[0].start, cur[-1].end, cur))
            cur = []
    return sentences
