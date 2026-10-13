"""Acquire a video (local file or URL) and gather metadata: chapters, SponsorBlock ranges."""
from __future__ import annotations

import json
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv", ".ts"}

# SponsorBlock categories that are never good clip material.
SPONSORBLOCK_CATEGORIES = ["sponsor", "intro", "outro", "selfpromo", "interaction", "preview"]


@dataclass
class Chapter:
    start: float
    end: float
    title: str


@dataclass
class Source:
    path: Path
    title: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    chapters: list[Chapter] = field(default_factory=list)
    # (start, end, reason) ranges that should never end up in a clip
    skip_ranges: list[tuple[float, float, str]] = field(default_factory=list)
    url: str | None = None
    video_id: str | None = None
    tags: list[str] = field(default_factory=list)


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def find_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def download(url: str, dest_dir: Path) -> Path:
    """Download with yt-dlp (YouTube, Twitch VODs/clips, and anything else it supports)."""
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        # Prefer H.264 so OpenCV/ffmpeg seek quickly; cap at 1080p (output is 1080x1920 anyway).
        "format": "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "%(title).80B [%(id)s].%(ext)s"),
        "writeinfojson": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return Path(info["requested_downloads"][0]["filepath"])


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", "-show_chapters", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    return json.loads(out)


def _fetch_sponsorblock(video_id: str) -> list[tuple[float, float, str]]:
    query = urllib.parse.urlencode({
        "videoID": video_id,
        "categories": json.dumps(SPONSORBLOCK_CATEGORIES),
    })
    try:
        with urllib.request.urlopen(f"https://sponsor.ajay.app/api/skipSegments?{query}", timeout=10) as r:
            data = json.load(r)
    except Exception:  # 404 = no segments; network errors are non-fatal
        return []
    return [(float(s["segment"][0]), float(s["segment"][1]), f"sponsorblock:{s['category']}") for s in data]


def load(path: Path, url: str | None = None, use_sponsorblock: bool = True) -> Source:
    probe = _ffprobe(path)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width, height = int(video["width"]), int(video["height"])
    rotation = 0
    for sd in video.get("side_data_list", []):
        rotation = int(sd.get("rotation", rotation))
    if abs(rotation) in (90, 270):
        width, height = height, width
    num, den = video.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0

    src = Source(
        path=path,
        title=probe["format"].get("tags", {}).get("title") or path.stem,
        duration=float(probe["format"]["duration"]),
        width=width,
        height=height,
        fps=fps or 30.0,
        has_audio=any(s["codec_type"] == "audio" for s in probe["streams"]),
        chapters=[Chapter(float(c["start_time"]), float(c["end_time"]), c.get("tags", {}).get("title", ""))
                  for c in probe.get("chapters", [])],
        url=url,
    )

    # yt-dlp sidecar (written by `download`, or by `yt-dlp --write-info-json`) has richer metadata.
    sidecar = path.with_suffix(".info.json")
    if sidecar.exists():
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        src.title = info.get("title") or src.title
        src.url = src.url or info.get("webpage_url")
        src.tags = info.get("tags") or []
        if info.get("chapters"):
            src.chapters = [Chapter(float(c["start_time"]), float(c["end_time"]), c.get("title", ""))
                            for c in info["chapters"]]
        if info.get("extractor_key") == "Youtube":
            src.video_id = info.get("id")

    if use_sponsorblock and src.video_id:
        src.skip_ranges = _fetch_sponsorblock(src.video_id)
    return src
