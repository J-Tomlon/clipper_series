"""ffmpeg encode: trim + reframe + burned captions + loudness normalisation, per platform."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .platforms import Platform


def render(src: Path, start: float, duration: float, reframe_chain: str, ass_file: Path | None,
           out: Path, p: Platform, src_fps: float, has_audio: bool, encoder: str = "libx264") -> None:
    fps = min(src_fps, p.fps_cap)
    chain = reframe_chain + f";[rf]fps={fps:.3f}"
    if ass_file is not None:
        # Relative filename + cwd avoids ffmpeg filter escaping of Windows drive letters.
        chain += f",ass={ass_file.name}"
    chain += "[v]"
    if has_audio:
        chain += ";[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]"

    if encoder == "h264_nvenc":
        venc = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(p.crf + 3), "-b:v", "0"]
    else:
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", str(p.crf)]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src.resolve()),
        "-filter_complex", chain, "-map", "[v]",
        *venc, "-maxrate", p.maxrate, "-bufsize", f"{int(p.maxrate[:-1]) * 2}M",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", p.audio_bitrate, "-ar", "48000"]
    cmd += ["-map_chapters", "-1", "-map_metadata", "-1", "-movflags", "+faststart", str(out.resolve())]
    subprocess.run(cmd, check=True, cwd=ass_file.parent if ass_file else None)
