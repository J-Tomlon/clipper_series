"""Command line entry point: python -m clipper [FILE_OR_URL ...] [options]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import captions, platforms, reframe, render, scoring, segments, source, transcribe


def parse_time(s: str | None) -> float | None:
    """'90', '1:30', '01:02:03' -> seconds."""
    if not s:
        return None
    total = 0.0
    for part in s.split(":"):
        total = total * 60 + float(part)
    return total


def fmt(t: float) -> str:
    return f"{int(t // 3600)}:{int(t // 60 % 60):02d}:{t % 60:05.2f}" if t >= 3600 else f"{int(t // 60)}:{t % 60:05.2f}"


def slug(text: str, n: int = 50) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (s[:n].rstrip("-") or "video")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m clipper",
        description="Cut a long video into vertical, captioned clips for TikTok, Instagram Reels and YouTube Shorts.",
    )
    ap.add_argument("inputs", nargs="*", help="video files and/or URLs (default: every video in ./input)")
    ap.add_argument("--input-dir", default="input", type=Path)
    ap.add_argument("--output-dir", default="output", type=Path)
    ap.add_argument("--work-dir", default="work", type=Path, help="cache for transcripts and audio")
    ap.add_argument("--download-dir", default="downloads", type=Path)

    g = ap.add_argument_group("what to clip")
    g.add_argument("--mode", choices=["auto", "chapters", "full", "best"], default="auto",
                   help="auto: chapters if the video has them, else full. "
                        "chapters: one clip per chapter (best part of long chapters). "
                        "full: the whole video, back to back. "
                        "best: only the top --max-clips moments, ranked by --scorer.")
    g.add_argument("--scorer", choices=["auto", "ai", "heuristic"], default="auto",
                   help="ai = Hugging Face zero-shot model + audio; heuristic = audio loudness + speech pace. "
                        "auto uses ai when transformers/torch are installed")
    g.add_argument("--hf-model", default=scoring.DEFAULT_HF_MODEL)
    g.add_argument("--max-clips", type=int, help="cap the number of clips (best mode default: 5)")
    g.add_argument("--per-chapter", type=int, default=1, help="clips per long chapter in chapters mode")
    g.add_argument("--min", dest="min_len", type=float, default=15)
    g.add_argument("--target", dest="target_len", type=float, default=45)
    g.add_argument("--max", dest="max_len", type=float, default=60)
    g.add_argument("--from", dest="range_from", help="only consider video after this time (e.g. 12:30)")
    g.add_argument("--to", dest="range_to", help="only consider video before this time")
    g.add_argument("--skip-intro", default="auto", help="'auto' or seconds to drop from the start (0 to keep)")
    g.add_argument("--skip-outro", default="auto", help="'auto' or seconds to drop from the end (0 to keep)")
    g.add_argument("--no-sponsorblock", action="store_true", help="don't look up YouTube SponsorBlock segments")

    g = ap.add_argument_group("look")
    g.add_argument("--reframe", choices=["face", "center", "blur"], default="face",
                   help="face: crop follows the main face; center: center crop; blur: full frame over blurred fill")
    g.add_argument("--caption-style", choices=["highlight", "word", "none"], default="highlight",
                   help="highlight: 3-word lines with the spoken word highlighted; word: one word at a time")
    g.add_argument("--font", default="Arial Black")
    g.add_argument("--no-uppercase", action="store_true")

    g = ap.add_argument_group("output")
    g.add_argument("--platforms", default="tiktok,instagram,youtube")
    g.add_argument("--tags", default="", help="extra hashtags, comma separated, always included")
    g.add_argument("--encoder", choices=["libx264", "h264_nvenc"], default="libx264")
    g.add_argument("--dry-run", action="store_true", help="transcribe and print the clip plan without rendering")

    g = ap.add_argument_group("transcription")
    g.add_argument("--whisper-model", default="small", help="tiny/base/small/medium/large-v3/distil-large-v3")
    g.add_argument("--language", help="force a language code (default: auto-detect)")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return ap


def process(item: str, args) -> None:
    if source.is_url(item):
        print(f"\n== downloading {item}")
        path = source.download(item, args.download_dir)
        src = source.load(path, url=item, use_sponsorblock=not args.no_sponsorblock)
    else:
        src = source.load(Path(item), use_sponsorblock=not args.no_sponsorblock)
    print(f"\n== {src.title}  ({fmt(src.duration)}, {src.width}x{src.height}, {len(src.chapters)} chapters)")

    name = slug(src.title)
    work = args.work_dir / name
    out_root = args.output_dir / name
    work.mkdir(parents=True, exist_ok=True)

    span = (parse_time(args.range_from) or 0.0, min(parse_time(args.range_to) or src.duration, src.duration))
    if not src.has_audio:
        sys.exit("video has no audio track; nothing to transcribe")
    wav = transcribe.extract_audio(src.path, work / f"audio_{int(span[0])}_{int(span[1])}.wav", *span)
    print("  transcribing...")
    words = transcribe.transcribe(wav, work / f"transcript_{args.whisper_model}_{int(span[0])}_{int(span[1])}.json",
                                  span[0], args.whisper_model, args.language, args.device)
    sentences = transcribe.build_sentences(words)
    print(f"  {len(words)} words, {len(sentences)} sentences")

    cfg = segments.Settings(args.min_len, args.target_len, args.max_len, args.max_clips, args.per_chapter,
                            args.skip_intro, args.skip_outro)
    exclusions = segments.find_exclusions(src, sentences, cfg)
    for s, e, why in sorted(exclusions):
        print(f"  skipping {fmt(s)} - {fmt(e)}  [{why}]")

    kind = args.scorer
    if kind == "auto":
        kind = "ai" if scoring.ai_available() else "heuristic"
    elif kind == "ai" and not scoring.ai_available():
        sys.exit("--scorer ai needs: pip install -r requirements-ai.txt")
    scorer = scoring.Scorer(kind, wav, span[0], sentences, args.hf_model)
    mode, clips = segments.plan(args.mode, src, sentences, exclusions, cfg, span, scorer)
    print(f"  mode: {mode}  scorer: {kind}  ->  {len(clips)} clips")
    for i, c in enumerate(clips, 1):
        score = f"  score {c.score:.2f}" if c.score is not None else ""
        print(f"   {i:2d}. {fmt(c.start)} - {fmt(c.end)}  ({c.duration:4.1f}s){score}  {c.title}")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(src.path), "url": src.url, "title": src.title, "mode": mode, "scorer": kind,
                "excluded": [{"start": s, "end": e, "reason": r} for s, e, r in exclusions],
                "clips": [{"index": i, "start": c.start, "end": c.end, "title": c.title, "chapter": c.chapter,
                           "score": c.score, "text": c.text} for i, c in enumerate(clips, 1)]}
    (out_root / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run or not clips:
        return

    targets = [platforms.PLATFORMS[k.strip()] for k in args.platforms.split(",") if k.strip()]
    extra_tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for i, c in enumerate(clips, 1):
        center = None
        if args.reframe == "face":
            center = reframe.face_center_x(src.path, c.start, c.end, src.width)
        chain = reframe.video_filter(args.reframe, src.width, src.height, center)
        hook = segments.first_sentence(c.text, 150)
        base = f"{i:02d}_{slug(c.title, 40)}"
        for p in targets:
            dur = min(c.duration, p.max_duration)
            folder = out_root / p.key
            folder.mkdir(parents=True, exist_ok=True)
            ass = None
            if args.caption_style != "none":
                ass = captions.build_ass(words, c.start, c.start + dur, work / f"{base}_{p.key}.ass",
                                         style=args.caption_style, margin_v=p.caption_margin_v, font=args.font,
                                         upper=not args.no_uppercase)
            out = folder / f"{base}.mp4"
            print(f"  [{i}/{len(clips)}] {p.name:16s} -> {out}")
            render.render(src.path, c.start, dur, chain, ass, out, p, src.fps, src.has_audio, args.encoder)
            tags = platforms.hashtags(c.text, c.title, src.title, src.tags, extra_tags, p.max_hashtags)
            (folder / f"{base}.txt").write_text(platforms.caption_text(p, c.title, hook, tags, src.url),
                                                encoding="utf-8")
    print(f"  done -> {out_root}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    args = build_parser().parse_args(argv)
    items = args.inputs or [str(p) for p in source.find_inputs(args.input_dir)]
    if not items:
        print(f"Nothing to do: put videos in {args.input_dir}/ or pass a file/URL.")
        return 1
    for item in items:
        process(item, args)
    return 0
