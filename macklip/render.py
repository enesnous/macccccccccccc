"""Klipleri FFmpeg ile keser, üstüne başlık yazar ve tek videoda birleştirir."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import Clip

WIDTH, HEIGHT, FPS = 1280, 720, 25


def has_audio(path: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", path],
        capture_output=True, text=True,
    )
    return bool(json.loads(out.stdout or "{}").get("streams"))


def duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return float(out.stdout.strip() or 0)


def _cut(clip: Clip, out: Path, audio: bool) -> None:
    vf = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}"
    )
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{clip.start:.2f}", "-t", f"{clip.duration:.2f}", "-i", clip.source]
    if not audio:
        cmd += ["-f", "lavfi", "-t", f"{clip.duration:.2f}", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-vf", vf, "-map", "0:v:0", "-map", "0:a:0" if audio else "1:a:0"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k", "-shortest", str(out)]
    subprocess.run(cmd, check=True)


def render(clips: list[Clip], player: str, output: Path, progress=None) -> None:
    """Klipleri maçtaki sırasıyla (zamana göre) keser ve birleştirir; üzerine yazı eklenmez."""
    clips = sorted(clips, key=lambda c: (c.source, c.start))
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg bulunamadı; lütfen kurun (https://ffmpeg.org)")
    audio_cache: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="macklip_") as tmp:
        parts = []
        for n, clip in enumerate(clips, 1):
            if clip.source not in audio_cache:
                audio_cache[clip.source] = has_audio(clip.source)
            part = Path(tmp) / f"{n:04d}.mp4"
            if progress:
                progress((n - 1) / len(clips))
            else:
                print(f"\r  kesiliyor: {n}/{len(clips)}", end="", flush=True)
            _cut(clip, part, audio_cache[clip.source])
            parts.append(part)
        if not progress:
            print()
        listfile = Path(tmp) / "list.txt"
        listfile.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listfile),
             "-c", "copy", "-movflags", "+faststart", str(output)],
            check=True,
        )
