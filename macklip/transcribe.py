"""Spiker anlatımını kelime bazında zaman damgalı metne çevirir."""
import json
import re
from pathlib import Path

from .models import Word


def transcribe(video: str, language: str = "tr", model_size: str = "small", device: str = "auto",
               progress=None) -> list[Word]:
    """progress verilirse 0-1 arası ilerleme ile çağrılır; verilmezse konsola yazılır."""
    from faster_whisper import WhisperModel

    compute_type = "int8" if device in ("cpu", "auto") else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        video,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(w.start, w.end, w.word.strip()))
        if progress:
            progress(min(1.0, seg.end / max(info.duration, 1)))
        else:
            print(f"\r  anlatım: {seg.end / 60:5.1f}. dakika", end="", flush=True)
    if not progress:
        print()
    return words


def save_words(words: list[Word], path: Path) -> None:
    path.write_text(json.dumps([w.__dict__ for w in words], ensure_ascii=False, indent=0), encoding="utf-8")


def load_words(path: Path) -> list[Word]:
    """Daha önce kaydedilmiş bir JSON ya da hazır bir SRT altyazı dosyasını okur."""
    if path.suffix.lower() == ".srt":
        return _load_srt(path)
    return [Word(**w) for w in json.loads(path.read_text(encoding="utf-8"))]


def _srt_time(s: str) -> float:
    h, m, rest = s.strip().split(":")
    sec, ms = rest.replace(".", ",").split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def _load_srt(path: Path) -> list[Word]:
    # SRT'de kelime zamanı yok; her satırın süresini kelimelere eşit dağıtıyoruz.
    words = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    for block in blocks:
        lines = block.strip().splitlines()
        timing = next((l for l in lines if "-->" in l), None)
        if not timing:
            continue
        a, b = timing.split("-->")
        start, end = _srt_time(a), _srt_time(b)
        text = " ".join(lines[lines.index(timing) + 1:])
        tokens = text.split()
        step = (end - start) / max(len(tokens), 1)
        for i, tok in enumerate(tokens):
            words.append(Word(start + i * step, start + (i + 1) * step, tok))
    return words
