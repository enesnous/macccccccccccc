"""Farklı kaynaklardan gelen klipleri birleştirir, çakışanları tekleştirir."""
from .models import CATEGORY_ORDER, Clip


def overlap(a: Clip, b: Clip) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _combine(a: Clip, b: Clip) -> Clip:
    # En önemli kategori kazanır; farklı sinyaller (örn. anlatım + yüz) aynı anı gösteriyorsa güven artar.
    winner = a if a.priority() <= b.priority() else b
    corroborated = a.category != b.category or a.reasons[0][:4] != b.reasons[0][:4]
    return Clip(
        a.source,
        min(a.start, b.start),
        max(a.end, b.end),
        winner.category,
        min(0.99, max(a.confidence, b.confidence) + (0.1 if corroborated else 0)),
        a.reasons + b.reasons,
    )


def merge_clips(clips: list[Clip], gap: float = 1.0) -> list[Clip]:
    """Aynı türdeki ya da büyük ölçüde örtüşen klipleri birleştirir; farklı türde olup
    sadece kenarlarından değen klipleri ayrı bırakıp örtüşen kısmı ortadan böler."""
    by_source: dict[str, list[Clip]] = {}
    for c in clips:
        by_source.setdefault(c.source, []).append(c)

    merged = []
    for source_clips in by_source.values():
        source_clips.sort(key=lambda c: c.start)
        out: list[Clip] = []
        for c in source_clips:
            if not out:
                out.append(c)
                continue
            prev = out[-1]
            shared = overlap(prev, c)
            if shared >= 0.5 * min(prev.duration, c.duration):
                out[-1] = _combine(prev, c)
            elif prev.category == c.category and c.start <= prev.end + gap:
                out[-1] = _combine(prev, c)
            elif shared > 0:
                cut = (c.start + prev.end) / 2
                prev.end = c.start = cut
                out.append(c)
            else:
                out.append(c)
        merged += out
    return merged


def order_clips(clips: list[Clip], mode: str, sources: list[str]) -> list[Clip]:
    if mode == "zaman":
        return sorted(clips, key=lambda c: (sources.index(c.source), c.start))
    return sorted(clips, key=lambda c: (CATEGORY_ORDER.index(c.category), sources.index(c.source), c.start))
