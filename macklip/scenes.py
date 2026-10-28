"""Kamera kesmelerini bulur; klip sınırları bu kesmelere oturtulur."""
from bisect import bisect_left


def detect_cuts(video: str, show_progress: bool = True) -> list[float]:
    from scenedetect import ContentDetector, detect

    scenes = detect(video, ContentDetector(threshold=27.0), show_progress=show_progress)
    return [s[0].get_seconds() for s in scenes[1:]]


def snap(cuts: list[float], start: float, end: float, max_shift: float = 3.0) -> tuple[float, float]:
    """Başlangıcı geriye, bitişi ileriye doğru en yakın kesmeye (en fazla max_shift sn) kaydırır."""
    if not cuts:
        return start, end
    i = bisect_left(cuts, start)
    before = [c for c in cuts[max(0, i - 3):i + 1] if start - max_shift <= c <= start]
    j = bisect_left(cuts, end)
    after = [c for c in cuts[j:j + 3] if end <= c <= end + max_shift]
    return (max(before) if before else start), (min(after) if after else end)
