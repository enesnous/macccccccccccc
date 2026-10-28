"""Komut satırı ve web arayüzünün ortak kullandığı analiz akışı."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import commentary, merge, render, scenes, transcribe
from .models import Clip

# Her aşamanın toplam ilerleme çubuğundaki payı (kabaca süreleriyle orantılı).
STAGES = [
    ("anlatim", "Spiker anlatımı yazıya dökülüyor", 0.45),
    ("yuz", "Yüz taraması yapılıyor", 0.30),
    ("sahne", "Kamera geçişleri bulunuyor", 0.10),
    ("video", "Klipler kesilip birleştiriliyor", 0.15),
]

Progress = Callable[[str, float], None]  # (aşama adı, 0-1 arası toplam ilerleme)


@dataclass
class Options:
    match: str
    player: str
    aliases: list[str] = field(default_factory=list)
    photos: list[str] = field(default_factory=list)
    training: list[str] = field(default_factory=list)
    extra_videos: list[str] = field(default_factory=list)
    language: str = "tr"
    model: str = "small"
    transcript: str | None = None  # hazır .srt / .json
    transcript_cache: str | None = None
    step: float = 1.0
    order: str = "kategori"
    excluded: set[str] = field(default_factory=set)
    min_confidence: float = 0.0
    detect_scenes: bool = True


def _stage(progress: Progress | None, key: str, stages=STAGES) -> Callable[[float], None] | None:
    """Bir aşamanın kendi 0-1 ilerlemesini toplam ilerlemeye çeviren geri çağrı döndürür."""
    if progress is None:
        return None
    before = 0.0
    for k, label, weight in stages:
        if k == key:
            return lambda f, label=label, before=before, weight=weight: progress(label, before + weight * f)
        before += weight
    raise KeyError(key)


def _log(progress: Progress | None, message: str) -> None:
    if progress is None:
        print(message)


def _words(match: str, o, progress: Progress | None, step) -> list:
    if step:
        step(0)
    if o.transcript:
        return transcribe.load_words(Path(o.transcript))
    if o.transcript_cache and Path(o.transcript_cache).exists():
        _log(progress, f"  önbellekten okunuyor: {Path(o.transcript_cache).name}")
        return transcribe.load_words(Path(o.transcript_cache))
    words = transcribe.transcribe(match, o.language, o.model, progress=step)
    if o.transcript_cache:
        transcribe.save_words(words, Path(o.transcript_cache))
    return words


def _fit_to_video(clips: list[Clip], match: str, cuts: list[float]) -> None:
    durations: dict[str, float] = {}
    for c in clips:
        if c.source not in durations:
            durations[c.source] = render.duration(c.source)
        if c.source == match:
            c.start, c.end = scenes.snap(cuts, c.start, c.end)
        c.end = min(c.end, durations[c.source])


def find_clips(o: Options, progress: Progress | None = None) -> list[Clip]:
    """Tek bir oyuncunun kliplerini bulur (komut satırı akışı)."""
    match = str(Path(o.match).resolve())
    clips: list[Clip] = []

    # 1) Spiker anlatımı
    _log(progress, "1/4 Anlatım çözümleniyor...")
    words = _words(match, o, progress, _stage(progress, "anlatim"))
    found = commentary.clips_from_commentary(words, o.player, o.aliases, match)
    _log(progress, f"  anlatımda {len(found)} an bulundu")
    clips += found

    # 2) Yüz tanıma
    _log(progress, "2/4 Yüz taraması...")
    step = _stage(progress, "yuz")
    videos = [(match, False)] + [(v, False) for v in o.extra_videos] + [(v, True) for v in o.training]
    refs = []
    if o.photos:
        from . import faces

        refs = faces.load_references(o.photos)
        if not refs:
            _log(progress, "  fotoğraflarda yüz bulunamadı, atlanıyor")
    for n, (path, training) in enumerate(videos):
        path = str(Path(path).resolve())
        if refs:
            _log(progress, f"  {Path(path).name}")
            width = faces.TRAINING_ANALYSIS_WIDTH if training else faces.MATCH_ANALYSIS_WIDTH
            sub = (lambda f, n=n: step((n + f) / len(videos))) if step else None
            hits = faces.scan_video(path, refs, o.step, width, progress=sub)
            face_clips = faces.clips_from_faces(hits, path, o.step, training)
        else:
            face_clips = []
        if training and not face_clips:
            _log(progress, "  yüz bulunamadı; antrenman videosunun tamamı ekleniyor")
            face_clips = [Clip(path, 0, render.duration(path), "antrenman", 0.5, ["antrenman videosu"])]
        clips += face_clips

    # 3) Kamera kesmelerine oturt, birleştir
    _log(progress, "3/4 Klipler düzenleniyor...")
    step = _stage(progress, "sahne")
    if step:
        step(0)
    cuts: list[float] = scenes.detect_cuts(match, progress is None) if o.detect_scenes and clips else []
    _fit_to_video(clips, match, cuts)

    clips = merge.merge_clips(clips)
    clips = [c for c in clips if c.category not in o.excluded and c.confidence >= o.min_confidence and c.duration > 1]
    sources = [match] + [str(Path(v).resolve()) for v in o.extra_videos + o.training]
    return merge.order_clips(clips, o.order, sources)


def make_video(clips: list[Clip], player: str, output: Path, progress: Progress | None = None) -> None:
    _log(progress, "4/4 Video hazırlanıyor...")
    render.render(clips, player, output, progress=_stage(progress, "video"))
    if progress:
        progress("Hazır", 1.0)


# ---------------------------------------------------------------------------
# Tüm maç: oyuncu adı vermeden, maçta adı geçen herkes için klipler
# ---------------------------------------------------------------------------

MATCH_STAGES = [
    ("anlatim", "Spiker anlatımı yazıya dökülüyor", 0.55),
    ("yuz", "Yüzler bulunup gruplanıyor", 0.30),
    ("sahne", "Kamera geçişleri bulunuyor", 0.10),
    ("oyuncu", "Oyuncular ve klipler eşleştiriliyor", 0.05),
]


@dataclass
class MatchOptions:
    match: str
    language: str = "tr"
    model: str = "small"
    transcript: str | None = None
    transcript_cache: str | None = None
    squad: list[str] = field(default_factory=list)  # boşsa oyuncular anlatımdan bulunur
    step: float = 1.0
    min_mentions: int = 2
    use_faces: bool = True
    detect_scenes: bool = True


@dataclass
class PlayerResult:
    name: str
    aliases: list[str]
    mentions: int
    clips: list[Clip]
    photo: object = None  # numpy BGR yüz görüntüsü ya da None


def _assign_faces(clusters, mention_times: dict[str, list[float]], step: float) -> dict[int, str]:
    """Her yüz grubunu, o yüz ekrandayken (ya da hemen önce) adı en çok geçen oyuncuya bağlar."""
    from .faces import group_hits

    owners = {}
    for n, cluster in enumerate(clusters):
        segments = group_hits(cluster.hits, step)
        scores = {}
        for name, times in mention_times.items():
            scores[name] = sum(
                any(seg[0].time - 15 <= t <= seg[-1].time + 5 for t in times) for seg in segments
            )
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] == 0:
            continue
        best, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if score > runner_up and score >= 0.3 * len(segments):
            owners[n] = best
    return owners


def analyze_match(o: MatchOptions, progress: Progress | None = None) -> list[PlayerResult]:
    match = str(Path(o.match).resolve())

    _log(progress, "1/4 Anlatım çözümleniyor...")
    words = _words(match, o, progress, _stage(progress, "anlatim", MATCH_STAGES))
    players = commentary.discover_players(words, o.squad or None, o.min_mentions)
    _log(progress, f"  anlatımda {len(players)} oyuncu bulundu")

    _log(progress, "2/4 Yüzler gruplanıyor...")
    step = _stage(progress, "yuz", MATCH_STAGES)
    clusters = []
    if o.use_faces and players:
        from . import faces

        clusters = faces.cluster_faces(match, o.step, progress=step)
        _log(progress, f"  {len(clusters)} farklı yüz bulundu")

    _log(progress, "3/4 Kamera geçişleri...")
    step = _stage(progress, "sahne", MATCH_STAGES)
    if step:
        step(0)
    cuts = scenes.detect_cuts(match, progress is None) if o.detect_scenes and players else []

    _log(progress, "4/4 Oyuncular eşleştiriliyor...")
    step = _stage(progress, "oyuncu", MATCH_STAGES)
    if step:
        step(0)
    mention_times = {
        p.name: [m.time for m in commentary.find_mentions(words, commentary.player_names(p.name, p.aliases))]
        for p in players
    }
    owners = _assign_faces(clusters, mention_times, o.step)

    results = []
    for p in players:
        clips = commentary.clips_from_commentary(words, p.name, p.aliases, match)
        mine = [c for n, c in enumerate(clusters) if owners.get(n) == p.name]
        photo = max(mine, key=lambda c: c.crop_size).crop if mine else None
        if mine:
            from . import faces

            hits = sorted((h for c in mine for h in c.hits), key=lambda h: h.time)
            clips += faces.clips_from_faces(hits, match, o.step)
        _fit_to_video(clips, match, cuts)
        clips = merge.order_clips([c for c in merge.merge_clips(clips) if c.duration > 1], "kategori", [match])
        if clips:
            results.append(PlayerResult(p.name, p.aliases, p.mentions, clips, photo))

    # Önce gol/asist gibi önemli anları olan oyuncular, sonra adı en çok geçenler
    def importance(r: PlayerResult):
        return (-sum(c.category in ("gol", "asist") for c in r.clips), -len(r.clips), -r.mentions)

    return sorted(results, key=importance)
