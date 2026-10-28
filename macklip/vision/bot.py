"""Görsel ölçümlerden sahneleri sınıflandırır, olayları bulur ve kişilere dağıtır."""
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from ..merge import merge_clips, order_clips
from ..models import Clip
from .scan import ScanResult, scan

# ---------------------------------------------------------------------------
# Sahneler
# ---------------------------------------------------------------------------


@dataclass
class Shot:
    start: float
    end: float
    kind: str = "diğer"  # oyun | yakın çekim | tekrar | röportaj | antrenman | tribün | diğer
    faces: Counter = field(default_factory=Counter)  # yüz grubu -> kaç örnekte göründü
    big_face: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def _segment(r: ScanResult) -> list[Shot]:
    """Kamera geçişlerinden sahneleri çıkarır ve her sahneyi görüntüsüne göre sınıflandırır."""
    if not r.frames:
        return []
    step = r.frames[1].t - r.frames[0].t if len(r.frames) > 1 else 0.5
    groups, current = [], []
    for f in r.frames:
        # Tabelanın kaybolması/geri gelmesi de geçiştir (tekrarlar canlı yayınla aynı renkte olabilir)
        board_flip = current and f.scoreboard != current[-1].scoreboard
        if (f.cut or board_flip) and current:
            groups.append(current)
            current = []
        current.append(f)
    groups.append(current)

    on_air = [f.t for f in r.frames if f.scoreboard]
    match_start, match_end = (min(on_air), max(on_air)) if len(on_air) > 20 else (0.0, r.duration)

    shots = []
    for fs in groups:
        shot = Shot(fs[0].t, fs[-1].t + step)
        n = len(fs)
        green = sum(f.green for f in fs) / n
        board = sum(f.scoreboard for f in fs) / n
        motion = sum(f.motion for f in fs) / n
        texture = sum(f.edge_density for f in fs) / n
        with_face = [f for f in fs if f.face_sizes]
        face_share = len(with_face) / max(1, sum(f.face_checked for f in fs))
        shot.big_face = max((max(f.face_sizes) for f in with_face), default=0.0)
        for f in fs:
            shot.faces.update(f.face_ids)
        in_match = match_start - 5 <= shot.start <= match_end + 5

        if face_share >= 0.5 and shot.big_face >= 0.12 and board < 0.3 and motion < 5 and shot.duration >= 6:
            shot.kind = "röportaj"
        elif green >= 0.3 and board >= 0.5:
            shot.kind = "yakın çekim" if shot.big_face >= 0.12 else "oyun"
        elif green >= 0.3:
            shot.kind = "tekrar" if in_match else "antrenman"
        elif face_share >= 0.3 and shot.big_face >= 0.1:
            shot.kind = "yakın çekim"
        elif texture > 0.12 and green < 0.2:
            shot.kind = "tribün"
        shots.append(shot)

    # Çok kısa sahneleri (geçiş efektleri) komşusuna kat
    merged: list[Shot] = []
    for s in shots:
        if merged and (s.duration < 1.0 or s.kind == merged[-1].kind):
            prev = merged[-1]
            prev.end = s.end
            prev.faces.update(s.faces)
            prev.big_face = max(prev.big_face, s.big_face)
        else:
            merged.append(s)
    return merged


# ---------------------------------------------------------------------------
# Ekran yazılarından isimler
# ---------------------------------------------------------------------------

NOT_NAMES = {
    "GOL", "GOAL", "SARI", "KIRMIZI", "KART", "YELLOW", "RED", "CARD", "OYUNCU", "DEGISIKLIGI", "DEGISIKLIK",
    "SUBSTITUTION", "SUB", "IN", "OUT", "MAC", "SONU", "DEVRE", "ARASI", "CANLI", "LIVE", "REPLAY", "TEKRAR",
    "HALF", "TIME", "FULL", "FT", "HT", "VAR", "PENALTI", "PENALTY", "KORNER", "CORNER", "OFSAYT", "OFFSIDE",
    "UZATMA", "DAKIKA", "MIN", "MINUTE", "SKOR", "SCORE", "TEKNIK", "DIREKTOR", "HAKEM", "REFEREE", "KAPTAN",
}
SUB_WORDS = ("DEGISIKLIK", "SUBSTITUTION", "GIREN", "CIKAN", "OYUNCU DEG")
GOAL_WORDS = ("GOL", "GOAL")
CARD_WORDS = ("SARI KART", "KIRMIZI KART", "YELLOW CARD", "RED CARD")


def fold(text: str) -> str:
    """Büyük harfe çevirip Türkçe karakterleri sadeleştirir: 'Güler' -> 'GULER'."""
    text = text.replace("İ", "I").replace("ı", "i")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9 ]+", " ", text.upper()).strip()


def _name_like(text: str) -> str | None:
    words = [w for w in fold(text).split() if not w.isdigit()]
    words = [w for w in words if w not in NOT_NAMES]
    if not 2 <= len(words) <= 3 or not all(2 <= len(w) <= 14 and w.isalpha() for w in words):
        return None
    return " ".join(words)


@dataclass
class NameSeen:
    t: float
    key: str  # sadeleştirilmiş ad
    display: str


def _names(r: ScanResult, squad: list[str]) -> list[NameSeen]:
    seen = []
    squad_keys = [(fold(s), s) for s in squad]
    for x in r.texts:
        if x.region != "alt":
            continue
        if squad_keys:
            folded = fold(x.text)
            best = max(squad_keys, key=lambda k: fuzz.partial_ratio(k[0], folded))
            # Soyadı tek başına yazılmış olabilir ("GULER 10"): soyadı eşleşmesi de yeter
            surname = best[0].split()[-1]
            if fuzz.partial_ratio(best[0], folded) >= 85 or (len(surname) >= 4 and surname in folded.split()):
                seen.append(NameSeen(x.t, best[0], best[1]))
            continue
        key = _name_like(x.text)
        if key:
            seen.append(NameSeen(x.t, key, key.title()))
    if squad_keys:
        return seen
    # Reklam/sponsor yazıları maç boyunca defalarca görünür; isimler ise birkaç kez.
    counts = Counter(n.key for n in seen)
    spread = defaultdict(list)
    for n in seen:
        spread[n.key].append(n.t)
    return [n for n in seen if counts[n.key] <= 30 and (max(spread[n.key]) - min(spread[n.key]) < 2400 or counts[n.key] <= 6)]


def _texts_between(r: ScanResult, a: float, b: float) -> str:
    return " ".join(fold(x.text) for x in r.texts if a <= x.t <= b and x.region == "alt")


# ---------------------------------------------------------------------------
# Olaylar
# ---------------------------------------------------------------------------


@dataclass
class Event:
    category: str
    start: float
    end: float
    anchor: float
    confidence: float
    reason: str
    who: dict[str, float] = field(default_factory=dict)  # kişi anahtarı -> ağırlık


def _stable_scores(scores):
    """OCR hatalarını ele: aynı skor arka arkaya en az iki kez okunmalı."""
    stable = []
    for (t1, s1), (_, s2) in zip(scores, scores[1:]):
        if s1 == s2 and (not stable or stable[-1][1] != s1):
            stable.append((t1, s1))
    return stable


def _break_run(shots: list[Shot], t: float, back: float = 150.0) -> tuple[float, float]:
    """t'den geriye bakıp oyunun kesildiği en uzun bölümü (yakın çekim/tekrar/tribün) bulur.
    Dönen (başlangıç, bitiş): başlangıç yaklaşık olarak olayın olduğu andır."""
    runs, cur = [], None
    for s in shots:
        if s.end < t - back or s.start > t:
            continue
        if s.kind == "oyun":
            if cur:
                runs.append(cur)
                cur = None
        else:
            cur = [s.start, s.end] if cur is None else [cur[0], s.end]
    if cur:
        runs.append(cur)
    if not runs:
        return t - 20, t
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return a, min(b, t)


def _goals(r: ScanResult, shots: list[Shot]) -> list[Event]:
    events = []
    stable = _stable_scores(r.scores)
    for (_, before), (t_seen, after) in zip(stable, stable[1:]):
        diff = (after[0] - before[0], after[1] - before[1])
        if not ((diff[0] >= 1 and diff[1] == 0) or (diff[1] >= 1 and diff[0] == 0)):
            continue  # skor düştüyse ya da iki taraf birden değiştiyse OCR hatasıdır
        moment, run_end = _break_run(shots, t_seen)
        start, end = max(0.0, moment - 15), min(run_end + 3, moment + 75)
        side = "ev sahibi" if diff[0] else "deplasman"
        events.append(Event("gol", start, end, moment, 0.9,
                            f"skor tabelası {before[0]}-{before[1]} → {after[0]}-{after[1]} ({side})"))
        # Golden hemen sonraki yakın çekimler: gol sevinci
        for s in shots:
            if moment <= s.start < run_end and s.kind == "yakın çekim":
                events.append(Event("gol sevinci", s.start, s.end, s.start, 0.75, "gol sonrası yakın çekim"))
    return events


def _highlights(shots: list[Shot], goals: list[Event]) -> list[Event]:
    """Yayıncı bir anı tekrar gösterdiyse o an önemlidir: şut, kurtarış, faul, tartışmalı pozisyon."""
    events = []
    used = [(g.start, g.end) for g in goals if g.category == "gol"]
    for i, s in enumerate(shots):
        if s.kind != "tekrar" or s.duration < 2.5:
            continue
        if any(a <= s.start <= b for a, b in used):
            continue
        moment, run_end = _break_run(shots, s.end, back=60)
        start, end = max(0.0, moment - 12), min(run_end + 2, moment + 60)
        if any(a <= moment <= b for a, b in used):
            continue
        used.append((start, end))
        events.append(Event("pozisyon", start, end, moment, 0.6, "yayıncı bu anı tekrar gösterdi"))
    return events


def _cards(r: ScanResult, names: list[NameSeen]) -> list[Event]:
    events = []
    by_color = defaultdict(list)
    for c in r.cards:
        by_color[c.color].append(c.t)
    for color, times in by_color.items():
        times.sort()
        i = 0
        while i < len(times):
            j = i
            while j + 1 < len(times) and times[j + 1] - times[j] <= 1.5:
                j += 1
            if j > i:  # en az iki örnekte görülmeli (kart birkaç saniye havada tutulur)
                t = times[i]
                events.append(Event("kart", max(0.0, t - 20), t + 8, t, 0.55, f"{color} kart görüldü"))
            i = j + 1
    # Ekrana "SARI KART" gibi yazı geldiyse
    for x in r.texts:
        f = fold(x.text)
        if any(w in f for w in CARD_WORDS) and not any(abs(e.anchor - x.t) < 30 for e in events):
            events.append(Event("kart", max(0.0, x.t - 25), x.t + 5, x.t, 0.8, f"ekran yazısı: {x.text}"))
    return events


def _subs(r: ScanResult) -> list[Event]:
    events = []
    for x in r.texts:
        if any(w in fold(x.text) for w in SUB_WORDS) and not any(abs(e.anchor - x.t) < 20 for e in events):
            events.append(Event("oyuncu değişikliği", max(0.0, x.t - 5), x.t + 12, x.t, 0.85,
                                f"ekran yazısı: {x.text}"))
    return events


# ---------------------------------------------------------------------------
# Kişiler
# ---------------------------------------------------------------------------


@dataclass
class Person:
    key: str
    name: str
    named: bool
    face_ids: set[int]
    photo: object = None
    screen_time: float = 0.0


@dataclass
class BotResult:
    people: list[tuple[Person, list[Clip]]]
    categories: dict[str, list[Clip]]
    goals: list[str]


def _people(r: ScanResult, names: list[NameSeen], step: float) -> list[Person]:
    """Yüz gruplarını ekrandaki isim yazılarıyla eşleştirir. İsim bulunamayan yüzler de kişi sayılır."""
    face_name = {}
    for g in r.faces:
        votes = Counter()
        for n in names:
            # İsim bandı geldiğinde ekrandaki yüz büyük olasılıkla o kişidir
            if any(abs(t - n.t) <= 2.5 for t in g.times):
                votes[n.key] += 1
        if votes:
            (key, v), *rest = votes.most_common()
            if not rest or v > rest[0][1]:
                face_name[g.id] = key

    display = {n.key: n.display for n in names}
    people: dict[str, Person] = {}
    for g in r.faces:
        if g.id in face_name:
            key = face_name[g.id]
            p = people.setdefault(key, Person(key, display[key], True, set()))
        elif len(g.times) >= 4:  # isimsiz ama ekranda yeterince görünen yüz
            key = f"yuz{g.id}"
            p = people.setdefault(key, Person(key, "", False, set()))
        else:
            continue
        p.face_ids.add(g.id)
        p.screen_time += len(g.times) * step
        if g.crop is not None and (p.photo is None or g.crop_size > getattr(p, "_crop_size", 0)):
            p.photo, p._crop_size = g.crop, g.crop_size
    # Yüzü hiç bulunamasa da adı ekranda geçenler (ör. sadece gol yazısında)
    for n in names:
        people.setdefault(n.key, Person(n.key, n.display, True, set()))

    unnamed = sorted((p for p in people.values() if not p.named), key=lambda p: -p.screen_time)
    for i, p in enumerate(unnamed, 1):
        p.name = f"İsimsiz kişi {i}"
    return list(people.values())


def _who_in(r: ScanResult, face_to_person: dict[int, str], a: float, b: float) -> Counter:
    c = Counter()
    for f in r.frames:
        if a <= f.t <= b:
            for fid, size in zip(f.face_ids, f.face_sizes):
                if fid in face_to_person:
                    c[face_to_person[fid]] += size  # büyük (yakın) yüz daha çok sayılır
    return c


def _names_in(names: list[NameSeen], a: float, b: float) -> Counter:
    return Counter(n.key for n in names if a <= n.t <= b)


def analyze(path: str, squad: list[str] | None = None, progress=None) -> BotResult:
    r = scan(path, progress=(lambda f: progress("Görüntü inceleniyor", 0.95 * f)) if progress else None)
    if progress:
        progress("Sahneler ve olaylar çıkarılıyor", 0.96)
    step = r.frames[1].t - r.frames[0].t if len(r.frames) > 1 else 0.5
    shots = _segment(r)
    names = _names(r, squad or [])
    people = _people(r, names, step)
    face_to_person = {fid: p.key for p in people for fid in p.face_ids}

    events = _goals(r, shots)
    events += _highlights(shots, events)
    events += _cards(r, names)
    events += _subs(r)

    # Olaylara kişi bağla: önce ekrandaki isim yazısı, yoksa o anlarda en çok yakın çekimi gelen yüz
    for e in events:
        window_names = _names_in(names, e.anchor - 5, e.end + (45 if e.category == "gol" else 5))
        window_faces = _who_in(r, face_to_person, e.anchor, e.end)
        if e.category == "oyuncu değişikliği":
            for key in _names_in(names, e.anchor - 2, e.end):
                e.who[key] = 1.0
        elif window_names:
            key, _ = window_names.most_common(1)[0]
            e.who[key] = 1.0
        elif window_faces:
            key, _ = window_faces.most_common(1)[0]
            e.who[key] = 0.6

    # Sahne türüne göre klipler (yakın çekim, röportaj, antrenman, tribün)
    shot_clips: list[tuple[Clip, Counter]] = []
    goal_windows = [(e.anchor, e.end) for e in events if e.category == "gol"]
    for s in shots:
        if s.kind not in ("yakın çekim", "röportaj", "antrenman", "tribün") or s.duration < 2:
            continue
        if s.kind == "yakın çekim" and any(a <= s.start <= b for a, b in goal_windows):
            continue  # gol sevinci olarak zaten var
        conf = 0.7 if s.kind != "tribün" else 0.6
        clip = Clip(path, max(0.0, s.start - 0.5), s.end + 0.5, s.kind, conf, [f"görüntü: {s.kind} sahnesi"])
        persons = Counter()
        for fid, n in s.faces.items():
            if fid in face_to_person:
                persons[face_to_person[fid]] += n
        if s.kind == "röportaj":
            for key in _names_in(names, s.start, s.end):
                persons[key] += 5  # röportajda isim bandı çok güçlü bir ipucu
        shot_clips.append((clip, persons))

    # Kategori görünümü: hepsi
    categories: dict[str, list[Clip]] = defaultdict(list)
    for e in events:
        categories[e.category].append(Clip(path, e.start, e.end, e.category, e.confidence, [e.reason]))
    for clip, _ in shot_clips:
        categories[clip.category].append(clip)
    categories = {k: sorted(merge_clips(v), key=lambda c: c.start) for k, v in categories.items()}

    # Kişi görünümü
    per_person: dict[str, list[Clip]] = defaultdict(list)
    for e in events:
        for key, weight in e.who.items():
            per_person[key].append(Clip(path, e.start, e.end, e.category, round(e.confidence * weight, 2),
                                        [e.reason]))
    for clip, persons in shot_clips:
        for key, n in persons.items():
            if n >= 2 or clip.category == "röportaj":
                per_person[key].append(Clip(path, clip.start, clip.end, clip.category, clip.confidence, clip.reasons))

    people_out = []
    for p in people:
        clips = order_clips([c for c in merge_clips(per_person.get(p.key, [])) if c.duration > 1], "zaman", [path])
        if clips:
            people_out.append((p, clips))

    def importance(item):
        p, clips = item
        cats = Counter(c.category for c in clips)
        return (not p.named, -cats["gol"], -cats["pozisyon"], -len(clips), -p.screen_time)

    people_out.sort(key=importance)
    goals = [e.reason for e in events if e.category == "gol"]
    if progress:
        progress("Hazır", 1.0)
    return BotResult(people_out, categories, goals)
