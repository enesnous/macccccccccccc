"""Spiker anlatımından oyuncunun adının geçtiği anları ve olay türünü çıkarır."""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import Clip, Word

# Kelimeler normalize edildikten sonra (küçük harf, noktalama yok) regex ile aranır.
EVENT_PATTERNS = {
    "gol": [r"^go+l+(ü|u)?$", r"^go{2,}", r"^ağlar", r"^filele", r"^goal"],
    "asist": [r"^asist", r"^pas(ı|ıyla|ında|la|ını)?$", r"^ortası", r"^orta(yı|yla)$", r"^assist", r"^pass"],
    "çalım": [r"^çalım", r"^dripling", r"^driblin", r"^geçti$", r"^geçiyor$", r"^ekarte", r"^dribbl", r"^skill"],
    "şut": [r"^şut", r"^vurdu", r"^vuruş", r"^direk", r"^kurtardı", r"^kurtarış", r"^shot", r"^shoots?$"],
    "faul": [r"^faul", r"^düşürdü", r"^düşürül", r"^yerde$", r"^kart", r"^sarıyı", r"^kırmızıyı", r"^penaltı",
             r"^serbest", r"^foul", r"^yellow", r"^tackle"],
}

# Olay kelimesi oyuncu adına göre hangi pencerede aranır (saniye).
LOOK_BEFORE = 3.0
LOOK_AFTER = 8.0
GOAL_LOOK_AFTER = 10.0

# Her olay türü için klibin olay anından ne kadar önce başlayıp ne kadar sonra biteceği.
CLIP_WINDOWS = {
    "gol": (12.0, 20.0),
    "asist": (12.0, 18.0),
    "çalım": (6.0, 6.0),
    "şut": (7.0, 8.0),
    "faul": (5.0, 10.0),
    "aksiyon": (5.0, 5.0),
}

_EVENT_REGEXES = [re.compile(p) for pats in EVENT_PATTERNS.values() for p in pats]

CONFIDENCE = {"gol": 0.9, "asist": 0.75, "çalım": 0.6, "şut": 0.7, "faul": 0.7, "aksiyon": 0.4}

_TR_LOWER = str.maketrans({"İ": "i", "I": "ı"})


def normalize(text: str) -> str:
    text = text.translate(_TR_LOWER).lower()
    text = text.replace("’", "'")
    text = text.split("'")[0]  # "Güler'in" -> "güler"
    return re.sub(r"[^\wçğıöşü]", "", text)


def _suffix(raw: str) -> str:
    raw = raw.replace("’", "'").translate(_TR_LOWER).lower()
    return re.sub(r"[^\wçğıöşü]", "", raw.split("'", 1)[1]) if "'" in raw else ""


@dataclass
class Mention:
    time: float
    index: int
    suffix: str  # ismin aldığı ek, örn. "e" -> "Güler'e"


def player_names(player: str, aliases: list[str]) -> set[str]:
    names = {normalize(player.split()[-1])} | {normalize(a) for a in aliases}
    names.discard("")
    return names


def _is_name(token: str, names: set[str]) -> bool:
    return any(token.startswith(n) for n in names)


def find_mentions(words: list[Word], names: set[str]) -> list[Mention]:
    """Oyuncunun adının (ya da soyadının / takma adının) geçtiği kelimeleri bulur."""
    mentions = []
    for i, w in enumerate(words):
        token = normalize(w.text)
        if len(token) < 3:
            continue
        for name in names:
            # "gülerden", "gülere" gibi ekli halleri de yakalamak için önek karşılaştırması
            stem = token[: len(name)]
            if token.startswith(name) or (len(name) >= 5 and fuzz.ratio(stem, name) >= 85):
                if len(token) - len(name) <= 5:
                    mentions.append(Mention(w.start, i, _suffix(w.text) or token[len(name):]))
                    break
    # Spiker aynı aksiyonda adı defalarca söyleyebilir; 2 saniyeden yakın tekrarları at.
    # "Arda Güler'e" iki eşleşme verir; ekli olan son kelimeyi esas al.
    deduped = []
    for m in mentions:
        if not deduped or m.time - deduped[-1].time > 2.0:
            deduped.append(m)
        elif m.index - deduped[-1].index <= 2:
            deduped[-1] = Mention(deduped[-1].time, m.index, m.suffix or deduped[-1].suffix)
    return deduped


def other_player_boundary(words: list[Word], after: int, names: set[str]) -> int:
    """Oyuncunun adından sonra başka bir özel isim (muhtemelen başka bir oyuncu) geçen ilk kelime.
    Bu noktadan sonraki olaylar artık bizim oyuncumuza ait sayılmaz."""
    for j in range(after + 1, len(words)):
        raw = words[j].text.strip("\"'“”(")
        token = normalize(raw)
        if len(token) < 3 or not raw[:1].isupper() or raw.isupper() or _is_name(token, names):
            continue
        if words[j - 1].text.rstrip()[-1:] in ".!?":
            continue  # cümle başı, özel isim değil
        return j
    return len(words)


def find_events(words: list[Word]) -> list[tuple[float, int, str]]:
    compiled = {cat: [re.compile(p) for p in pats] for cat, pats in EVENT_PATTERNS.items()}
    events = []
    for i, w in enumerate(words):
        token = normalize(w.text)
        if not token:
            continue
        for cat, regexes in compiled.items():
            if any(r.search(token) for r in regexes):
                events.append((w.start, i, cat))
                break
    return events


def _context(words: list[Word], center: int, radius: int = 8) -> str:
    lo, hi = max(0, center - radius), min(len(words), center + radius + 1)
    return " ".join(w.text for w in words[lo:hi])


def clips_from_commentary(words: list[Word], player: str, aliases: list[str], source: str) -> list[Clip]:
    names = player_names(player, aliases)
    mentions = find_mentions(words, names)
    events = find_events(words)
    clips = []
    for m in mentions:
        boundary = other_player_boundary(words, m.index, names)
        nearby = [(t, i, c) for t, i, c in events if m.time - LOOK_BEFORE <= t <= m.time + GOAL_LOOK_AFTER]
        # Oyuncunun adından hemen sonraki birkaç kelimede pas/orta geçiyorsa gol değil asist;
        # asistte golü başka oyuncu attığı için sınırın ötesine bakılır.
        passed = any(c == "asist" and 0 <= i - m.index <= 4 for _, i, c in nearby)
        goal = next((e for e in nearby if e[2] == "gol" and (passed or e[1] < boundary)), None)
        close = [e for e in nearby if e[0] <= m.time + LOOK_AFTER and e[2] != "gol" and e[1] < boundary]

        if goal and passed:
            category, anchor = "asist", goal[0]
        elif goal:
            category, anchor = "gol", goal[0]
        elif close:
            order = ["faul", "çalım", "şut", "asist"]
            t, _, category = min(close, key=lambda e: (order.index(e[2]), abs(e[0] - m.time)))
            if category == "asist":
                category = "aksiyon"  # golle bitmeyen pas
            anchor = t
        else:
            category, anchor = "aksiyon", m.time

        reason = f"anlatım: \"{_context(words, m.index)}\""
        if category == "faul":
            # "Güler'e faul" -> oyuncuya yapılan; "Güler'in faulü" -> oyuncunun yaptığı
            who = "oyuncuya yapılan faul" if m.suffix in ("e", "a", "ye", "ya") else "faul (anlatımda adı geçiyor)"
            reason = f"{who}; {reason}"

        before, after = CLIP_WINDOWS[category]
        clips.append(Clip(source, max(0.0, anchor - before), anchor + after, category, CONFIDENCE[category], [reason]))
    return clips


# Büyük harfle başlayan ama oyuncu olmayan, anlatımda sık geçen kelimeler.
NOT_PLAYERS = {
    "galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor", "başakşehir", "türkiye", "real", "madrid",
    "barcelona", "barça", "bayern", "münih", "juventus", "milan", "inter", "liverpool", "chelsea", "arsenal",
    "manchester", "city", "united", "tottenham", "psg", "paris", "dortmund", "ajax", "porto", "benfica",
    "atletico", "napoli", "roma", "lazio", "sevilla", "almanya", "fransa", "ispanya", "italya", "ingiltere",
    "portekiz", "hollanda", "belçika", "brezilya", "arjantin", "hırvatistan", "avrupa", "dünya", "uefa",
    "fifa", "şampiyonlar", "ligi", "lig", "süper", "kupa", "var", "hakem", "allah", "tanrı", "efendim",
    "evet", "hayır", "tamam", "bakın", "bakalım", "şimdi", "ve", "ama", "bir", "bu", "şu", "o", "the",
    "and", "what", "oh", "yes", "no", "goal", "gol", "pazartesi", "salı", "çarşamba", "perşembe", "cuma",
    "cumartesi", "pazar", "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül",
    "ekim", "kasım", "aralık", "stadyum", "stadı", "arena", "park",
}


@dataclass
class Player:
    name: str  # gösterilecek ad, örn. "Arda Güler"
    aliases: list[str]  # anlatımda geçen diğer biçimler, örn. ["Arda"]
    mentions: int


def _clean(raw: str) -> str:
    raw = raw.strip("\"'“”()[],.!?;:…-").replace("’", "'")
    return raw.split("'")[0]


def discover_players(words: list[Word], squad: list[str] | None = None, min_mentions: int = 2) -> list[Player]:
    """Anlatımda adı geçen oyuncuları bulur. Kadro listesi verilirse sadece onları arar."""
    if squad:
        found = []
        for name in squad:
            names = player_names(name, [])
            n = len(find_mentions(words, names))
            if n:
                found.append(Player(name, [], n))
        return sorted(found, key=lambda p: -p.mentions)

    # Sık sık küçük harfle de yazılan kelimeler ("Top", "Şimdi") isim değildir.
    lower_counts: Counter[str] = Counter()
    upper_counts: Counter[str] = Counter()
    for w in words:
        raw = _clean(w.text)
        (lower_counts if raw[:1].islower() else upper_counts)[normalize(raw)] += 1

    counts: Counter[str] = Counter()  # soyad -> kaç kez geçti
    forms: dict[str, Counter[str]] = defaultdict(Counter)  # soyad -> "Arda Güler" gibi tam biçimler

    def candidate(i: int) -> bool:
        raw = _clean(words[i].text)
        token = normalize(raw)
        return (len(token) >= 3 and raw[:1].isupper() and not (raw.isupper() and len(raw) > 3)
                and token not in NOT_PLAYERS and lower_counts[token] <= 0.3 * upper_counts[token]
                and not any(r.search(token) for r in _EVENT_REGEXES))

    i = 0
    while i < len(words):
        if not candidate(i):
            i += 1
            continue
        parts = [_clean(words[i].text)]
        j = i + 1
        # "Arda Güler", "Kerem Aktürkoğlu" gibi ardışık büyük harfli kelimeleri tek isim say
        while (j < len(words) and j - i < 3 and candidate(j)
               and words[j - 1].text.rstrip()[-1:] not in ".!?,;:"):
            parts.append(_clean(words[j].text))
            j += 1
        key = normalize(parts[-1])
        counts[key] += 1
        forms[key][" ".join(parts)] += 1
        i = j

    # Tek başına geçen ön ad ("Arda") başka bir oyuncunun tam adının ilk kelimesiyse onunla birleştir.
    aliases: dict[str, list[str]] = defaultdict(list)
    first_names: dict[str, set[str]] = defaultdict(set)
    for key, variants in forms.items():
        if key in counts:
            for v in variants:
                parts = v.split()
                if len(parts) >= 2:
                    first_names[normalize(parts[0])].add(key)
    for key in list(counts):
        owners = first_names.get(key, set()) - {key}
        if len(owners) == 1 and all(len(v.split()) == 1 for v in forms[key]):
            owner = owners.pop()
            counts[owner] += counts.pop(key)
            aliases[owner].append(next(iter(forms[key])))

    players = []
    for key, n in counts.items():
        if n < min_mentions:
            continue
        variants = forms[key]
        multi = [v for v in variants if len(v.split()) >= 2]
        name = max(multi or variants, key=lambda v: variants[v])
        players.append(Player(name, aliases[key], n))
    return sorted(players, key=lambda p: -p.mentions)
