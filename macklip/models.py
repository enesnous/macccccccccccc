from dataclasses import dataclass, field

# Kliplerin birleştirilmiş videoda görüneceği sıra.
CATEGORY_ORDER = [
    "gol", "asist", "gol sevinci", "pozisyon", "çalım", "şut", "faul", "kart", "oyuncu değişikliği",
    "yakın çekim", "röportaj", "antrenman", "tribün", "aksiyon",
]

CATEGORY_TITLES = {
    "gol": "GOL",
    "asist": "ASİST",
    "gol sevinci": "GOL SEVİNCİ",
    "pozisyon": "POZİSYON",
    "çalım": "ÇALIM",
    "şut": "ŞUT",
    "faul": "FAUL",
    "kart": "KART",
    "oyuncu değişikliği": "OYUNCU DEĞİŞİKLİĞİ",
    "yakın çekim": "YAKIN ÇEKİM",
    "röportaj": "RÖPORTAJ",
    "antrenman": "ANTRENMAN",
    "tribün": "TRİBÜN",
    "aksiyon": "AKSİYON",
}


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Clip:
    source: str
    start: float
    end: float
    category: str
    confidence: float
    reasons: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def priority(self) -> int:
        return CATEGORY_ORDER.index(self.category)


def clip_to_dict(c: Clip) -> dict:
    from pathlib import Path

    return {
        "kaynak": Path(c.source).name, "baslangic": round(c.start, 2), "bitis": round(c.end, 2),
        "kategori": c.category, "guven": round(c.confidence, 2), "neden": c.reasons,
    }


def upper_tr(text: str) -> str:
    return text.translate(str.maketrans({"i": "İ", "ı": "I"})).upper()
