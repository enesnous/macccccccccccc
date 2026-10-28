"""Videoyu bir kez dolaşıp her örnek kare için görsel ölçümleri çıkarır. Ses kullanılmaz.

Toplanan bilgiler:
  - çim oranı, hareket, kamera geçişleri (renk histogramı farkı)
  - yüzler (aynı kişinin yüzleri gruplanır)
  - skor tabelası: konumu kendiliğinden bulunur, skoru OCR ile okunur
  - ekrana gelen yazılar (isim bandı, gol yazısı, oyuncu değişikliği...): OCR
  - olası sarı/kırmızı kart
"""
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import facelib as face_recognition

STEP = 0.5  # saniyede 2 örnek kare
FACE_EVERY = 2  # yüz araması her 2 örnekte bir (saniyede 1)
ANALYSIS_WIDTH = 640
FACE_WIDTH = 960  # yüz algılayıcı ~80 piksellik yüzden küçüğünü göremez
THUMB = (320, 180)
FACE_TOLERANCE = 0.55
MIN_FACE_SIZE = 0.08  # yüz yüksekliği / kare yüksekliği; uzak yüzler güvenilmez
SCOREBOARD_READ_EVERY = 10.0  # saniye
CAPTION_READ_GAP = 2.0  # aynı yazıyı tekrar tekrar okumamak için en kısa aralık


@dataclass
class Frame:
    t: float
    green: float  # çim rengindeki piksel oranı
    motion: float  # önceki örneğe göre ortalama piksel farkı
    cut: bool  # önceki örnekle arasında kamera geçişi var mı
    edge_density: float  # kalabalık/tribün gibi dokulu görüntüler için
    scoreboard: bool = False
    face_checked: bool = False  # bu karede yüz arandı mı (her karede aranmaz)
    face_ids: list[int] = field(default_factory=list)
    face_sizes: list[float] = field(default_factory=list)


@dataclass
class TextSeen:
    t: float
    text: str
    confidence: float
    region: str  # "alt" (isim bandı bölgesi) ya da "tabela"


@dataclass
class CardSeen:
    t: float
    color: str  # "sarı" | "kırmızı"


@dataclass
class FaceGroup:
    id: int
    times: list[float] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)
    encodings: list[np.ndarray] = field(default_factory=list)
    crop: np.ndarray | None = None
    crop_size: float = 0.0

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.encodings[-30:], axis=0)


@dataclass
class ScanResult:
    duration: float
    frames: list[Frame]
    texts: list[TextSeen]
    scores: list[tuple[float, tuple[int, int]]]
    cards: list[CardSeen]
    faces: list[FaceGroup]
    scoreboard_box: tuple[int, int, int, int] | None  # x, y, w, h (THUMB ölçeğinde)


class _OCR:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def __call__(self, image: np.ndarray) -> list[tuple[str, float, float]]:
        """(yazı, güven, dikey konum 0-1) listesi döndürür."""
        result, _ = self.engine(image)
        out = []
        for box, text, conf in result or []:
            y = (box[0][1] + box[2][1]) / 2 / image.shape[0]
            out.append((text.strip(), float(conf), y))
        return out


def _green_ratio(small_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (32, 50, 40), (88, 255, 255))
    return float(mask.mean() / 255)


def _hist(small_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _cards(small_bgr: np.ndarray) -> list[str]:
    """Hakemin havaya kaldırdığı sarı/kırmızı kartı arar: küçük, düz renkli, dik duran dikdörtgen."""
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    area = small_bgr.shape[0] * small_bgr.shape[1]
    found = []
    masks = {
        "sarı": cv2.inRange(hsv, (20, 150, 170), (34, 255, 255)),
        "kırmızı": cv2.inRange(hsv, (0, 160, 140), (7, 255, 255)) | cv2.inRange(hsv, (173, 160, 140), (180, 255, 255)),
    }
    for color, mask in masks.items():
        if mask.mean() / 255 > 0.04:
            continue  # o renkte forma/pano var; kart ayırt edilemez
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            a = cv2.contourArea(c)
            if not 0.0006 * area <= w * h <= 0.012 * area:
                continue
            if 1.15 <= h / max(w, 1) <= 1.9 and a / (w * h) >= 0.8 and y < small_bgr.shape[0] * 0.75:
                found.append(color)
                break
    return found


_CLOCK = re.compile(r"\d{1,3}[:.]\d{2}")
_NUM = re.compile(r"\d{1,2}")


def parse_score(texts: list[tuple[str, float, float]]) -> tuple[int, int] | None:
    """Tabela yazılarından skoru çıkarır; maç saatini (34:12) ayıklar."""
    joined = " ".join(t for t, conf, _ in texts if conf > 0.5)
    joined = _CLOCK.sub(" ", joined)
    # OCR rakamları bazen harf okur: tek başına duran ya da rakama yapışık O/I harflerini düzelt
    joined = re.sub(r"(?<![A-Za-z])[Oo](?![A-Za-z])", "0", joined)
    joined = re.sub(r"(?<![A-Za-z])[Il|](?![A-Za-z])", "1", joined)
    m = re.search(r"(\d{1,2})\s*[-–:|]\s*(\d{1,2})", joined)
    nums = [int(m.group(1)), int(m.group(2))] if m else [int(n) for n in _NUM.findall(joined)]
    nums = [n for n in nums if n <= 15]
    return (nums[0], nums[1]) if len(nums) >= 2 else None


def _crop_face(frame: np.ndarray, loc: tuple, out: int = 200) -> np.ndarray:
    top, right, bottom, left = loc
    pad = int((bottom - top) * 0.35)
    h, w = frame.shape[:2]
    y0, y1 = max(0, top - pad), min(h, bottom + pad)
    x0, x1 = max(0, left - pad), min(w, right + pad)
    side = max(1, min(y1 - y0, x1 - x0))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    face = frame[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2]
    return cv2.resize(face, (out, out)) if face.size else np.zeros((out, out, 3), np.uint8)


def scan(path: str, progress=None) -> ScanResult:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"video açılamadı: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    every = max(1, round(fps * STEP))
    ocr = _OCR()

    frames: list[Frame] = []
    texts: list[TextSeen] = []
    scores: list[tuple[float, tuple[int, int]]] = []
    cards: list[CardSeen] = []
    groups: list[FaceGroup] = []

    prev_gray = prev_hist = None
    band_history: list[np.ndarray] = []  # alt bölgenin son iki örneği (sabit yazı tespiti için)
    last_caption_t = -99.0
    last_caption_edges = None
    # Skor tabelası: saha karelerinde hep aynı yerde duran kenarlar. Yeri maçın tamamından
    # öğrenilir; bunun için her karenin üst/alt şerit kenarları sıkıştırılıp saklanır.
    edge_sum = np.zeros(THUMB[::-1], np.float32)
    learned = 0
    band_edges: list[np.ndarray] = []

    frame_no = -1
    sample = -1
    while True:
        frame_no += 1
        if frame_no % every:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        sample += 1
        t = frame_no / fps
        h, w = frame.shape[:2]
        scale = ANALYSIS_WIDTH / w if w > ANALYSIS_WIDTH else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        thumb = cv2.resize(small, THUMB)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        hist = _hist(thumb)

        green = _green_ratio(thumb)
        motion = float(np.mean(cv2.absdiff(gray, prev_gray))) if prev_gray is not None else 0.0
        cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.45
        f = Frame(t, green, motion, cut, float(edges.mean() / 255))
        prev_gray, prev_hist = gray, hist

        # --- Skor tabelası için kenarları biriktir ---
        if green > 0.3:
            edge_sum += edges / 255.0
            learned += 1
        band_edges.append(np.packbits(edges.astype(bool)))

        # --- Yüzler ---
        if sample % FACE_EVERY == 0:
            f.face_checked = True
            fscale = FACE_WIDTH / w if w > FACE_WIDTH else 1.0
            face_img = cv2.resize(frame, None, fx=fscale, fy=fscale) if fscale != 1.0 else frame
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=0)
            if locs:
                encs = face_recognition.face_encodings(rgb, locs)
                for loc, enc in zip(locs, encs):
                    size = (loc[2] - loc[0]) / rgb.shape[0]
                    if size < MIN_FACE_SIZE:
                        continue
                    dists = [float(np.linalg.norm(g.centroid - enc)) for g in groups]
                    best = int(np.argmin(dists)) if dists else -1
                    if best >= 0 and dists[best] <= FACE_TOLERANCE:
                        g = groups[best]
                    else:
                        g = FaceGroup(len(groups))
                        groups.append(g)
                    g.times.append(t)
                    g.sizes.append(size)
                    g.encodings.append(enc)
                    if size > g.crop_size:
                        full_loc = tuple(int(v / fscale) for v in loc)
                        g.crop, g.crop_size = _crop_face(frame, full_loc), size
                    f.face_ids.append(g.id)
                    f.face_sizes.append(size)

        # --- Kart ---
        for color in _cards(small):
            cards.append(CardSeen(t, color))

        # --- Ekrana gelen yazılar (alt %40) ---
        # Yazı bandı arka plan hareket etse de sabit durur: son üç örnekte değişmeyen kenarlar.
        band = edges[int(THUMB[1] * 0.6):, :]
        band_gray = gray[int(THUMB[1] * 0.6):, :]
        band_history = (band_history + [band_gray])[-3:]
        if len(band_history) == 3 and t - last_caption_t >= CAPTION_READ_GAP:
            still = (cv2.absdiff(band_history[0], band_history[2]) < 8) & (cv2.absdiff(band_history[1], band_history[2]) < 8)
            static_edges = band.astype(bool) & still
            density = static_edges.mean()
            moving_bg = motion > 3.0
            fresh = last_caption_edges is None or np.mean(static_edges != last_caption_edges) > 0.02
            if density > 0.012 and fresh and (moving_bg or not f.scoreboard):
                last_caption_t, last_caption_edges = t, static_edges
                y0 = int(h * 0.6)
                region = frame[y0:, :]
                if region.shape[1] > 1280:
                    region = cv2.resize(region, None, fx=1280 / region.shape[1], fy=1280 / region.shape[1])
                for s, c, _ in ocr(region):
                    if c > 0.6 and len(s) >= 3:
                        texts.append(TextSeen(t, s, c, "alt"))

        frames.append(f)
        if progress and sample % 20 == 0:
            progress(min(1.0, frame_no / total))

    cap.release()

    # --- Tabelanın yeri, hangi karelerde göründüğü ve skor ---
    box = None
    if learned >= 20:
        box, box_mask = _find_scoreboard(edge_sum / learned)
        if box_mask is not None:
            n_pixels = THUMB[0] * THUMB[1]
            for f, packed in zip(frames, band_edges):
                e = np.unpackbits(packed)[:n_pixels].reshape(THUMB[::-1])
                f.scoreboard = _has_scoreboard(e, box_mask)
            scores, board_texts = _read_scores(path, frames, box, ocr)
            texts += board_texts
    groups, id_map = _merge_groups(groups)
    for f in frames:
        f.face_ids = [id_map.get(i, i) for i in f.face_ids]
    return ScanResult(frame_no / fps, frames, texts, scores, cards, groups, box)


def _find_scoreboard(edge_freq: np.ndarray):
    """Saha karelerinin çoğunda aynı yerde duran, yazıya benzeyen kenar kümesi = skor tabelası."""
    mask = (edge_freq > 0.55).astype(np.uint8)
    hgt, wid = mask.shape
    # Saha çizgisi gibi uzun ince sabit çizgileri çıkar; tabelanın yazıları kalır
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((int(hgt * 0.15), 1), np.uint8))
    lines |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, int(wid * 0.3)), np.uint8))
    mask &= 1 - lines
    # Saha çizgisi gibi uzun ince sabit çizgileri çıkar; tabelanın yazıları kalır
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((int(hgt * 0.15), 1), np.uint8))
    lines |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, int(wid * 0.3)), np.uint8))
    mask &= 1 - lines
    # Tabela ekranın üst ya da alt kenarına yakındır
    mask[int(hgt * 0.3):int(hgt * 0.75), :] = 0
    if mask.sum() < 25:
        return None, None
    dil = cv2.dilate(mask, np.ones((5, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dil)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        density = mask[labels == i].mean()
        # Yatay, alçak, içi yazı dolu bir kutu olmalı (saha çizgileri ince ve uzundur)
        if not 0.035 * hgt <= h <= 0.15 * hgt or w < 1.8 * h or w > 0.7 * wid or density < 0.12:
            continue
        score = stats[i, cv2.CC_STAT_AREA] * density
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None, None
    x, y, w, h = stats[best, :4]
    region = (labels == best) & mask.astype(bool)
    return (int(x), int(y), int(w), int(h)), region


def _has_scoreboard(edges: np.ndarray, mask: np.ndarray) -> bool:
    return float((edges.astype(bool) & mask).sum() / max(mask.sum(), 1)) > 0.45


def _read_scores(path: str, frames: list[Frame], box, ocr) -> tuple[list, list]:
    """Tabelanın göründüğü karelerden belli aralıklarla skoru okur (ikinci, hızlı geçiş)."""
    cap = cv2.VideoCapture(path)
    scores, texts = [], []
    last = -99.0
    x, y, bw, bh = box
    for f in frames:
        if not f.scoreboard or f.t - last < SCOREBOARD_READ_EVERY:
            continue
        last = f.t
        cap.set(cv2.CAP_PROP_POS_MSEC, f.t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        sx, sy = w / THUMB[0], h / THUMB[1]
        crop = frame[max(0, int((y - 2) * sy)):int((y + bh + 2) * sy), max(0, int((x - 2) * sx)):int((x + bw + 2) * sx)]
        if not crop.size:
            continue
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        read = ocr(crop)
        score = parse_score(read)
        if score:
            scores.append((f.t, score))
        texts += [TextSeen(f.t, s, c, "tabela") for s, c, _ in read]
    cap.release()
    return scores, texts


def _merge_groups(groups: list[FaceGroup], threshold: float = 0.45) -> tuple[list[FaceGroup], dict[int, int]]:
    """Aynı kişinin farklı açı/ışıkta ayrı gruplara düşmüş yüzlerini birleştirir."""
    id_map: dict[int, int] = {}
    merged: list[FaceGroup] = []
    for g in sorted(groups, key=lambda g: -len(g.times)):
        target = next((m for m in merged if np.linalg.norm(m.centroid - g.centroid) <= threshold), None)
        if target is None:
            merged.append(g)
            continue
        id_map[g.id] = target.id
        target.times += g.times
        target.sizes += g.sizes
        target.encodings += g.encodings
        if g.crop_size > target.crop_size:
            target.crop, target.crop_size = g.crop, g.crop_size
    for m in merged:
        order = sorted(range(len(m.times)), key=lambda i: m.times[i])
        m.times = [m.times[i] for i in order]
        m.sizes = [m.sizes[i] for i in order]
    return merged, id_map
