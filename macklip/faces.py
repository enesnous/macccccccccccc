"""Referans fotoğraflardaki yüzü videoda arar: yakın çekim, röportaj ve antrenman klipleri."""
from dataclasses import dataclass

import cv2
import numpy as np

from . import facelib as face_recognition

from .models import Clip

TOLERANCE = 0.55  # yüz kodlamaları arası mesafe; küçük = daha katı
# Maçta yalnızca yakın çekim/röportajdaki büyük yüzler aranır, düşük çözünürlük yeterli.
# Antrenman görüntüleri genelde geniş açı; yüzler küçük olduğu için tam çözünürlükte bakılır.
MATCH_ANALYSIS_WIDTH = 640
TRAINING_ANALYSIS_WIDTH = 1280
INTERVIEW_MIN_SECONDS = 20.0


@dataclass
class FaceHit:
    time: float
    distance: float
    size: float  # yüz yüksekliği / kare yüksekliği


def load_references(paths: list[str]) -> list[np.ndarray]:
    encodings = []
    for p in paths:
        image = face_recognition.load_image_file(p)
        found = face_recognition.face_encodings(image)
        if not found:
            print(f"  uyarı: {p} içinde yüz bulunamadı, atlanıyor")
            continue
        # Fotoğrafta birden fazla kişi varsa en büyük yüzü al
        locs = face_recognition.face_locations(image)
        biggest = max(range(len(locs)), key=lambda i: locs[i][2] - locs[i][0])
        encodings.append(found[biggest])
    return encodings


def _faces_in_video(path: str, step: float, width: int, progress=None):
    """Videodan her step saniyede bir kare alıp (zaman, kare, [(konum, kodlama), ...]) üretir."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"video açılamadı: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every = max(1, round(fps * step))
    frame_no = 0
    while True:
        if frame_no % every:
            if not cap.grab():
                break
            frame_no += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_no / fps
        frame_no += 1
        w = frame.shape[1]
        scale = width / w if w > width else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=0)
        faces = []
        if locs:
            encs = face_recognition.face_encodings(rgb, locs)
            # Konumları tam çözünürlüğe çevir, yüz yüksekliğini kare oranı olarak ver
            faces = [(tuple(int(v / scale) for v in loc), enc, (loc[2] - loc[0]) / rgb.shape[0])
                     for loc, enc in zip(locs, encs)]
        yield t, frame, faces
        if total and frame_no % (every * 10) == 1:
            if progress:
                progress(min(1.0, frame_no / total))
            else:
                print(f"\r  yüz taraması: %{100 * frame_no / total:4.1f}", end="", flush=True)
    cap.release()


def scan_video(path: str, references: list[np.ndarray], step: float = 1.0,
               width: int = MATCH_ANALYSIS_WIDTH, progress=None) -> list[FaceHit]:
    hits = []
    for t, _, faces in _faces_in_video(path, step, width, progress):
        for _, enc, size in faces:
            dist = float(np.min(face_recognition.face_distance(references, enc)))
            if dist <= TOLERANCE:
                hits.append(FaceHit(t, dist, size))
                break
    if not progress:
        print(f"\r  yüz taraması: tamam ({len(hits)} eşleşme)          ")
    return hits


@dataclass
class FaceCluster:
    """Fotoğraf olmadan bulunan, aynı kişiye ait olduğu düşünülen yüzler."""
    hits: list[FaceHit]
    encodings: list[np.ndarray]
    crop: np.ndarray | None = None  # en büyük görünümden kırpılmış yüz (BGR)
    crop_size: float = 0.0

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.encodings[-20:], axis=0)


def _crop(frame: np.ndarray, loc: tuple, out: int = 200) -> np.ndarray:
    top, right, bottom, left = loc
    pad = int((bottom - top) * 0.35)
    h, w = frame.shape[:2]
    y0, y1 = max(0, top - pad), min(h, bottom + pad)
    x0, x1 = max(0, left - pad), min(w, right + pad)
    side = min(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    face = frame[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2]
    return cv2.resize(face, (out, out))


def cluster_faces(path: str, step: float = 1.0, width: int = MATCH_ANALYSIS_WIDTH,
                  progress=None, min_hits: int = 3) -> list[FaceCluster]:
    """Videodaki tüm yüzleri bulup kişilere göre gruplar (kim olduklarını bilmeden)."""
    clusters: list[FaceCluster] = []
    for t, frame, faces in _faces_in_video(path, step, width, progress):
        for loc, enc, size in faces:
            if size < 0.08:
                continue  # uzak, küçük yüzler güvenilmez
            dists = [float(np.linalg.norm(c.centroid - enc)) for c in clusters]
            best = int(np.argmin(dists)) if dists else -1
            if best >= 0 and dists[best] <= TOLERANCE - 0.05:
                cluster = clusters[best]
                cluster.hits.append(FaceHit(t, dists[best], size))
                cluster.encodings.append(enc)
            else:
                cluster = FaceCluster([FaceHit(t, 0.0, size)], [enc])
                clusters.append(cluster)
            if size > cluster.crop_size:
                cluster.crop, cluster.crop_size = _crop(frame, loc), size
    if not progress:
        print(f"\r  yüz taraması: tamam ({len(clusters)} farklı yüz)          ")
    return [c for c in clusters if len(c.hits) >= min_hits]


def group_hits(hits: list[FaceHit], step: float) -> list[list[FaceHit]]:
    groups: list[list[FaceHit]] = []
    for h in hits:
        # Kamera kısa süre başka yere dönse bile aynı sahne sayılsın diye birkaç adım boşluğa izin ver
        if groups and h.time - groups[-1][-1].time <= step * 3 + 0.5:
            groups[-1].append(h)
        else:
            groups.append([h])
    return groups


def clips_from_faces(hits: list[FaceHit], source: str, step: float, training: bool = False) -> list[Clip]:
    clips = []
    for g in group_hits(hits, step):
        start, end = g[0].time - 1.0, g[-1].time + step + 1.0
        duration = end - start
        best = min(h.distance for h in g)
        confidence = round(max(0.3, min(0.95, 1.0 - best)), 2)
        reason = f"yüz eşleşmesi ({len(g)} kare, en iyi mesafe {best:.2f})"
        if training:
            category = "antrenman"
        elif duration >= INTERVIEW_MIN_SECONDS:
            category = "röportaj"
        else:
            category = "yakın çekim"
            if duration < 2.5:
                continue  # tek karelik yanlış eşleşmeleri ele
        clips.append(Clip(source, max(0.0, start), end, category, confidence, [reason]))
    return clips
