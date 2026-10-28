"""Tarayıcı arayüzü: maçı yükle; bot görüntüyü inceleyip klipleri böler, kategoriler ve
her oyuncu ile her kategori için birleştirilmiş video hazırlar.

Çalıştırma:  macklip-arayuz   (ya da python -m macklip.web)
"""
import argparse
import itertools
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import pipeline
from .models import CATEGORY_ORDER, CATEGORY_TITLES, Clip, clip_to_dict

STATIC = Path(__file__).parent / "static"
DATA_DIR = Path.home() / "macklip_isler"

app = FastAPI(title="macklip")


@dataclass
class Collection:
    """Tek videoda birleştirilecek klipler: bir oyuncunun ya da bir kategorinin tüm klipleri."""
    no: int
    kind: str  # "oyuncu" | "kategori"
    name: str
    clips: list[Clip]
    selected: list[int]
    category: str | None = None  # kind == "kategori" ise
    named: bool = True  # oyuncunun adı ekranda okunabildi mi
    has_photo: bool = False
    status: str = "bekliyor"  # bekliyor | hazırlanıyor | hazır | hata
    progress: float = 0.0
    version: int = 0
    rendered: list[int] | None = None
    error: str | None = None

    def to_dict(self, job_id: str) -> dict:
        chosen = [self.clips[i] for i in self.selected]
        counts = {}
        for c in chosen:
            counts[c.category] = counts.get(c.category, 0) + 1
        return {
            "no": self.no,
            "tur": self.kind,
            "ad": self.name,
            "kategori": self.category,
            "isimli": self.named,
            "foto": f"/api/maclar/{job_id}/koleksiyon/{self.no}/foto" if self.has_photo else None,
            "durum": self.status,
            "ilerleme": round(self.progress, 3),
            "surum": self.version,
            "guncel": self.rendered == self.selected,
            "hata": self.error,
            "sure": round(sum(c.duration for c in chosen), 1),
            "sayilar": [{"kategori": k, "baslik": CATEGORY_TITLES[k], "adet": counts[k]}
                        for k in CATEGORY_ORDER if k in counts],
            "klipler": [{**clip_to_dict(c), "no": i, "baslik": CATEGORY_TITLES[c.category],
                         "secili": i in self.selected} for i, c in enumerate(self.clips)],
        }


@dataclass
class Job:
    id: str
    folder: Path
    match: str
    squad: list[str]
    title: str
    status: str = "sırada"  # sırada | analiz | hazır | hata
    stage: str = "Sırada bekliyor"
    progress: float = 0.0
    collections: list[Collection] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def report(self, stage: str, progress: float) -> None:
        self.stage, self.progress = stage, round(progress, 4)

    def video(self, c: Collection) -> Path:
        return self.folder / "videolar" / f"{c.no:03d}_v{c.version}.mp4"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "baslik": self.title,
            "durum": self.status,
            "asama": self.stage,
            "ilerleme": self.progress,
            "gecen_sure": round((self.finished or time.time()) - self.started),
            "hata": self.error,
            "goller": self.goals,
            "kaynak": f"/api/maclar/{self.id}/kaynak",
            "koleksiyonlar": [c.to_dict(self.id) for c in self.collections],
        }

    def save(self) -> None:
        """Analiz sonucunu diske yazar; uygulama kapanıp açılsa da maç kaybolmaz."""
        state = {
            "match": self.match, "squad": self.squad, "title": self.title, "goals": self.goals,
            "started": self.started, "finished": self.finished,
            "collections": [{**{k: v for k, v in asdict(c).items() if k != "clips"},
                             "clips": [asdict(x) for x in c.clips]} for c in self.collections],
        }
        (self.folder / "analiz.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "Job":
        s = json.loads((folder / "analiz.json").read_text(encoding="utf-8"))
        job = cls(folder.name, folder, s["match"], s["squad"], s["title"], status="hazır", stage="Hazır",
                  progress=1.0, goals=s["goals"], started=s["started"], finished=s["finished"])
        for c in s["collections"]:
            c["clips"] = [Clip(**x) for x in c["clips"]]
            col = Collection(**c)
            if col.status != "hazır" or not job.video(col).exists():
                col.status, col.rendered, col.progress = "bekliyor", None, 0.0
            job.collections.append(col)
        return job


jobs: dict[str, Job] = {}

# ---------------------------------------------------------------------------
# İş kuyruğu: işler işlemciyi tamamen kullandığı için tek tek, öncelik sırasıyla yürür.
# 0 = maç analizi, 1 = kullanıcının açtığı video, 5 = arka planda diğer videolar
# ---------------------------------------------------------------------------
tasks: queue.PriorityQueue = queue.PriorityQueue()
_seq = itertools.count()
PRIORITY_ANALYSIS, PRIORITY_OPENED, PRIORITY_BACKGROUND = 0, 1, 5


def enqueue(priority: int, job: Job, no: int | None = None) -> None:
    tasks.put((priority, next(_seq), job.id, no))


def _worker() -> None:
    while True:
        _, _, job_id, no = tasks.get()
        job = jobs.get(job_id)
        try:
            if job is None:
                continue
            if no is None:
                _analyze(job)
            else:
                _render(job, job.collections[no])
        except Exception:
            traceback.print_exc()
        finally:
            tasks.task_done()


def _analyze(job: Job) -> None:
    from .vision.bot import analyze

    job.status, job.started = "analiz", time.time()
    try:
        result = analyze(job.match, job.squad, job.report)
        photos = job.folder / "fotolar"
        photos.mkdir(exist_ok=True)
        for person, clips in result.people:
            c = Collection(len(job.collections), "oyuncu", person.name, clips, list(range(len(clips))),
                           named=person.named)
            if person.photo is not None:
                import cv2

                c.has_photo = cv2.imwrite(str(photos / f"{c.no:03d}.jpg"), person.photo)
            job.collections.append(c)
        for cat in CATEGORY_ORDER:
            clips = result.categories.get(cat)
            if clips:
                job.collections.append(Collection(len(job.collections), "kategori", CATEGORY_TITLES[cat], clips,
                                                  list(range(len(clips))), category=cat))
        job.goals = result.goals
        if not job.collections:
            raise RuntimeError("Videoda hiç klip bulunamadı. Bunun bir maç yayını olduğundan emin ol.")
        job.status, job.stage, job.progress, job.finished = "hazır", "Hazır", 1.0, time.time()
        job.save()
        # Önce kategoriler ve isimli oyuncular, sonra isimsiz kişiler
        for c in sorted(job.collections, key=lambda c: (c.kind == "oyuncu" and not c.named, c.kind == "oyuncu")):
            enqueue(PRIORITY_BACKGROUND, job, c.no)
    except Exception as e:
        traceback.print_exc()
        job.status, job.error, job.finished = "hata", str(e) or e.__class__.__name__, time.time()


def _render(job: Job, c: Collection) -> None:
    wanted = list(c.selected)
    if c.status == "hazır" and c.rendered == wanted:
        return
    c.status, c.progress, c.error = "hazırlanıyor", 0.0, None
    try:
        old = job.video(c) if c.version else None
        c.version += 1
        output = job.video(c)
        output.parent.mkdir(exist_ok=True)

        def progress(_stage: str, overall: float) -> None:
            # make_video toplam ilerlemenin son %15'ini raporlar; 0-1'e çevir
            c.progress = max(0.0, min(1.0, (overall - 0.85) / 0.15))

        pipeline.make_video([c.clips[i] for i in wanted], c.name, output, progress)
        if old and old != output:
            old.unlink(missing_ok=True)
        c.rendered, c.status, c.progress = wanted, "hazır", 1.0
    except Exception as e:
        traceback.print_exc()
        c.status, c.error = "hata", str(e)
    job.save()


def _restore() -> None:
    if not DATA_DIR.exists():
        return
    for folder in DATA_DIR.iterdir():
        if (folder / "analiz.json").exists():
            try:
                job = Job.load(folder)
            except Exception:
                traceback.print_exc()
                continue
            jobs[job.id] = job
            for c in job.collections:
                if c.status != "hazır":
                    enqueue(PRIORITY_BACKGROUND, job, c.no)


# ---------------------------------------------------------------------------
# HTTP uç noktaları
# ---------------------------------------------------------------------------

def _get(job_id: str) -> Job:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Maç bulunamadı")
    return job


def _collection(job: Job, no: int) -> Collection:
    if not 0 <= no < len(job.collections):
        raise HTTPException(404, "Bulunamadı")
    return job.collections[no]


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/maclar")
def list_jobs():
    return [{"id": j.id, "baslik": j.title, "durum": j.status,
             "oyuncu_sayisi": sum(c.kind == "oyuncu" for c in j.collections), "tarih": j.started}
            for j in sorted(jobs.values(), key=lambda j: -j.started)]


@app.post("/api/maclar")
def create_job(mac: UploadFile = File(...), kadro: str = Form("")):
    job_id = uuid.uuid4().hex[:10]
    folder = DATA_DIR / job_id
    folder.mkdir(parents=True)
    filename = Path(mac.filename or "mac.mp4").name
    match = folder / ("mac" + (Path(filename).suffix or ".mp4"))
    with match.open("wb") as f:
        shutil.copyfileobj(mac.file, f, length=8 * 1024 * 1024)
    squad = [line.strip() for line in kadro.replace(",", "\n").splitlines() if line.strip()]
    job = Job(job_id, folder, str(match), squad, title=Path(filename).stem)
    jobs[job_id] = job
    enqueue(PRIORITY_ANALYSIS, job)
    return job.to_dict()


@app.get("/api/maclar/{job_id}")
def job_status(job_id: str):
    return _get(job_id).to_dict()


@app.delete("/api/maclar/{job_id}")
def delete_job(job_id: str):
    job = _get(job_id)
    if job.status == "analiz" or any(c.status == "hazırlanıyor" for c in job.collections):
        raise HTTPException(409, "Bu maç üzerinde hâlâ çalışılıyor")
    jobs.pop(job_id)
    shutil.rmtree(job.folder, ignore_errors=True)
    return {"silindi": job_id}


@app.post("/api/maclar/{job_id}/koleksiyon/{no}/oncelik")
def prioritize(job_id: str, no: int):
    """Kullanıcı bir oyuncuyu/kategoriyi açınca videosu sıradakilerin önüne alınır."""
    job = _get(job_id)
    c = _collection(job, no)
    if c.status in ("bekliyor", "hata"):
        enqueue(PRIORITY_OPENED, job, no)
    return c.to_dict(job.id)


class Rename(BaseModel):
    ad: str


@app.post("/api/maclar/{job_id}/koleksiyon/{no}/ad")
def rename(job_id: str, no: int, body: Rename):
    """Bot adını okuyamadığı kişilere kullanıcı ad verebilir."""
    job = _get(job_id)
    c = _collection(job, no)
    if body.ad.strip():
        c.name, c.named = body.ad.strip(), True
        job.save()
    return c.to_dict(job.id)


class Selection(BaseModel):
    secili: list[int]


@app.post("/api/maclar/{job_id}/koleksiyon/{no}/yeniden")
def rerender(job_id: str, no: int, body: Selection):
    job = _get(job_id)
    c = _collection(job, no)
    chosen = sorted({i for i in body.secili if 0 <= i < len(c.clips)})
    if not chosen:
        raise HTTPException(400, "En az bir klip seç")
    c.selected = chosen
    if c.status != "hazırlanıyor":
        c.status = "bekliyor"
    enqueue(PRIORITY_OPENED, job, no)
    return c.to_dict(job.id)


@app.get("/api/maclar/{job_id}/koleksiyon/{no}/video")
def collection_video(job_id: str, no: int, indir: bool = False):
    job = _get(job_id)
    c = _collection(job, no)
    path = job.video(c)
    if not c.version or not path.exists():
        raise HTTPException(404, "Video henüz hazır değil")
    slug = "_".join(c.name.lower().split())
    return FileResponse(path, media_type="video/mp4", filename=f"{job.title}_{slug}.mp4" if indir else None)


@app.get("/api/maclar/{job_id}/koleksiyon/{no}/foto")
def collection_photo(job_id: str, no: int):
    job = _get(job_id)
    path = job.folder / "fotolar" / f"{no:03d}.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/maclar/{job_id}/kaynak")
def source_video(job_id: str):
    """Klip önizlemesi için maçın kendisi (tarayıcı istenen saniyeye atlar)."""
    return FileResponse(_get(job_id).match, media_type="video/mp4")


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(prog="macklip-arayuz", description="macklip web arayüzünü başlatır")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1", help="ağdaki başka cihazlardan erişim için 0.0.0.0")
    p.add_argument("--tarayici-acma", action="store_true", help="tarayıcıyı otomatik açma")
    a = p.parse_args()
    _restore()
    threading.Thread(target=_worker, daemon=True).start()
    url = f"http://localhost:{a.port}"
    print(f"\n  macklip arayüzü: {url}\n  Kapatmak için Ctrl+C\n")
    if not a.tarayici_acma:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
