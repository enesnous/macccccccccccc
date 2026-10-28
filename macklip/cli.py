"""Komut satırı: macklip MAÇ.mp4 --oyuncu "Arda Güler" --foto arda.jpg"""
import argparse
import json
import sys
from pathlib import Path

from . import pipeline
from .models import CATEGORY_ORDER, CATEGORY_TITLES, clip_to_dict


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="macklip",
        description="Bir futbol maçından seçtiğin oyuncunun tüm kliplerini çıkarıp tek videoda birleştirir.",
    )
    p.add_argument("mac", help="maç videosu (mp4, mkv...)")
    p.add_argument("--oyuncu", required=True, help='oyuncunun adı, örn. "Arda Güler"')
    p.add_argument("--takma-ad", action="append", default=[], help="spikerin kullandığı başka ad (tekrarlanabilir)")
    p.add_argument("--foto", action="append", default=[], help="oyuncunun yüz fotoğrafı (birkaç tane verirsen daha iyi)")
    p.add_argument("--antrenman", action="append", default=[], help="antrenman videosu (tekrarlanabilir)")
    p.add_argument("--ek-video", action="append", default=[],
                   help="röportaj gibi başka videolar; yüz taramasıyla aranır")
    p.add_argument("--cikti", default=None, help="çıktı videosu (varsayılan: <oyuncu>_klipler.mp4)")
    p.add_argument("--dil", default="tr", help="anlatım dili (tr, en, es...)")
    p.add_argument("--model", default="small", help="Whisper modeli: tiny/base/small/medium/large-v3")
    p.add_argument("--anlatim", help="hazır anlatım dosyası (.srt ya da önceki çalıştırmanın .json'u)")
    p.add_argument("--adim", type=float, default=1.0, help="yüz taramasında kaç saniyede bir kare bakılsın")
    p.add_argument("--sirala", choices=["kategori", "zaman"], default="zaman")
    p.add_argument("--haric", default="", help=f"dahil edilmeyecek kategoriler, virgülle: {','.join(CATEGORY_ORDER)}")
    p.add_argument("--min-guven", type=float, default=0.0, help="bu güvenin altındaki klipleri at (0-1)")
    p.add_argument("--sahne-yok", action="store_true", help="kamera kesmesi tespitini atla (daha hızlı)")
    p.add_argument("--sadece-rapor", action="store_true", help="video üretme, sadece klip listesini çıkar")
    return p.parse_args(argv)


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def main(argv=None) -> int:
    a = parse_args(argv)
    match = str(Path(a.mac).resolve())
    if not Path(match).exists():
        print(f"hata: {a.mac} bulunamadı", file=sys.stderr)
        return 1
    slug = "_".join(a.oyuncu.lower().split())
    output = Path(a.cikti or f"{slug}_klipler.mp4").resolve()
    work = output.with_suffix("")
    clips = pipeline.find_clips(pipeline.Options(
        match=match,
        player=a.oyuncu,
        aliases=a.takma_ad,
        photos=a.foto,
        training=a.antrenman,
        extra_videos=a.ek_video,
        language=a.dil,
        model=a.model,
        transcript=a.anlatim,
        transcript_cache=f"{work}_anlatim.json",
        step=a.adim,
        order=a.sirala,
        excluded={c.strip() for c in a.haric.split(",") if c.strip()},
        min_confidence=a.min_guven,
        detect_scenes=not a.sahne_yok,
    ))

    if not clips:
        print("Hiç klip bulunamadı. Oyuncu adını/takma adı ya da fotoğrafları kontrol et.")
        return 2

    report = Path(f"{work}_rapor.json")
    report.write_text(json.dumps([clip_to_dict(c) for c in clips], ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(c.duration for c in clips)
    print(f"\n  {len(clips)} klip, toplam {fmt(total)}")
    for cat in CATEGORY_ORDER:
        n = sum(1 for c in clips if c.category == cat)
        if n:
            print(f"    {CATEGORY_TITLES[cat]:<12} {n}")
    for c in clips:
        print(f"    [{fmt(c.start)} - {fmt(c.end)}] {CATEGORY_TITLES[c.category]:<12} güven {c.confidence:.2f}  "
              f"{Path(c.source).name}")
    print(f"  rapor: {report}")

    if a.sadece_rapor:
        return 0

    pipeline.make_video(clips, a.oyuncu, output)
    print(f"\nHazır: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
