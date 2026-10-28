# macklip ⚽🎬 — maç klip botu

Maç videosunu yüklüyorsun. Bot maçı **görüntüden** izliyor (sesi kullanmıyor), sahneleri bölüyor, kategorilere ayırıyor ve:
- **her oyuncu için** o oyuncunun bütün kliplerini,
- **her kategori için** (bütün goller, bütün röportajlar...) o kategorideki bütün klipleri

maçtaki sırasıyla tek bir videoda birleştiriyor. Sen istediğini seçip indiriyorsun. Her şey kendi bilgisayarında çalışır; internet, hesap ya da API anahtarı gerekmez.

## Başlatma

1. [Python 3.10+](https://www.python.org/downloads/) ve FFmpeg kur:
   - Windows: `winget install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`
2. Başlatma dosyasına çift tıkla:
   - Windows: **`baslat.bat`**
   - macOS: **`baslat.command`**
   - Linux: `./baslat.sh`

İlk açılışta gerekenler kendiliğinden kurulur (birkaç dakika sürer). Sonra tarayıcıda arayüz açılır: **http://localhost:8000**

## Bot neye bakıyor?

| Kategori | Bot nasıl anlıyor (görüntüden) |
|---|---|
| **Gol** | Skor tabelasının yerini kendi bulur ve skoru okur. Skor değişince gol olmuştur; klip golden önceki oyundan başlar, sevinç ve tekrarlarla devam eder. |
| **Gol sevinci** | Golden hemen sonraki yakın çekimler |
| **Pozisyon** | Yayıncının **tekrar** gösterdiği anlar (şut, kurtarış, faul, tartışmalı pozisyon). Tekrar = sahada ama skor tabelası yok. |
| **Kart** | Havaya kaldırılmış sarı/kırmızı dikdörtgen ya da ekrandaki "SARI KART" yazısı |
| **Oyuncu değişikliği** | Ekrandaki "OYUNCU DEĞİŞİKLİĞİ" grafiği |
| **Yakın çekim** | Ekranı kaplayan büyük yüz |
| **Röportaj** | Sabit kamera, büyük yüz, skor tabelası yok |
| **Antrenman** | Çim var, skor tabelası yok, maç başlamadan önce ya da bittikten sonra |
| **Tribün** | Çim yok, kalabalık/dokulu görüntü |

**Oyuncular** iki yolla tanınır:
- **Ekran yazıları (OCR):** yayının ekrana yazdığı isim bandı, gol yazısı ya da oyuncu değişikliği grafiği okunur.
- **Yüzler:** aynı kişinin yüzleri gruplanır. İsim bandı göründüğü anda ekrandaki yüz o isimle eşleştirilir; böylece kişi isim bandı olmayan sahnelerde de tanınır.

Adı ekranda hiç yazmayan kişiler "İsimsiz kişi 1, 2..." olarak listelenir. Arayüzden onlara ad verebilirsin.

Gelişmiş ayarlardan **kadro listesi** girersen ekranda okunan yazılar bu listeyle eşleştirilir ve isimler daha doğru çıkar.

## Arayüzde

- **Oyuncular / Kategoriler** sekmeleri, her kartta **İndir** düğmesi
- Analiz bitince videolar arka planda sırayla hazırlanır. Açtığın kart sıranın başına alınır.
- Kartı açınca klipleri tek tek görürsün; ▶ ile maçtaki o anı izleyebilirsin.
- Yanlış bir klibin işaretini kaldırıp **Videoyu yeniden oluştur** dersen o klip videodan çıkarılır.
- Maçlar `~/macklip_isler` klasöründe saklanır. Uygulamayı kapatıp açsan da kaybolmaz.

## Süre

Bot saniyede iki kareye bakar. 90 dakikalık bir maç sıradan bir bilgisayarda yaklaşık **30-60 dakika** sürer (tahmin; 4 dakikalık test videosu 1 dakikada bitti).

## Sınırlamalar (dürüst liste)

- **Şut, faul ve çalım ayrı ayrı ayırt edilmiyor.** Hepsi yayıncı tekrar gösterdiyse **Pozisyon** olarak gelir. Bunları birbirinden ayırmak için eğitilmiş bir hareket tanıma modeli gerekir (sonraki adım).
- **Geniş açıda oyuncu tanıma yok.** Oyuncu ancak yüzü yakın görünürse ya da adı ekrana yazılırsa tanınır. Forma numarası okuma sonraki adım.
- **Gol, skor tabelasına bağlı.** Tabelası olmayan ya da tabelası çok küçük/şeffaf yayınlarda gol bulunamayabilir.
- **Sentetik testte doğrulandı, gerçek maçta değil.** Gerçek bir yayınla denemek eşikleri ayarlamak için gerekli.

## Komut satırı (eski, spiker anlatımına dayalı yöntem)

`macklip mac.mp4 --oyuncu "Arda Güler" --foto arda.jpg` hâlâ çalışır. Bu yöntem sesi kullanır; arayüz kullanmaz.

## Testler

```bash
.venv/bin/pip install pytest && .venv/bin/pytest -q
```
