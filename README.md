# Araba Gövde Tipi Sınıflandırma Projesi

Kocaeli Üniversitesi – Bilgisayar Mühendisliği – Yazılım Laboratuvarı-II / Proje III

8 araba gövde tipini (SUV, VAN, STATION WAGON, MICRO, AÇIK TEKERLEKLİ, SEDAN,
HATCHBACK, PICK UP) görüntüden sınıflandıran derin öğrenme projesi.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Veri Seti

Ham görseller `data/` altında, her sınıf bir alt klasör olacak şekilde durmalıdır:

```
data/
├── hatchback/      ├── sedan/
├── micro/          ├── station_wagon/
├── open_wheel/     ├── suv/
├── pickup/         └── van/
```

> `data/` klasörü `.gitignore` ile hariç tutulur (ham görseller GitHub'a yüklenmez).
> Kaynak: Kaggle "Cars Body Type Cropped" & "Stanford Car Body Type Data" referans
> alınarak oluşturulan özelleştirilmiş veri seti.

Mevcut veri seti dengeli: sınıf başına ~597–680 görsel, toplam 5253 (dengesizlik oranı 1.14x).

## Proje Yapısı

```
.
├── data/                       # Ham veri seti (gitignored)
├── src/
│   ├── config.py               # Tüm sabitler/hiperparametreler (tek doğruluk kaynağı)
│   ├── data/
│   │   └── dataset.py          # Transform + stratified split + DataLoader hattı
│   ├── models/                 # Model mimarileri (Adım 2)
│   └── utils/
│       └── analysis.py         # Sınıf dengesi analizi, sınıf ağırlıkları, bozuk görüntü taraması
├── scripts/
│   └── analyze_dataset.py      # Veri hattını uçtan uca doğrulayan + raporlayan script
├── app/                        # Web arayüzü (Adım 4)
├── outputs/figures/            # Üretilen grafikler
└── requirements.txt
```

## Kullanım

Veri seti analizini çalıştır (sınıf dengesi raporu + split özeti + grafikler):

```bash
python scripts/analyze_dataset.py            # analiz + grafik
python scripts/analyze_dataset.py --verify   # ek olarak bozuk görüntü taraması
```

## Ön İşleme Adımları (Adım 1)

- **Boyutlandırma:** 224×224 (model giriş boyutu)
- **Normalizasyon:** ImageNet mean/std (transfer learning ile uyumlu)
- **Veri çoğaltma (sadece eğitim):** RandomResizedCrop, yatay çevirme, ±15° döndürme,
  ColorJitter (parlaklık/kontrast/doygunluk/ton), küçük öteleme, RandomErasing
- **Bölme:** Stratified %70 train / %15 val / %15 test (sınıf dağılımı korunur, sabit tohum=42)
- Val/test verisine veri çoğaltma **uygulanmaz** (deterministik değerlendirme için).

## Yol Haritası

- [x] **Adım 1:** Proje yapısı + veri ön işleme (preprocessing + augmentation)
- [ ] **Adım 2:** Model mimarisi + eğitim hattı (EarlyStopping, <95 MB model)
- [ ] **Adım 3:** Değerlendirme metrikleri + grafikler (F1 öncelikli, confusion matrix)
- [ ] **Adım 4:** Canlı tahmin web arayüzü
