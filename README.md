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

## Model Eğitimi (Adım 2)

```bash
python train.py                              # varsayılan: EfficientNet-B0, 30 epoch
python train.py --epochs 40 --lr 5e-4 --dropout 0.4 --batch-size 32
python train.py --arch mobilenet_v3_large    # mimariyi değiştir
python train.py --smoke-test                 # 2 epoch x 2 batch hızlı doğrulama
```

- **Mimari:** EfficientNet-B0 (varsayılan; ~15.5 MB ≪ 95 MB sınırı). Ayrıca
  `mobilenet_v3_large/small`, `resnet18` desteklenir. Transfer learning (ImageNet).
- **Ayarlanabilir hiperparametreler:** `--epochs`, `--batch-size`, `--lr`,
  `--weight-decay`, `--dropout`, `--label-smoothing`, `--patience`, `--freeze-epochs`.
- **EarlyStopping:** doğrulama macro-F1 izlenir (F1 = 1. öncelikli metrik), iyileşme
  durunca eğitim durur.
- **Çıktılar:** en iyi model `outputs/checkpoints/best_model.pt`; epoch logları
  `outputs/logs/training_log.csv` ve `history.json` (Adım 3 grafikleri bunları kullanır).

## Değerlendirme ve Grafikler (Adım 3)

```bash
python evaluate.py                 # test kümesinde değerlendir + 3 grafiği üret
python evaluate.py --split val     # doğrulama kümesinde değerlendir
```

- **Metrikler:** sınıf bazlı (per-class) + macro/weighted ortalama Accuracy, Precision,
  Recall, F1 → ekrana yazılır ve `outputs/metrics.json` + `classification_report.txt`.
- **Üretilen grafikler** (`outputs/figures/`):
  - `loss_curve.png` — Training & Validation Loss
  - `accuracy_curve.png` — Training & Validation Accuracy
  - `confusion_matrix.png` — Normalized 8x8 Confusion Matrix (heatmap)

## Yol Haritası

- [x] **Adım 1:** Proje yapısı + veri ön işleme (preprocessing + augmentation)
- [x] **Adım 2:** Model mimarisi + eğitim hattı (EarlyStopping, <95 MB model)
- [x] **Adım 3:** Değerlendirme metrikleri + grafikler (F1 öncelikli, confusion matrix)
- [ ] **Adım 4:** Canlı tahmin web arayüzü
