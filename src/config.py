"""
config.py
=========
Projenin TÜM sabitlerini ve hiperparametre varsayılanlarını tek bir yerde toplar.
Böylece sınıf isimleri, görüntü boyutu, normalizasyon değerleri gibi bilgiler
dataset / train / evaluate / app dosyalarında tutarlı şekilde kullanılır.

Neden tek bir config dosyası?  -> "Single Source of Truth" (tek doğruluk kaynağı)
ilkesi. Bir değeri (örn. görüntü boyutu) değiştirmek istediğimizde sadece burayı
güncelleriz; kod tekrarını ve uyumsuzluk hatalarını önler.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1) DOSYA YOLLARI
# ---------------------------------------------------------------------------
# __file__  -> bu dosyanın (config.py) yolu.  .parent.parent -> proje kök dizini.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"          # 8 sınıf alt klasörünü içeren ham veri seti
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"     # grafikler, loglar, model çıktıları
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"      # kaydedilen grafikler (png)
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"  # eğitilmiş model ağırlıkları (.pt)
LOGS_DIR: Path = OUTPUTS_DIR / "logs"            # epoch bazlı eğitim logları (csv/json)

# ---------------------------------------------------------------------------
# 2) SINIFLAR (8 araba gövde tipi)
# ---------------------------------------------------------------------------
# torchvision.datasets.ImageFolder klasör adlarını ALFABETİK sıraya göre
# indeksler. Aşağıdaki liste bu alfabetik sırayla BİREBİR aynıdır; bu sayede
# model çıktısındaki indeks (0..7) ile sınıf adı her zaman doğru eşleşir.
#   0: hatchback, 1: micro, 2: open_wheel, 3: pickup,
#   4: sedan, 5: station_wagon, 6: suv, 7: van
CLASS_NAMES: list[str] = [
    "hatchback",
    "micro",
    "open_wheel",
    "pickup",
    "sedan",
    "station_wagon",
    "suv",
    "van",
]
NUM_CLASSES: int = len(CLASS_NAMES)  # = 8

# Klasör adı -> arayüzde/raporda gösterilecek okunabilir etiket (proje dokümanına göre).
# Not: Sunumda verilecek test scripti farklı bir etiket beklerse SADECE burası güncellenir.
DISPLAY_NAMES: dict[str, str] = {
    "hatchback": "HATCHBACK",
    "micro": "MICRO",
    "open_wheel": "AÇIK TEKERLEKLİ",
    "pickup": "PICK UP",
    "sedan": "SEDAN",
    "station_wagon": "STATION WAGON",
    "suv": "SUV",
    "van": "VAN",
}

# İndeks sırasına göre gösterim etiketleri (grafiklerde eksen etiketi olarak kullanılır).
DISPLAY_LABELS: list[str] = [DISPLAY_NAMES[name] for name in CLASS_NAMES]

# ---------------------------------------------------------------------------
# 3) GÖRÜNTÜ ÖN İŞLEME SABİTLERİ
# ---------------------------------------------------------------------------
IMG_SIZE: int = 224  # Modelin giriş boyutu (proje dokümanında önerilen 224x224)

# ImageNet ortalama (mean) ve standart sapma (std) değerleri.
# Transfer learning'de önceden eğitilmiş ağırlıklar bu istatistiklerle eğitildiği
# için girdiyi aynı şekilde normalize etmek modelin daha hızlı/iyi öğrenmesini sağlar.
NORM_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
NORM_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# 4) VERİ BÖLME (SPLIT) ORANLARI
# ---------------------------------------------------------------------------
# Kendi iç değerlendirmemiz için veriyi train/val/test olarak ayırıyoruz.
# (Jüri sunum sırasında AYRICA hiç görülmemiş kendi test setini verecek.)
VAL_SPLIT: float = 0.15   # doğrulama (validation) oranı
TEST_SPLIT: float = 0.15  # iç test oranı
# Geri kalan %70 -> eğitim (train)
RANDOM_SEED: int = 42     # Tekrarlanabilirlik (her çalıştırmada aynı bölme) için sabit tohum

# ---------------------------------------------------------------------------
# 5) VARSAYILAN HİPERPARAMETRELER (Adım 2'de train.py argümanlarıyla ezilebilir)
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
NUM_WORKERS: int = 4  # DataLoader'da paralel veri okuma işçisi sayısı
