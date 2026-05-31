"""
analysis.py
===========
Veri seti üzerinde keşifsel analiz (EDA) yardımcı fonksiyonları.

Proje dokümanı, sınıflar arası DENGEYE dikkat edilmesini istiyor ("bir sınıfta çok
fazla, diğerinde çok az görsel olması modelin performansını olumsuz etkiler").
Bu modül tam da bunu ölçüp ekrana raporlar.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PIL import Image

from src import config

# Geçerli kabul edilen görüntü uzantıları (küçük harf).
VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ===========================================================================
# 1) SINIF BAŞINA GÖRÜNTÜ SAYIMI
# ===========================================================================
def count_images_per_class(
    data_dir: Path | str = config.DATA_DIR,
    class_names: list[str] | None = None,
) -> "OrderedDict[str, int]":
    """
    Her sınıf klasöründeki geçerli görüntü sayısını döndürür.

    Args:
        data_dir:    8 sınıf alt klasörünü içeren kök dizin.
        class_names: Sayılacak sınıflar (None ise config.CLASS_NAMES kullanılır).

    Returns:
        {sınıf_adı: görüntü_sayısı} sözlüğü (config sırasıyla).
    """
    data_dir = Path(data_dir)
    class_names = class_names or config.CLASS_NAMES

    counts: "OrderedDict[str, int]" = OrderedDict()
    for name in class_names:
        class_dir = data_dir / name
        if not class_dir.is_dir():
            # Klasör yoksa 0 yaz (eksikliği raporda görebilmek için hata fırlatmıyoruz).
            counts[name] = 0
            continue
        # Sadece geçerli uzantıya sahip dosyaları say.
        n = sum(
            1 for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        counts[name] = n
    return counts


# ===========================================================================
# 2) SINIF DENGESİ ANALİZİ + EKRAN RAPORU
# ===========================================================================
def analyze_class_balance(
    data_dir: Path | str = config.DATA_DIR,
    imbalance_threshold: float = 1.5,
) -> "OrderedDict[str, int]":
    """
    Sınıf dağılımını hesaplar ve ekrana okunabilir bir tablo olarak yazdırır.

    Dengesizlik ölçütü olarak (en kalabalık sınıf / en seyrek sınıf) oranını
    kullanır. Bu oran 'imbalance_threshold' değerini aşarsa uyarı gösterir.

    Args:
        data_dir:            Kök veri dizini.
        imbalance_threshold: Bu oranın üstü "dengesiz" sayılır (varsayılan 1.5).

    Returns:
        Sınıf başına görüntü sayısı sözlüğü.
    """
    counts = count_images_per_class(data_dir)
    total = sum(counts.values())

    print("=" * 60)
    print("SINIF DENGESİ ANALİZİ (Class Balance Analysis)")
    print("=" * 60)

    if total == 0:
        print("UYARI: Hiç görüntü bulunamadı! Veri dizinini kontrol edin:")
        print(f"  {Path(data_dir).resolve()}")
        return counts

    # Tablo başlığı.
    print(f"{'Sınıf':<16}{'Görüntü':>9}{'Yüzde':>8}   {'Dağılım'}")
    print("-" * 60)

    max_count = max(counts.values())
    for name, n in counts.items():
        pct = 100.0 * n / total
        # Basit metin tabanlı çubuk grafik (en kalabalık sınıf = 30 karakter).
        bar = "█" * int(round(30 * n / max_count)) if max_count else ""
        print(f"{name:<16}{n:>9}{pct:>8.1f}%  {bar}")

    print("-" * 60)
    print(f"{'TOPLAM':<16}{total:>9}{100.0:>8.1f}%")
    print("-" * 60)

    # Dengesizlik metriği.
    min_count = min(counts.values())
    max_count = max(counts.values())
    min_class = min(counts, key=counts.get)
    max_class = max(counts, key=counts.get)
    ratio = (max_count / min_count) if min_count > 0 else float("inf")

    print(f"Ortalama / sınıf : {total / len(counts):.1f}")
    print(f"En seyrek sınıf  : {min_class} ({min_count})")
    print(f"En kalabalık     : {max_class} ({max_count})")
    print(f"Dengesizlik oranı: {ratio:.2f}x  (max/min)")

    if ratio > imbalance_threshold:
        print(
            f"\n⚠️  DİKKAT: Dengesizlik oranı {ratio:.2f}x > eşik {imbalance_threshold}x.\n"
            "   Eğitimde sınıf ağırlıklı kayıp (weighted loss) veya örnekleme "
            "(oversampling) düşünülebilir."
        )
    else:
        print(f"\n✅ Veri seti DENGELİ (oran {ratio:.2f}x ≤ eşik {imbalance_threshold}x).")
    print("=" * 60)

    return counts


# ===========================================================================
# 3) SINIF AĞIRLIKLARI (eğitimde weighted loss için)
# ===========================================================================
def compute_class_weights(
    data_dir: Path | str = config.DATA_DIR,
) -> list[float]:
    """
    Ters frekans (inverse frequency) yöntemiyle sınıf ağırlıklarını hesaplar.

    Az örnekli sınıflara daha yüksek ağırlık verilir; böylece eğitimde model bu
    sınıfları ihmal etmez. Ağırlıklar ortalaması ~1 olacak şekilde normalize edilir.
    (Adım 2'de CrossEntropyLoss(weight=...) içinde kullanılabilir.)

    Returns:
        config.CLASS_NAMES sırasına göre ağırlık listesi.
    """
    counts = count_images_per_class(data_dir)
    n_classes = len(counts)
    total = sum(counts.values())
    if total == 0:
        return [1.0] * n_classes

    # weight_c = total / (n_classes * count_c)  -> seyrek sınıf büyük ağırlık alır.
    weights = [
        (total / (n_classes * n)) if n > 0 else 0.0
        for n in counts.values()
    ]
    return weights


# ===========================================================================
# 4) BOZUK GÖRÜNTÜ DOĞRULAMA
# ===========================================================================
def verify_images(
    data_dir: Path | str = config.DATA_DIR,
) -> list[Path]:
    """
    Tüm görüntüleri açmayı deneyerek BOZUK/okunamayan dosyaları tespit eder.

    Eğitim sırasında okunamayan bir dosya hata verip süreci durdurabilir; bu
    fonksiyonu önceden çalıştırıp sorunlu dosyaları temizlemek güvenlidir.

    Returns:
        Açılamayan dosya yollarının listesi (boşsa tüm görüntüler sağlam).
    """
    data_dir = Path(data_dir)
    corrupt: list[Path] = []
    checked = 0

    for name in config.CLASS_NAMES:
        class_dir = data_dir / name
        if not class_dir.is_dir():
            continue
        for p in class_dir.iterdir():
            if not (p.is_file() and p.suffix.lower() in VALID_EXTENSIONS):
                continue
            checked += 1
            try:
                # verify() dosyanın bütünlüğünü piksel verisini tam yüklemeden kontrol eder.
                with Image.open(p) as img:
                    img.verify()
            except Exception:  # bozuk/eksik dosya
                corrupt.append(p)

    print(f"Doğrulanan görüntü: {checked} | Bozuk: {len(corrupt)}")
    for p in corrupt:
        print(f"  ✗ {p}")
    return corrupt
