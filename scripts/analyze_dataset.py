"""
analyze_dataset.py
==================
Veri ön işleme hattını uçtan uca DOĞRULAYAN ve veri setini RAPORLAYAN script.

Çalıştırma (proje kök dizininden):
    python scripts/analyze_dataset.py                # analiz + grafik + split özeti
    python scripts/analyze_dataset.py --verify       # ek olarak bozuk görüntü taraması
    python scripts/analyze_dataset.py --no-figures   # grafik kaydetme

Ne yapar?
  1) Sınıf dengesi analizini ekrana yazar.
  2) (opsiyonel) Bozuk görüntüleri tarar.
  3) train/val/test DataLoader'larını kurup bölme boyutlarını ve bir örnek
     batch'in tensör şeklini yazdırır  -> ön işleme hattının çalıştığını kanıtlar.
  4) Sınıf dağılımı çubuk grafiğini ve örnek augmentation ızgarasını
     outputs/figures/ altına kaydeder.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# --- Proje kökünü import yoluna ekle (script doğrudan çalıştırılabilsin diye) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # ekran (GUI) gerektirmeyen arka uç; grafiği dosyaya yazar
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.data.dataset import build_dataloaders, get_train_transforms
from src.utils.analysis import analyze_class_balance, compute_class_weights, verify_images


def save_distribution_plot(counts: dict[str, int], out_path: Path) -> None:
    """Sınıf dağılımını çubuk grafik olarak kaydeder."""
    labels = [config.DISPLAY_NAMES.get(k, k) for k in counts.keys()]
    values = list(counts.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#4C72B0")
    plt.title("Sınıf Başına Görüntü Sayısı (Veri Seti Dağılımı)")
    plt.ylabel("Görüntü sayısı")
    plt.xticks(rotation=30, ha="right")
    # Her çubuğun üstüne sayıyı yaz.
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 2, str(v),
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → Dağılım grafiği kaydedildi: {out_path}")


def save_augmentation_grid(out_path: Path, n: int = 8) -> None:
    """
    Aynı görüntüye uygulanan rastgele augmentation örneklerini ızgara olarak kaydeder.
    Veri çoğaltmanın görsel olarak çalıştığını doğrulamak için kullanışlıdır.
    """
    from torchvision.datasets import ImageFolder

    # Dönüşümsüz ham ImageFolder'dan bir görüntü yolu al.
    raw = ImageFolder(str(config.DATA_DIR))
    if len(raw) == 0:
        print("  ! Augmentation ızgarası için görüntü bulunamadı, atlanıyor.")
        return
    img_path, _ = raw.samples[0]
    from PIL import Image
    pil_img = Image.open(img_path).convert("RGB")

    tfm = get_train_transforms()
    mean = np.array(config.NORM_MEAN)
    std = np.array(config.NORM_STD)

    cols = 4
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(n):
        tensor = tfm(pil_img)                       # rastgele augmentation uygula
        arr = tensor.permute(1, 2, 0).numpy()       # C,H,W -> H,W,C
        arr = (arr * std + mean).clip(0, 1)         # normalizasyonu geri al (görüntüleme için)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(arr)
        ax.axis("off")
    plt.suptitle("Eğitim Veri Çoğaltma (Augmentation) Örnekleri")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → Augmentation örnekleri kaydedildi: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Veri seti analizi ve ön işleme doğrulaması")
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR),
                        help="8 sınıf klasörünü içeren veri dizini")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--verify", action="store_true",
                        help="Bozuk/okunamayan görüntüleri tara")
    parser.add_argument("--no-figures", action="store_true",
                        help="Grafik kaydetmeyi atla")
    args = parser.parse_args()

    # --- 1) Sınıf dengesi analizi ---
    counts = analyze_class_balance(args.data_dir)

    # Eğitimde kullanılabilecek sınıf ağırlıkları (bilgi amaçlı yazdır).
    weights = compute_class_weights(args.data_dir)
    print("\nÖnerilen sınıf ağırlıkları (weighted loss için, ters frekans):")
    for name, w in zip(config.CLASS_NAMES, weights):
        print(f"  {name:<16}: {w:.3f}")

    # --- 2) (opsiyonel) Bozuk görüntü taraması ---
    if args.verify:
        print("\nGörüntü bütünlüğü taranıyor (bu işlem biraz sürebilir)...")
        verify_images(args.data_dir)

    # --- 3) DataLoader'ları kur ve doğrula ---
    print("\nDataLoader'lar kuruluyor (stratified train/val/test bölmesi)...")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,  # script doğrulaması için tek işçi yeterli ve taşınabilir
    )
    print(f"  Sınıflar ({len(class_names)}): {class_names}")
    print(f"  Eğitim   (train): {len(train_loader.dataset):>5} görüntü, {len(train_loader)} batch")
    print(f"  Doğrulama(val)  : {len(val_loader.dataset):>5} görüntü, {len(val_loader)} batch")
    print(f"  Test            : {len(test_loader.dataset):>5} görüntü, {len(test_loader)} batch")

    # Bölmelerin sınıf dağılımını koruduğunu (stratified) doğrula.
    def split_distribution(subset) -> Counter:
        # Subset.indices -> temel veri setindeki örnek indeksleri.
        base_targets = subset.dataset.targets
        return Counter(base_targets[i] for i in subset.indices)

    print("\n  Bölme başına sınıf dağılımı (stratified doğrulaması):")
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        dist = split_distribution(loader.dataset)
        dist_str = ", ".join(f"{config.CLASS_NAMES[c]}:{dist[c]}" for c in range(config.NUM_CLASSES))
        print(f"    {split_name:<6}: {dist_str}")

    # Bir batch çekip tensör şeklini doğrula (ön işlemenin uçtan uca çalıştığının kanıtı).
    images, labels = next(iter(train_loader))
    print(f"\n  Örnek batch -> görüntü tensörü: {tuple(images.shape)} "
          f"(beklenen: (B, 3, {config.IMG_SIZE}, {config.IMG_SIZE}))")
    print(f"  Örnek batch -> etiket tensörü : {tuple(labels.shape)}, "
          f"değer aralığı [{int(labels.min())}, {int(labels.max())}]")
    print(f"  Piksel değer aralığı (normalize sonrası): "
          f"[{images.min():.2f}, {images.max():.2f}]")

    # --- 4) Grafikleri kaydet ---
    if not args.no_figures:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        print("\nGrafikler kaydediliyor...")
        save_distribution_plot(counts, config.FIGURES_DIR / "class_distribution.png")
        save_augmentation_grid(config.FIGURES_DIR / "augmentation_samples.png")

    print("\n✅ Veri ön işleme hattı baştan sona çalıştı.")


if __name__ == "__main__":
    main()
