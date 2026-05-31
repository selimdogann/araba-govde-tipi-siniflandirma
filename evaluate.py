"""
evaluate.py
===========
Eğitilmiş modelin performansını DEĞERLENDİREN ve proje grafiklerini üreten script.

Yaptıkları:
  1) outputs/checkpoints/best_model.pt modelini yükler.
  2) Test kümesi (eğitimde KULLANILMAYAN, sabit tohumla ayrılan %15) üzerinde tahmin yapar.
  3) Sınıf bazlı + macro/weighted Accuracy, Precision, Recall, F1 hesaplar (F1 önceliklidir).
  4) Üç grafiği outputs/figures/ altına kaydeder:
       - loss_curve.png            (Training & Validation Loss)
       - accuracy_curve.png        (Training & Validation Accuracy)
       - confusion_matrix.png      (Normalized 8x8 Confusion Matrix)
  5) Metrikleri outputs/metrics.json ve metnsel raporu outputs/classification_report.txt olarak yazar.

Kullanım:
    python evaluate.py
    python evaluate.py --split test          # (varsayılan) test kümesinde değerlendir
    python evaluate.py --split val           # doğrulama kümesinde değerlendir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.nn as nn

from src import config
from src.data.dataset import build_dataloaders
from src.engine import evaluate, get_device
from src.models import load_checkpoint_model
from src.utils.metrics import print_metrics_report, sklearn_text_report
from src.utils.visualization import (
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_loss_curves,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model değerlendirme ve grafik üretimi")
    p.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINTS_DIR / "best_model.pt"),
                   help="Değerlendirilecek model dosyası")
    p.add_argument("--split", type=str, default="test", choices=["test", "val"],
                   help="Hangi küme üzerinde değerlendirilsin (varsayılan: test)")
    p.add_argument("--output-dir", type=str, default=str(config.OUTPUTS_DIR))
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Cihaz: {device}")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Modeli yükle ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Model bulunamadı: {ckpt_path}\nÖnce 'python train.py' ile eğitin.")
    model, ckpt = load_checkpoint_model(str(ckpt_path), device)
    class_names = ckpt.get("class_names", config.CLASS_NAMES)
    print(f"Model yüklendi: {ckpt['arch']} | eğitimdeki en iyi val_f1={ckpt.get('val_f1', float('nan')):.4f} "
          f"(epoch {ckpt.get('epoch', '?')})")

    # --- Veri (aynı tohum -> aynı test bölmesi; eğitimde görülmedi) ---
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        img_size=ckpt.get("img_size", config.IMG_SIZE),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=config.RANDOM_SEED,
    )
    loader = test_loader if args.split == "test" else val_loader
    print(f"Değerlendirme kümesi: {args.split} ({len(loader.dataset)} görüntü)")

    # --- Tahminleri topla ---
    criterion = nn.CrossEntropyLoss()
    _, accuracy, y_pred, y_true = evaluate(model, loader, criterion, device)

    # --- Metrikleri hesapla ve yazdır ---
    metrics = print_metrics_report(y_true, y_pred, class_names)

    # Metrikleri JSON + metin raporu olarak kaydet.
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"split": args.split, **metrics}, f, indent=2, ensure_ascii=False)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(sklearn_text_report(y_true, y_pred, class_names))
    print(f"\nMetrikler kaydedildi: {output_dir / 'metrics.json'}")

    # --- GRAFİK 1 & 2: history.json'dan loss/accuracy eğrileri ---
    history_path = config.LOGS_DIR / "history.json"
    print("\nGrafikler üretiliyor...")
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_loss_curves(history, figures_dir / "loss_curve.png")
        plot_accuracy_curves(history, figures_dir / "accuracy_curve.png")
    else:
        print(f"  ! Uyarı: {history_path} yok; loss/accuracy grafikleri atlandı "
              "(önce train.py çalıştırın).")

    # --- GRAFİK 3: Normalized Confusion Matrix ---
    plot_confusion_matrix(
        y_true, y_pred,
        figures_dir / "confusion_matrix.png",
        class_labels=[config.DISPLAY_NAMES.get(c, c) for c in class_names],
    )

    print(f"\n✅ Değerlendirme tamamlandı. Macro F1 = {metrics['macro_avg']['f1']:.4f}")


if __name__ == "__main__":
    main()
