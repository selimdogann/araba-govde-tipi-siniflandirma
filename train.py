"""
train.py
========
Araba gövde tipi sınıflandırma modelinin EĞİTİM scripti.

Özellikler:
  • Hiperparametreler (batch size, epoch, dropout, learning rate ...) komut satırı
    argümanı olarak dışarıdan ayarlanabilir.
  • Transfer learning + (opsiyonel) iki aşamalı eğitim: önce head, sonra fine-tune.
  • Overfitting'e karşı EarlyStopping.
  • Her epoch sonunda Train/Val Loss ve Accuracy + Val macro-F1 değerleri CSV ve
    JSON log dosyalarına yazılır (Adım 3 grafikleri bunları kullanır).
  • En iyi model (en yüksek doğrulama F1'i) outputs/checkpoints/best_model.pt olarak kaydedilir.

Örnek kullanım:
    python train.py                                   # varsayılan hiperparametrelerle tam eğitim
    python train.py --epochs 40 --lr 5e-4 --dropout 0.4
    python train.py --smoke-test                      # 2 batch'lik hızlı doğrulama (test eğitimi)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from src import config
from src.data.dataset import build_dataloaders
from src.engine import evaluate, get_device, train_one_epoch
from src.models import (
    SUPPORTED_ARCHS,
    build_model,
    count_parameters,
    estimate_model_size_mb,
    set_backbone_trainable,
)
from src.utils import EarlyStopping

# Diske kaydedilecek modelin AŞMAMASI gereken sınır (proje şartı).
MODEL_SIZE_LIMIT_MB = 95.0


def set_seed(seed: int) -> None:
    """Tekrarlanabilirlik için tüm rastgelelik kaynaklarını sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Komut satırı hiperparametrelerini tanımlar ve okur."""
    p = argparse.ArgumentParser(description="Araba gövde tipi sınıflandırıcı eğitimi")
    # --- Mimari ---
    p.add_argument("--arch", type=str, default="efficientnet_b0", choices=SUPPORTED_ARCHS,
                   help="Model mimarisi (varsayılan: efficientnet_b0)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="ImageNet ön-eğitimli ağırlıkları KULLANMA (sıfırdan eğit)")
    # --- Hiperparametreler (dışarıdan ayarlanabilir) ---
    p.add_argument("--epochs", type=int, default=30, help="Maksimum epoch sayısı")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch (yığın) boyutu")
    p.add_argument("--lr", type=float, default=1e-3, help="Öğrenme oranı (learning rate)")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Ağırlık çürümesi (L2 düzenlileştirme)")
    p.add_argument("--dropout", type=float, default=0.3, help="Sınıflandırıcı head dropout oranı")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Etiket yumuşatma (aşırı güveni azaltıp genellemeyi artırır)")
    # --- Eğitim stratejisi ---
    p.add_argument("--freeze-epochs", type=int, default=0,
                   help="İlk kaç epoch boyunca backbone dondurulup sadece head eğitilsin")
    p.add_argument("--patience", type=int, default=7, help="EarlyStopping sabrı (epoch)")
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE, help="Görüntü giriş boyutu")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="DataLoader işçi sayısı")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Rastgelelik tohumu")
    # --- Çıktı / hızlı test ---
    p.add_argument("--output-dir", type=str, default=str(config.OUTPUTS_DIR), help="Çıktı kök dizini")
    p.add_argument("--smoke-test", action="store_true",
                   help="Sadece 2 epoch x 2 batch çalıştır (pipeline'ı hızlıca doğrulamak için)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Cihaz: {device}")

    # --- Çıktı klasörleri ---
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # --- Smoke-test ayarları (hızlı doğrulama) ---
    limit_batches = 2 if args.smoke_test else 0
    epochs = 2 if args.smoke_test else args.epochs

    # --- Veri ---
    print("Veri yükleniyor...")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"  train={len(train_loader.dataset)} | val={len(val_loader.dataset)} | test={len(test_loader.dataset)}")

    # --- Model ---
    model = build_model(
        arch=args.arch,
        num_classes=config.NUM_CLASSES,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)
    print(f"Mimari: {args.arch} | Parametre: {total_params:,} (eğitilebilir: {trainable_params:,})")
    print(f"Tahmini model boyutu: {size_mb:.1f} MB (sınır: {MODEL_SIZE_LIMIT_MB} MB)")
    # 95 MB sınırını eğitim BAŞLAMADAN kontrol et; aşıyorsa erken uyar.
    if size_mb > MODEL_SIZE_LIMIT_MB:
        raise SystemExit(f"HATA: Model {size_mb:.1f} MB > {MODEL_SIZE_LIMIT_MB} MB sınırı! Daha küçük bir --arch seçin.")

    # --- Kayıp, optimizer, scheduler ---
    # CrossEntropyLoss: çok sınıflı sınıflandırmanın standart kaybı (logits + label).
    # label_smoothing: modelin tek sınıfa %100 güvenmesini engelleyip genellemeyi artırır.
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # AdamW: adaptif öğrenme oranlı, ağırlık çürümesini doğru uygulayan optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ReduceLROnPlateau: val F1 platoya girince öğrenme oranını düşürür (daha ince ayar).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 2)
    )

    # --- İki aşamalı eğitim: ilk freeze_epochs boyunca backbone dondurulur ---
    if args.freeze_epochs > 0:
        set_backbone_trainable(model, args.arch, trainable=False)
        print(f"Backbone donduruldu: ilk {args.freeze_epochs} epoch sadece head eğitilecek.")

    # --- EarlyStopping: val macro-F1'i izle (F1 = 1. öncelikli metrik), maksimize et ---
    stopper = EarlyStopping(patience=args.patience, mode="max")

    # --- Log yapıları ---
    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "lr": [],
    }
    csv_path = logs_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr"])

    best_f1 = 0.0
    print("\nEğitim başlıyor...\n" + "=" * 70)

    for epoch in range(1, epochs + 1):
        # Donmuş aşama bittiğinde backbone'u çöz (fine-tuning'e geç).
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, args.arch, trainable=True)
            print(f"[Epoch {epoch}] Backbone çözüldü -> tüm ağ fine-tune ediliyor.")

        t0 = time.time()
        # --- Eğitim adımı ---
        train_loss, train_acc = train_one_epoch(
            model, tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False),
            criterion, optimizer, device, limit_batches,
        )
        # --- Doğrulama adımı ---
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device, limit_batches,
        )
        # Doğrulama macro-F1 (sınıf başına F1'lerin ortalaması; dengesizliğe duyarlı).
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        # Scheduler'a izlenen metriği ver (LR'yi gerekirse düşürür).
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"Epoch {epoch:>2}/{epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
              f"lr={current_lr:.2e} | {dt:.1f}s")

        # --- Logla (JSON history + CSV) ---
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                                    f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}", f"{current_lr:.8f}"])
        with open(logs_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # --- EarlyStopping + en iyi modeli kaydet ---
        improved = stopper.step(val_f1)
        if improved:
            best_f1 = val_f1
            # state_dict + meta veriyi birlikte kaydet ki app/evaluate modeli yeniden kurabilsin.
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "class_names": class_names,
                "display_labels": config.DISPLAY_LABELS,
                "img_size": args.img_size,
                "num_classes": config.NUM_CLASSES,
                "dropout": args.dropout,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_dir / "best_model.pt")
            print(f"        ↑ En iyi model güncellendi (val_f1={val_f1:.4f}) -> best_model.pt")

        if stopper.should_stop:
            print(f"\nEarlyStopping: {args.patience} epoch boyunca iyileşme yok, eğitim durduruldu.")
            break

    print("=" * 70)
    print(f"Eğitim tamamlandı. En iyi doğrulama macro-F1: {best_f1:.4f}")

    # Kaydedilen model dosyasının gerçek boyutunu doğrula.
    ckpt_path = ckpt_dir / "best_model.pt"
    if ckpt_path.exists():
        real_mb = ckpt_path.stat().st_size / (1024 ** 2)
        print(f"Kaydedilen model: {ckpt_path}  ({real_mb:.1f} MB / {MODEL_SIZE_LIMIT_MB} MB sınırı)")
    print(f"Loglar: {csv_path}  ve  {logs_dir / 'history.json'}")


if __name__ == "__main__":
    main()
