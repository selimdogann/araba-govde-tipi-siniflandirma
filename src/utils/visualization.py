"""
visualization.py
================
Proje dokümanında istenen 3 GRAFİĞİ üreten modül:

  GRAFİK 1 -> Training & Validation Loss   (X: epoch, Y: loss, iki çizgi)
  GRAFİK 2 -> Training & Validation Accuracy (X: epoch, Y: accuracy %, iki çizgi)
  GRAFİK 3 -> Normalized Confusion Matrix (8x8 heatmap, hücreler 0.00-1.00)

Grafikler ekran (GUI) gerektirmeden doğrudan dosyaya kaydedilir.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # başsız (headless) arka uç: grafiği dosyaya yazar
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src import config


# ===========================================================================
# GRAFİK 1: Training & Validation Loss
# ===========================================================================
def plot_loss_curves(history: dict, out_path: Path) -> None:
    """
    Epoch bazlı eğitim ve doğrulama KAYIP (loss) eğrilerini çizer.

    İki çizgi arasındaki "gap" (fark) küçükse model iyi genelliyor demektir;
    validation loss eğitim loss'undan çok yukarıda ve artıyorsa overfitting işaretidir.
    """
    epochs = history["epoch"]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "o-", label="Training Loss", color="#4C72B0")
    plt.plot(epochs, history["val_loss"], "s-", label="Validation Loss", color="#DD8452")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → Loss grafiği: {out_path}")


# ===========================================================================
# GRAFİK 2: Training & Validation Accuracy
# ===========================================================================
def plot_accuracy_curves(history: dict, out_path: Path) -> None:
    """
    Epoch bazlı eğitim ve doğrulama DOĞRULUK (accuracy) eğrilerini çizer (% olarak).
    """
    epochs = history["epoch"]
    # Oranı (0-1) yüzdeye çevir.
    train_acc = [100.0 * a for a in history["train_acc"]]
    val_acc = [100.0 * a for a in history["val_acc"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, "o-", label="Training Accuracy", color="#4C72B0")
    plt.plot(epochs, val_acc, "s-", label="Validation Accuracy", color="#55A868")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → Accuracy grafiği: {out_path}")


# ===========================================================================
# GRAFİK 3: Normalized Confusion Matrix (8x8)
# ===========================================================================
def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    out_path: Path,
    class_labels: list[str] | None = None,
) -> np.ndarray:
    """
    Satır bazında NORMALİZE edilmiş 8x8 karışıklık matrisini heatmap olarak çizer.

    Satırlar: gerçek sınıf, Sütunlar: tahmin edilen sınıf.
    Her satır kendi içinde 1.0'a normalize edilir; köşegen (diagonal) değerler
    yüksekse model o sınıfı doğru tanıyor demektir.

    Returns:
        Normalize edilmiş confusion matrix (numpy array).
    """
    class_labels = class_labels or config.DISPLAY_LABELS
    n = len(class_labels)

    # Ham sayım matrisi (satır=gerçek, sütun=tahmin).
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    # Satır toplamına böl -> her satır kendi içinde oran (0-1). 0'a bölmeyi önle.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        cm_norm,
        annot=True, fmt=".2f",          # her hücreye 0.00-1.00 değerini yaz
        cmap="Blues", vmin=0.0, vmax=1.0,
        xticklabels=class_labels, yticklabels=class_labels,
        square=True, linewidths=0.5, cbar_kws={"label": "Oran"},
    )
    plt.title("Normalized Confusion Matrix (8x8)")
    plt.xlabel("Tahmin Edilen Sınıf (Predicted)")
    plt.ylabel("Gerçek Sınıf (True)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → Confusion matrix: {out_path}")
    return cm_norm
