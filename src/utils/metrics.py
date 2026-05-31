"""
metrics.py
==========
Sınıflandırma performans METRİKLERİNİ hesaplayan ve raporlayan modül.

Proje şartı: Accuracy, Precision, Recall, F1-Score değerleri hem SINIF BAZLI
(per-class) hem de ORTALAMA (macro ve weighted) olarak hesaplanmalı.
1. ÖNCELİKLİ metrik F1-Score'dur.

Tanımlar (kısaca):
  • Precision (Kesinlik) = TP / (TP + FP) -> "pozitif dediklerimin ne kadarı doğru?"
  • Recall (Duyarlılık)  = TP / (TP + FN) -> "gerçek pozitiflerin ne kadarını yakaladım?"
  • F1 = Precision ve Recall'un harmonik ortalaması (dengesizliğe karşı dengeli ölçü).
  • macro avg    = her sınıfın metriğinin DÜZ ortalaması (her sınıf eşit ağırlık).
  • weighted avg = sınıfların örnek sayısına göre AĞIRLIKLI ortalaması.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from src import config


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict:
    """
    Sınıf bazlı ve ortalama metrikleri hesaplar.

    Args:
        y_true:      Gerçek sınıf indeksleri.
        y_pred:      Modelin tahmin ettiği sınıf indeksleri.
        class_names: Sınıf adları (None ise config.CLASS_NAMES).

    Returns:
        {
          "accuracy": float,
          "per_class": {sınıf: {precision, recall, f1, support}},
          "macro_avg":   {precision, recall, f1},
          "weighted_avg":{precision, recall, f1},
        }
    """
    class_names = class_names or config.CLASS_NAMES
    labels = list(range(len(class_names)))  # 0..7 (tüm sınıfların raporda yer almasını garanti eder)

    # Genel doğruluk (doğru tahmin / toplam).
    accuracy = accuracy_score(y_true, y_pred)

    # Sınıf bazlı precision/recall/f1/support (average=None -> her sınıf ayrı).
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }

    # Ortalamalar.
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "per_class": per_class,
        "macro_avg": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted_avg": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
    }


def print_metrics_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict:
    """
    Metrikleri hesaplar ve ekrana okunabilir bir tablo olarak yazar.

    Returns:
        compute_metrics çıktısı (dosyaya kaydetmek için).
    """
    class_names = class_names or config.CLASS_NAMES
    display = [config.DISPLAY_NAMES.get(c, c) for c in class_names]
    m = compute_metrics(y_true, y_pred, class_names)

    print("=" * 72)
    print("MODEL DEĞERLENDİRME METRİKLERİ  (1. öncelik: F1-Score)")
    print("=" * 72)
    print(f"{'Sınıf':<18}{'Precision':>11}{'Recall':>10}{'F1-Score':>11}{'Support':>10}")
    print("-" * 72)
    for name, disp in zip(class_names, display):
        c = m["per_class"][name]
        print(f"{disp:<18}{c['precision']:>11.4f}{c['recall']:>10.4f}{c['f1']:>11.4f}{c['support']:>10}")
    print("-" * 72)
    print(f"{'Accuracy':<18}{'':>11}{'':>10}{m['accuracy']:>11.4f}{sum(c['support'] for c in m['per_class'].values()):>10}")
    ma, wa = m["macro_avg"], m["weighted_avg"]
    print(f"{'Macro avg':<18}{ma['precision']:>11.4f}{ma['recall']:>10.4f}{ma['f1']:>11.4f}")
    print(f"{'Weighted avg':<18}{wa['precision']:>11.4f}{wa['recall']:>10.4f}{wa['f1']:>11.4f}")
    print("=" * 72)
    # En önemli tek satır özet (jüri için).
    print(f">> Macro F1 = {ma['f1']:.4f} | Weighted F1 = {wa['f1']:.4f} | Accuracy = {m['accuracy']:.4f}")
    print("=" * 72)
    return m


# sklearn'in hazır metin raporu (yedek/karşılaştırma amaçlı).
def sklearn_text_report(y_true: list[int], y_pred: list[int], class_names: list[str] | None = None) -> str:
    class_names = class_names or config.CLASS_NAMES
    display = [config.DISPLAY_NAMES.get(c, c) for c in class_names]
    return classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=display, zero_division=0, digits=4,
    )
