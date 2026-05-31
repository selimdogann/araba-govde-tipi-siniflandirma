"""
engine.py
=========
Eğitim ve değerlendirme DÖNGÜLERİ (train/eval loop) ve cihaz seçimi.

train.py (Adım 2) ve evaluate.py (Adım 3) bu fonksiyonları ortak kullanır;
böylece eğitim ve değerlendirmede AYNI ileri-besleme (forward) mantığı çalışır.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """
    Kullanılabilir en hızlı cihazı seçer.

    Öncelik: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
    Bu Mac'te MPS kullanılır (Metal Performance Shaders).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int = 0,
) -> tuple[float, float]:
    """
    Modeli bir epoch boyunca EĞİTİR (ağırlıklar güncellenir).

    Args:
        model:         Eğitilecek model.
        loader:        Eğitim DataLoader'ı.
        criterion:     Kayıp fonksiyonu (ör. CrossEntropyLoss).
        optimizer:     Optimizasyon algoritması (ör. AdamW).
        device:        Hesaplama cihazı (mps/cuda/cpu).
        limit_batches: >0 ise yalnızca bu kadar batch işlenir (hızlı smoke-test için).

    Returns:
        (ortalama_kayıp, doğruluk) bu epoch için.
    """
    model.train()  # Dropout/BatchNorm gibi katmanları EĞİTİM moduna alır.
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        # Veriyi hesaplama cihazına taşı.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()            # önceki adımın gradyanlarını temizle
        outputs = model(images)          # ileri besleme: ham skorlar (logits)
        loss = criterion(outputs, labels)  # tahmin ile gerçek arasındaki kayıp
        loss.backward()                  # geri yayılım: gradyanları hesapla
        optimizer.step()                 # ağırlıkları güncelle

        # İstatistikler (kayıp toplam görüntüyle ağırlıklandırılır).
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)    # en yüksek skorlu sınıf = tahmin
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()  # değerlendirmede gradyan hesaplama (bellek/hız tasarrufu)
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    limit_batches: int = 0,
) -> tuple[float, float, list[int], list[int]]:
    """
    Modeli bir veri kümesi üzerinde DEĞERLENDİRİR (ağırlık güncellenmez).

    Returns:
        (ortalama_kayıp, doğruluk, tüm_tahminler, tüm_gerçek_etiketler).
        Tahmin ve etiket listeleri Adım 3'te metrik/confusion matrix için kullanılır.
    """
    model.eval()  # Dropout kapanır, BatchNorm çalışan istatistikleri kullanır.
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for i, (images, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Metrik hesabı için CPU'ya alıp listeye ekle.
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_preds, all_labels
