"""
conftest.py
===========
Testlerin ortak hazırlıkları (pytest "fixture"ları).

Gerçek veri seti (5253 görsel) ve eğitilmiş model GitHub'da yok; testler bunlara
bağlı olursa kimsenin bilgisayarında çalışmaz. Bu yüzden:
  • tiny_dataset  -> 8 sınıf klasörlü, sınıf başına birkaç düz renkli görselden
                     oluşan MİNİK sahte veri seti üretir.
  • dummy_checkpoint -> ImageNet ağırlığı İNDİRMEDEN (pretrained=False) kurulan,
                     rastgele ağırlıklı bir modeli train.py ile AYNI formatta kaydeder.
Böylece testler "doğru tahmin" değil, hattın UÇTAN UCA ÇALIŞTIĞINI doğrular.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# Testler "from src import ..." diyebilsin diye proje kökünü import yoluna ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.models import build_model  # noqa: E402

IMAGES_PER_CLASS = 10  # stratified split'in her bölmeye en az 1 örnek koyabilmesi için yeterli


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    """8 sınıf × 10 görsellik sahte veri seti (her sınıf farklı renk)."""
    root = tmp_path_factory.mktemp("data")
    for class_idx, name in enumerate(config.CLASS_NAMES):
        class_dir = root / name
        class_dir.mkdir()
        for i in range(IMAGES_PER_CLASS):
            color = (class_idx * 30 % 256, i * 20 % 256, 128)
            Image.new("RGB", (64, 48), color).save(class_dir / f"{i}.jpg")
    return root


@pytest.fixture(scope="session")
def dummy_checkpoint(tmp_path_factory) -> Path:
    """train.py'nin kaydettiği sözlük formatında rastgele ağırlıklı küçük bir model."""
    arch = "mobilenet_v3_small"  # en küçük mimari: testler hızlı kalsın
    model = build_model(arch=arch, num_classes=config.NUM_CLASSES, dropout=0.2, pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "best_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "arch": arch,
        "class_names": config.CLASS_NAMES,
        "display_labels": config.DISPLAY_LABELS,
        "img_size": config.IMG_SIZE,
        "num_classes": config.NUM_CLASSES,
        "dropout": 0.2,
        "val_f1": 0.5,
        "val_acc": 0.5,
        "epoch": 1,
    }, path)
    return path
