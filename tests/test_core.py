"""
test_core.py
============
Kodun yapı taşlarını tek tek doğrulayan birim testleri:
config, veri dönüşümleri, stratified split, model kurucu, EarlyStopping, metrikler.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from PIL import Image

from src import config
from src.data.dataset import build_datasets, get_eval_transforms, get_train_transforms
from src.models import SUPPORTED_ARCHS, build_model, set_backbone_trainable
from src.utils import EarlyStopping, compute_metrics


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_class_names_are_alphabetical():
    # ImageFolder klasörleri alfabetik indeksler; liste farklı sırada olursa
    # model "sedan" dediğinde arayüz başka bir sınıf adı gösterir.
    assert config.CLASS_NAMES == sorted(config.CLASS_NAMES)
    assert config.NUM_CLASSES == 8


def test_every_class_has_a_display_name():
    assert set(config.DISPLAY_NAMES) == set(config.CLASS_NAMES)
    assert len(config.DISPLAY_LABELS) == config.NUM_CLASSES


# ---------------------------------------------------------------------------
# Dönüşümler (transforms)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [(640, 480), (100, 300), (224, 224)])
def test_transforms_give_model_input_shape(size):
    image = Image.new("RGB", size, (200, 10, 10))
    for tfm in (get_train_transforms(), get_eval_transforms()):
        assert tfm(image).shape == (3, config.IMG_SIZE, config.IMG_SIZE)


def test_eval_transform_is_deterministic():
    image = Image.new("RGB", (300, 200), (10, 120, 30))
    tfm = get_eval_transforms()
    assert torch.equal(tfm(image), tfm(image))


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def test_split_covers_all_images_without_overlap(tiny_dataset):
    train_ds, val_ds, test_ds, class_names = build_datasets(data_dir=tiny_dataset)
    assert class_names == config.CLASS_NAMES

    train_idx, val_idx, test_idx = set(train_ds.indices), set(val_ds.indices), set(test_ds.indices)
    # Hiçbir görüntü iki bölmede birden olmamalı (veri sızıntısı = şişirilmiş test skoru).
    assert not train_idx & val_idx
    assert not train_idx & test_idx
    assert not val_idx & test_idx
    assert len(train_idx | val_idx | test_idx) == 8 * 10


def test_split_keeps_every_class_in_test(tiny_dataset):
    _, _, test_ds, _ = build_datasets(data_dir=tiny_dataset)
    targets = test_ds.dataset.targets
    counts = Counter(targets[i] for i in test_ds.indices)
    assert set(counts) == set(range(config.NUM_CLASSES))


def test_split_is_reproducible(tiny_dataset):
    a = build_datasets(data_dir=tiny_dataset)
    b = build_datasets(data_dir=tiny_dataset)
    assert a[2].indices == b[2].indices


def test_val_and_test_use_eval_transform(tiny_dataset):
    train_ds, val_ds, test_ds, _ = build_datasets(data_dir=tiny_dataset)
    # Augmentation yalnız eğitimde olmalı: val/test aynı (çoğaltmasız) base veri setini paylaşır.
    assert val_ds.dataset is test_ds.dataset
    assert train_ds.dataset is not val_ds.dataset


def test_missing_data_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_datasets(data_dir=tmp_path / "yok")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
def test_every_arch_outputs_8_logits(arch):
    model = build_model(arch=arch, num_classes=8, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 8)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model(arch="vgg16", pretrained=False)


def test_freeze_leaves_only_head_trainable():
    model = build_model(arch="mobilenet_v3_small", pretrained=False)
    set_backbone_trainable(model, "mobilenet_v3_small", trainable=False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("classifier") for n in trainable)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------
def test_early_stopping_stops_after_patience():
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.5) is True
    assert stopper.step(0.5) is False   # iyileşme yok (1)
    assert not stopper.should_stop
    assert stopper.step(0.4) is False   # iyileşme yok (2) -> dur
    assert stopper.should_stop


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(1.0)
    stopper.step(1.1)
    assert stopper.step(0.9) is True
    assert stopper.counter == 0


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def test_metrics_perfect_prediction():
    y = list(range(8)) * 2
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_avg"]["f1"] == 1.0


def test_metrics_report_includes_absent_classes():
    # Testte hiç örneği olmayan sınıf da raporda görünmeli (F1=0, support=0).
    m = compute_metrics([0, 0, 1], [0, 1, 1])
    assert len(m["per_class"]) == 8
    assert m["per_class"]["van"]["support"] == 0
