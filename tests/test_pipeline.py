"""
test_pipeline.py
================
Uçtan uca testler: tahmin API'si (Predictor), predict.py komut satırı,
train.py --smoke-test ve evaluate.py. Gerçek veri/model yerine conftest.py'deki
minik sahte veri seti ve rastgele ağırlıklı model kullanılır.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from src import config
from src.inference import Predictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_script(*args: str) -> subprocess.CompletedProcess:
    """Proje kökünden bir python scripti çalıştırır ve çıktısını döndürür."""
    return subprocess.run(
        [sys.executable, *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600,
    )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def predictor(dummy_checkpoint) -> Predictor:
    return Predictor(checkpoint_path=dummy_checkpoint)


def test_predict_returns_valid_distribution(predictor):
    result = predictor.predict(Image.new("RGB", (320, 240), (90, 90, 90)))
    probs = result["probabilities"]
    assert list(probs) == config.DISPLAY_LABELS
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert result["pred_label"] == max(probs, key=probs.get)
    assert result["confidence"] == pytest.approx(max(probs.values()))
    assert result["pred_class"] == config.CLASS_NAMES[result["pred_index"]]
    assert result["inference_ms"] > 0


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_predict_accepts_any_color_mode(predictor, mode):
    result = predictor.predict(Image.new(mode, (100, 80)))
    assert result["pred_label"] in config.DISPLAY_LABELS


def test_predict_respects_exif_orientation(predictor, tmp_path, monkeypatch):
    # Telefon fotoğrafı simülasyonu: pikseller yan yatık, EXIF "90° döndür" diyor.
    # Doğru okunan görüntü ile aynı tahmini (aynı olasılıkları) vermeli.
    upright = Image.new("RGB", (300, 200))
    upright.paste((255, 0, 0), (0, 0, 300, 100))  # üst yarı kırmızı, alt yarı siyah
    sideways = upright.rotate(-90, expand=True)     # saat yönünde 90° yatırılmış pikseller
    exif = sideways.getexif()
    exif[0x0112] = 8                                # Orientation=8: göstermeden önce 90° sola çevir
    path = tmp_path / "phone.jpg"
    sideways.save(path, exif=exif, quality=100)

    with Image.open(path) as img:
        assert img.size == (200, 300)  # dosyadaki ham piksel yönü dik değil

    # Modele giden görüntüyü yakala: rastgele ağırlıklı model çevrilmiş/çevrilmemiş
    # görüntüye benzer cevap verebileceği için olasılıklara değil girdiye bakıyoruz.
    seen = []
    original = predictor.transform

    def spy(image):
        seen.append(image.copy())
        return original(image)

    monkeypatch.setattr(predictor, "transform", spy)
    predictor.predict_path(path)
    fixed = seen[0]
    assert fixed.size == (300, 200)                    # yatay (doğru) yön
    assert fixed.getpixel((150, 20))[0] > 200          # üst taraf kırmızı
    assert fixed.getpixel((150, 180))[0] < 50          # alt taraf siyah


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(checkpoint_path=tmp_path / "yok.pt")


# ---------------------------------------------------------------------------
# predict.py (komut satırı)
# ---------------------------------------------------------------------------
def test_predict_cli_json(dummy_checkpoint, tmp_path):
    img = tmp_path / "car.png"
    Image.new("RGB", (120, 90), (30, 60, 90)).save(img)
    proc = run_script("predict.py", str(img), "--checkpoint", str(dummy_checkpoint), "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["pred_label"] in config.DISPLAY_LABELS


def test_predict_cli_missing_image(dummy_checkpoint, tmp_path):
    proc = run_script("predict.py", str(tmp_path / "yok.jpg"), "--checkpoint", str(dummy_checkpoint))
    assert proc.returncode == 1
    assert "görüntü bulunamadı" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_predict_cli_not_an_image(dummy_checkpoint, tmp_path):
    bad = tmp_path / "not_image.jpg"
    bad.write_text("bu bir görüntü değil")
    proc = run_script("predict.py", str(bad), "--checkpoint", str(dummy_checkpoint))
    assert proc.returncode == 1
    assert "bozuk" in proc.stderr


# ---------------------------------------------------------------------------
# train.py --smoke-test  +  evaluate.py
# ---------------------------------------------------------------------------
def test_train_then_evaluate(tiny_dataset, tmp_path):
    out = tmp_path / "outputs"
    train = run_script(
        "train.py", "--smoke-test", "--no-pretrained", "--arch", "mobilenet_v3_small",
        "--data-dir", str(tiny_dataset), "--output-dir", str(out),
        "--num-workers", "0", "--batch-size", "8",
    )
    assert train.returncode == 0, train.stdout[-2000:] + train.stderr[-2000:]
    ckpt = out / "checkpoints" / "best_model.pt"
    history = json.loads((out / "logs" / "history.json").read_text())
    assert len(history["epoch"]) == 2  # smoke-test = 2 epoch

    # Rastgele model val'de F1=0 alabilir; EarlyStopping hiç "iyileşme" görmezse
    # checkpoint yazılmaz. Bu durumda evaluate kısmını atla (train yine doğrulandı).
    if not ckpt.exists():
        pytest.skip("Smoke-test modeli hiç iyileşme görmedi, checkpoint yazılmadı")

    ev = run_script(
        "evaluate.py", "--checkpoint", str(ckpt), "--data-dir", str(tiny_dataset),
        "--output-dir", str(out), "--split", "val",
    )
    assert ev.returncode == 0, ev.stdout[-2000:] + ev.stderr[-2000:]
    metrics = json.loads((out / "metrics.json").read_text())
    assert 0.0 <= metrics["macro_avg"]["f1"] <= 1.0
    for name in ("loss_curve.png", "accuracy_curve.png", "confusion_matrix.png"):
        assert (out / "figures" / name).exists(), name
