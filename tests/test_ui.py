"""
test_ui.py
==========
Arayüz testleri: app/ui.py yardımcıları + Streamlit AppTest ile sayfanın
tarayıcı açmadan çalıştırılması (model var / model yok durumları).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import ui

APP_FILE = str(Path(__file__).resolve().parent.parent / "app" / "app.py")


# ---------------------------------------------------------------------------
# ui.py yardımcıları
# ---------------------------------------------------------------------------
def test_format_percent_turkish():
    assert ui.format_percent(0.9558) == "%95,58"
    assert ui.format_percent(0.9558, 0) == "%96"
    assert ui.format_percent(0.0) == "%0,00"


def test_uncertain_when_two_classes_close():
    assert ui.is_uncertain({"SEDAN": 0.48, "HATCHBACK": 0.41, "SUV": 0.11})


def test_uncertain_when_confidence_low():
    assert ui.is_uncertain({"SEDAN": 0.55, "HATCHBACK": 0.10, "SUV": 0.25, "VAN": 0.10})


def test_certain_prediction():
    assert not ui.is_uncertain({"SEDAN": 0.95, "HATCHBACK": 0.03, "SUV": 0.02})


def test_ranked_sorts_descending():
    assert [k for k, _ in ui.ranked({"VAN": 0.1, "SEDAN": 0.7, "SUV": 0.2})] == ["SEDAN", "SUV", "VAN"]


# ---------------------------------------------------------------------------
# Sayfanın kendisi (AppTest)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_config():
    # Testten sonra config'i gerçek ortam değişkenleriyle yeniden yükle (diğer testler etkilenmesin).
    yield
    import src.config
    importlib.reload(src.config)


def _run_app(monkeypatch, model_path: Path) -> AppTest:
    # config.CHECKPOINT_PATH, MODEL_PATH ortam değişkeninden okunur.
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    import src.config
    importlib.reload(src.config)
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    return at


def test_app_without_model_shows_clear_error(monkeypatch, tmp_path):
    at = _run_app(monkeypatch, tmp_path / "yok.pt")
    assert not at.exception
    assert len(at.error) == 1
    assert "Model dosyası bulunamadı" in at.error[0].value


def test_app_with_model_waits_for_upload(monkeypatch, dummy_checkpoint):
    at = _run_app(monkeypatch, dummy_checkpoint)
    assert not at.exception
    assert not at.error
    # Başlık altındaki satır model adını ve 8 sınıfı checkpoint'ten okur.
    assert "mobilenet_v3_small tabanlı model" in at.caption[0].value
    assert "AÇIK TEKERLEKLİ" in at.caption[0].value
    assert "araba görseli yükleyin" in at.info[0].value
