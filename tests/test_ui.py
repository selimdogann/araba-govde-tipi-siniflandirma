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
    assert ui.format_percent(0.9558) == "%95,6"
    assert ui.format_percent(0.9558, 0) == "%96"
    assert ui.format_percent(0.0) == "%0,0"


def test_uncertain_when_two_classes_close():
    assert ui.is_uncertain({"SEDAN": 0.48, "HATCHBACK": 0.41, "SUV": 0.11})


def test_uncertain_when_confidence_low():
    assert ui.is_uncertain({"SEDAN": 0.55, "HATCHBACK": 0.10, "SUV": 0.35 - 0.10, "VAN": 0.10})


def test_certain_prediction():
    assert not ui.is_uncertain({"SEDAN": 0.95, "HATCHBACK": 0.03, "SUV": 0.02})


def test_probs_html_sorted_and_marks_top():
    html = ui.probs_html({"VAN": 0.1, "SEDAN": 0.7, "SUV": 0.2}, "SEDAN")
    assert html.index("SEDAN") < html.index("SUV") < html.index("VAN")
    assert html.count("is-top") == 1


def test_html_is_escaped():
    # Etiketler checkpoint dosyasından gelir; HTML olarak yorumlanmamalı.
    assert "<script>" not in ui.plate_html("<script>x</script>", 0.9)
    assert "<b>" not in ui.probs_html({"<b>X</b>": 1.0}, "<b>X</b>")


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


def test_app_with_model_shows_empty_state(monkeypatch, dummy_checkpoint):
    at = _run_app(monkeypatch, dummy_checkpoint)
    assert not at.exception
    assert not at.error
    page = " ".join(m.value for m in at.markdown)
    assert "Model şu tipleri tanır" in page
    assert "AÇIK TEKERLEKLİ" in page
    # Fotoğraf yokken Tahmin Yap butonu pasif olmalı.
    assert at.button[0].label == "Tahmin Yap"
    assert at.button[0].disabled
