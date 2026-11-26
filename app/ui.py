"""
ui.py — Arayüzün küçük yardımcı fonksiyonları
=============================================
Streamlit'ten bağımsız, saf Python fonksiyonları. Ayrı dosyada durmalarının
nedeni: pytest ile tarayıcı açmadan test edilebilmeleri (tests/test_ui.py).

  • format_percent  -> 0.9558 -> "%95,58"  (Türkçe yazım: % önde, ondalık virgül)
  • ranked          -> olasılıkları büyükten küçüğe sıralar
  • is_uncertain    -> model iki sınıf arasında mı kaldı?
"""

from __future__ import annotations

# Kararsızlık eşikleri: en yüksek olasılık %60'ın altındaysa YA DA ilk iki sınıf
# arasındaki fark 25 puandan azsa kullanıcıya "model emin değil" notu gösterilir.
# (Test kümesinde en çok HATCHBACK <-> SEDAN karışıyor; bu not o durumu dürüstçe söyler.)
MIN_CONFIDENCE = 0.60
MIN_MARGIN = 0.25


def format_percent(p: float, digits: int = 2) -> str:
    """0-1 arası olasılığı Türkçe yüzde yazımına çevirir: 0.9558 -> '%95,58'."""
    return f"%{p * 100:.{digits}f}".replace(".", ",")


def ranked(probabilities: dict[str, float]) -> list[tuple[str, float]]:
    """Olasılıkları büyükten küçüğe sıralar."""
    return sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)


def is_uncertain(probabilities: dict[str, float]) -> bool:
    """En olası iki sınıf birbirine çok yakınsa ya da güven düşükse True döner."""
    top = ranked(probabilities)
    first = top[0][1]
    second = top[1][1] if len(top) > 1 else 0.0
    return first < MIN_CONFIDENCE or (first - second) < MIN_MARGIN
