"""
app.py  —  Araba Gövde Tipi Sınıflandırma Web Arayüzü (Streamlit)
================================================================
Proje şartlarını karşılayan canlı tahmin arayüzü:

  Bölüm 1: Görüntü yükleme alanı (drag & drop) + önizleme.
  Bölüm 2: "Tahmin Yap" butonu.
  Bölüm 3: Tahmin edilen sınıf (büyük) + güven skoru.
  Bölüm 4: 8 sınıf için olasılık dağılımı çubuk grafiği (bar chart).
           Yüklenen görsel ile sonuç YAN YANA gösterilir.

Hız: Model @st.cache_resource ile yalnızca BİR KEZ yüklenir; sonraki tahminler hızlıdır.

Çalıştırma (proje kök dizininden):
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit scriptinden src paketini import edebilmek için proje kökünü yola ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from src import config
from src.inference import Predictor

# ---------------------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Araba Gövde Tipi Sınıflandırma",
    page_icon="🚗",
    layout="wide",  # geniş düzen: görsel ve sonuç yan yana sığsın
)

CHECKPOINT_PATH = config.CHECKPOINTS_DIR / "best_model.pt"


# ---------------------------------------------------------------------------
# Model yükleme (önbelleğe alınır -> her etkileşimde yeniden yüklenmez)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor...")
def load_predictor() -> Predictor:
    """Eğitilmiş modeli tek seferlik yükler. CPU = tek görüntüde düşük gecikme."""
    return Predictor(checkpoint_path=CHECKPOINT_PATH, device="cpu")


def probability_bar_chart(probabilities: dict[str, float], pred_label: str):
    """8 sınıfın olasılık dağılımını yatay çubuk grafik olarak çizer (tahmin vurgulu)."""
    # Olasılığa göre artan sırala (en yüksek en üstte görünsün diye barh ile).
    items = sorted(probabilities.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v * 100 for _, v in items]  # yüzdeye çevir
    # Tahmin edilen sınıfın çubuğunu farklı renkle vurgula.
    colors = ["#DD8452" if k == pred_label else "#4C72B0" for k in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Olasılık (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Sınıf Olasılık Dağılımı")
    # Her çubuğun ucuna yüzde değerini yaz.
    for bar, v in zip(bars, values):
        ax.text(min(v + 1, 96), bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
st.title("🚗 Araba Gövde Tipi Sınıflandırma")
st.caption("Bir araba görseli yükleyin; model 8 gövde tipinden birini tahmin etsin "
           "(SUV, VAN, STATION WAGON, MICRO, AÇIK TEKERLEKLİ, SEDAN, HATCHBACK, PICK UP).")

# Model dosyası yoksa kullanıcıyı bilgilendir.
if not CHECKPOINT_PATH.exists():
    st.error(f"Eğitilmiş model bulunamadı: `{CHECKPOINT_PATH}`\n\n"
             "Önce terminalden `python train.py` ile modeli eğitin.")
    st.stop()

predictor = load_predictor()

# ---------------------------------------------------------------------------
# Bölüm 1: Görüntü yükleme alanı (drag & drop) + önizleme
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Araba görseli yükleyin (sürükle-bırak veya dosya seç)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is not None:
    image = Image.open(uploaded)

    # Yüklenen görsel (sol) ve sonuçlar (sağ) YAN YANA.
    col_img, col_result = st.columns(2)

    with col_img:
        st.subheader("Yüklenen Görsel")
        st.image(image, use_container_width=True)

    with col_result:
        st.subheader("Tahmin Sonucu")
        # Bölüm 2: Tahmin butonu.
        if st.button("🔍 Tahmin Yap", type="primary", use_container_width=True):
            result = predictor.predict(image)

            # Bölüm 3: Büyük ve belirgin tahmin + güven skoru.
            st.markdown(f"## ➜ {result['pred_label']}")
            st.metric(label="Güven Skoru (Confidence)",
                      value=f"{result['confidence'] * 100:.1f}%")
            st.caption(f"Çıkarım süresi: {result['inference_ms']:.1f} ms")

            # Bölüm 4: Olasılık dağılımı çubuk grafiği.
            st.pyplot(probability_bar_chart(result["probabilities"], result["pred_label"]))
        else:
            st.info("Görsel yüklendi. Tahmin için **Tahmin Yap** butonuna basın.")
else:
    st.info("Başlamak için yukarıdan bir araba görseli yükleyin.")
