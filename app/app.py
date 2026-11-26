"""
app.py  —  Araba Gövde Tipi Sınıflandırma Web Arayüzü (Streamlit)
================================================================
Proje şartlarını karşılayan canlı tahmin arayüzü:

  Bölüm 1: Görüntü yükleme alanı (sürükle-bırak) + önizleme.
  Bölüm 2: "Sınıflandır" butonu.
  Bölüm 3: Tahmin edilen sınıf (büyük) + güven skoru + süre + ilk 3 sınıf tablosu.
  Bölüm 4: 8 sınıf için olasılık dağılımı çubuk grafiği.
           Yüklenen görsel ile sonuç YAN YANA gösterilir.

Hız: Model @st.cache_resource ile yalnızca BİR KEZ yüklenir; sonraki tahminler hızlıdır.
Tema (koyu, kırmızı buton): .streamlit/config.toml

Çalıştırma (proje kök dizininden):
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit scriptinden src paketini import edebilmek için proje kökünü yola ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from app import ui
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

CHECKPOINT_PATH = config.CHECKPOINT_PATH


# ---------------------------------------------------------------------------
# Model yükleme (önbelleğe alınır -> her etkileşimde yeniden yüklenmez)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor...")
def load_predictor() -> Predictor:
    """Eğitilmiş modeli tek seferlik yükler. CPU = tek görüntüde düşük gecikme."""
    return Predictor(checkpoint_path=CHECKPOINT_PATH, device="cpu")


# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
st.title("🚗 Araba Gövde Tipi Sınıflandırma")

# Model dosyası yoksa ne yapılacağını söyle ve dur.
if not CHECKPOINT_PATH.exists():
    st.error(
        f"Model dosyası bulunamadı: `{CHECKPOINT_PATH}`\n\n"
        "Modeli eğitin (`python train.py`) ya da hazır `best_model.pt` dosyasını bu yola "
        "koyun. Dosya başka bir yerdeyse `MODEL_PATH=/yol/best_model.pt streamlit run app/app.py` "
        "ile başlatın."
    )
    st.stop()

predictor = load_predictor()
# Model adı ve sınıflar checkpoint'ten okunur (elle yazılmaz -> her zaman doğru).
st.caption(f"{predictor.arch} tabanlı model · {len(predictor.display_labels)} sınıf: "
           + ", ".join(predictor.display_labels))

# ---------------------------------------------------------------------------
# Bölüm 1: Görüntü yükleme alanı (sürükle-bırak)
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Bir araba görseli", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is None:
    st.info("Başlamak için yukarıdan bir araba görseli yükleyin.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()  # bozuk dosyayı burada yakala, tahmin anında değil
except (UnidentifiedImageError, OSError):
    st.error("Bu dosya açılamadı. Bozuk olabilir; başka bir görsel deneyin.")
    st.stop()

# Sonucu oturumda sakla: aynı görsel için sayfa yeniden çizilse de kaybolmasın,
# yeni görsel yüklenince eski sonuç temizlensin.
if st.session_state.get("result_for") != uploaded.file_id:
    st.session_state.pop("result", None)

# Yüklenen görsel (sol) ve sonuçlar (sağ) YAN YANA.
col_img, col_result = st.columns(2)

with col_img:
    st.subheader("Yüklenen Görüntü")
    st.image(image, width="stretch")

with col_result:
    st.subheader("Tahmin Sonucu")
    # Bölüm 2: Tahmin butonu.
    if st.button("🔍 Sınıflandır", type="primary", width="stretch"):
        st.session_state["result"] = predictor.predict(image)
        st.session_state["result_for"] = uploaded.file_id

    result = st.session_state.get("result")
    if result is not None:
        probs = result["probabilities"]

        # Bölüm 3: Büyük ve belirgin tahmin + güven skoru + süre.
        st.markdown(f"### 🏆 {result['pred_label']}")
        st.metric("Güven Skoru", ui.format_percent(result["confidence"]))
        st.caption(f"Tahmini süre: {result['inference_ms']:.0f} ms")

        # Model iki tip arasında kaldıysa bunu açıkça söyle (emin olduğu durumda görünmez).
        if ui.is_uncertain(probs):
            (a, pa), (b, pb) = ui.ranked(probs)[:2]
            st.warning(f"Model {a} ({ui.format_percent(pa, 0)}) ile {b} "
                       f"({ui.format_percent(pb, 0)}) arasında kaldı. Arabanın tamamının "
                       "göründüğü başka bir fotoğraf daha net sonuç verir.")

        # En olası 3 sınıf tablosu.
        top3 = pd.DataFrame(
            [(label, ui.format_percent(p)) for label, p in ui.ranked(probs)[:3]],
            columns=["Sınıf", "Olasılık"],
        )
        st.dataframe(top3, hide_index=True, width="stretch")

        # Bölüm 4: 8 sınıfın olasılık dağılımı (0-1 arası).
        st.subheader("Tüm Sınıfların Olasılık Dağılımı")
        st.bar_chart(pd.Series(probs, name="Olasılık"))
