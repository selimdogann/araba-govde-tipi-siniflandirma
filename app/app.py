"""
app.py  —  Araba Gövde Tipi Sınıflandırma Web Arayüzü (Streamlit)
================================================================
Proje şartlarını karşılayan canlı tahmin arayüzü:

  Bölüm 1: Görüntü yükleme alanı (sürükle-bırak) + önizleme.
  Bölüm 2: "Tahmin Yap" butonu.
  Bölüm 3: Tahmin edilen sınıf (büyük, plaka biçiminde) + güven skoru + süre.
  Bölüm 4: 8 sınıf için olasılık dağılımı (sıralı yatay çubuklar).
           Yüklenen görsel ile sonuç YAN YANA gösterilir (telefonda alt alta).

Hız: Model @st.cache_resource ile yalnızca BİR KEZ yüklenir; sonraki tahminler hızlıdır.
Görünüm: renkler/yazı tipi .streamlit/config.toml'da, sayfa stilleri app/ui.py'de.

Çalıştırma (proje kök dizininden):
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Streamlit scriptinden src paketini import edebilmek için proje kökünü yola ekle.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    page_icon=":material/directions_car:",  # emoji yerine Material ikon
    layout="wide",  # geniş düzen: görsel ve sonuç yan yana sığsın
)
st.markdown(ui.CSS, unsafe_allow_html=True)

CHECKPOINT_PATH = config.CHECKPOINT_PATH
METRICS_PATH = config.OUTPUTS_DIR / "metrics.json"


# ---------------------------------------------------------------------------
# Model ve metrik yükleme (önbelleğe alınır -> her etkileşimde yeniden yüklenmez)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor...")
def load_predictor() -> Predictor:
    """Eğitilmiş modeli tek seferlik yükler. CPU = tek görüntüde düşük gecikme."""
    return Predictor(checkpoint_path=CHECKPOINT_PATH, device="cpu")


@st.cache_data
def load_test_metrics() -> dict | None:
    """evaluate.py'nin yazdığı test metriklerini okur (yoksa None)."""
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


ARCH_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "resnet18": "ResNet-18",
}


def facts_html(predictor: Predictor | None) -> str:
    """Başlığın altındaki kısa model bilgileri (değerler dosyalardan okunur, elle yazılmaz)."""
    items: list[tuple[str, str]] = []
    metrics = load_test_metrics()
    if metrics:
        support = sum(c["support"] for c in metrics["per_class"].values())
        items.append(("Test kümesinde macro F1", f"{metrics['macro_avg']['f1']:.3f}".replace(".", ",")))
        items.append(("Test görüntüsü", f"{support:,}".replace(",", ".")))
    if predictor is not None:
        items.append(("Model", ARCH_NAMES.get(predictor.arch, predictor.arch)))
        size_mb = CHECKPOINT_PATH.stat().st_size / 1024 ** 2
        items.append(("Dosya boyutu", f"{size_mb:.0f} MB"))
    cells = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in items)
    return f'<dl class="facts">{cells}</dl>' if items else ""


# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
model_ready = CHECKPOINT_PATH.exists()
predictor = load_predictor() if model_ready else None

st.markdown(
    '<div class="intro">'
    "<h1>Araba fotoğrafından gövde tipi</h1>"
    "<p>Bir araba fotoğrafı yükleyin. Model onu 8 gövde tipinden birine ayırır ve "
    "her tip için ne kadar emin olduğunu gösterir.</p>"
    f"{facts_html(predictor)}"
    "</div>",
    unsafe_allow_html=True,
)

# Model dosyası yoksa ne yapılacağını adım adım söyle ve dur.
if not model_ready:
    st.error(
        f"Model dosyası bulunamadı: `{CHECKPOINT_PATH}`\n\n"
        "Modeli eğitin (`python train.py`) ya da hazır `best_model.pt` dosyasını bu yola "
        "koyun. Dosya başka bir yerdeyse `MODEL_PATH=/yol/best_model.pt streamlit run app/app.py` "
        "ile başlatın."
    )
    st.stop()

st.write("")  # başlık ile çalışma alanı arasında nefes payı

# ---------------------------------------------------------------------------
# Yan yana düzen: solda yükleme + önizleme, sağda sonuç
# ---------------------------------------------------------------------------
col_img, col_result = st.columns([5, 6], gap="large")

with col_img:
    st.markdown('<h2 class="section-title">Fotoğraf</h2>', unsafe_allow_html=True)
    # Bölüm 1: Görüntü yükleme alanı (sürükle-bırak) + önizleme.
    uploaded = st.file_uploader(
        "Araba fotoğrafı (JPG, PNG, BMP veya WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            image.load()  # bozuk dosyayı burada yakala, tahmin anında değil
        except (UnidentifiedImageError, OSError):
            st.error("Bu dosya açılamadı. Bozuk olabilir; başka bir fotoğraf deneyin.")
            image = None

    if image is not None:
        st.image(image, width="stretch", alt="Yüklenen araba fotoğrafı")

    # Bölüm 2: Tahmin butonu (fotoğraf yokken pasif).
    clicked = st.button("Tahmin Yap", type="primary", width="stretch", disabled=image is None)

# Sonucu oturumda sakla: aynı fotoğraf için sayfa yeniden çizilse de kaybolmasın,
# yeni fotoğraf yüklenince eski sonuç temizlensin.
file_id = uploaded.file_id if uploaded is not None else None
if st.session_state.get("result_for") != file_id:
    st.session_state.pop("result", None)
if clicked and image is not None:
    st.session_state["result"] = predictor.predict(image)
    st.session_state["result_for"] = file_id
    # Telefonda sonuç sütunu butonun altında kalır; kısa bildirim tahminin
    # hazır olduğunu ekranın neresinde olunursa olunsun gösterir.
    r = st.session_state["result"]
    st.toast(f"Tahmin: {r['pred_label']} ({ui.format_percent(r['confidence'], 0)})")

with col_result:
    st.markdown('<h2 class="section-title">Sonuç</h2>', unsafe_allow_html=True)
    result = st.session_state.get("result")

    if result is None:
        # Boş durum: ne yapılacağını ve modelin hangi tipleri tanıdığını söyle.
        hint = ("Fotoğraf hazır. Tahmin Yap'a basın." if image is not None
                else "Bir fotoğraf yükleyin, sonra Tahmin Yap'a basın.")
        types = "".join(f"<li>{label}</li>" for label in predictor.display_labels)
        st.markdown(
            f'<div class="empty"><p>{hint} Model şu tipleri tanır:</p>'
            f'<ul class="types">{types}</ul></div>',
            unsafe_allow_html=True,
        )
    else:
        probs = result["probabilities"]
        # Bölüm 3: Büyük ve belirgin tahmin + güven skoru (plaka biçiminde).
        st.markdown(ui.plate_html(result["pred_label"], result["confidence"]), unsafe_allow_html=True)
        st.markdown(
            f'<p class="meta">Güven skoru {ui.format_percent(result["confidence"])}, '
            f'tahmin {result["inference_ms"]:.0f} ms sürdü.</p>',
            unsafe_allow_html=True,
        )
        # Model iki tip arasında kaldıysa bunu açıkça söyle.
        if ui.is_uncertain(probs):
            (a, pa), (b, pb) = ui.ranked(probs)[:2]
            st.markdown(
                f'<div class="note">Model {a} ({ui.format_percent(pa, 0)}) ile '
                f'{b} ({ui.format_percent(pb, 0)}) arasında kaldı. Arabanın tamamının '
                "yandan ya da önden çapraz göründüğü bir fotoğraf daha net sonuç verir.</div>",
                unsafe_allow_html=True,
            )
        # Bölüm 4: 8 sınıfın olasılık dağılımı.
        st.markdown('<h2 class="section-title">Tüm tipler</h2>', unsafe_allow_html=True)
        st.markdown(ui.probs_html(probs, result["pred_label"]), unsafe_allow_html=True)
