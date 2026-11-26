"""
ui.py — Arayüzün HTML/CSS parçaları ve küçük karar kuralları
=============================================================
Streamlit'ten bağımsız, saf Python fonksiyonları. Ayrı dosyada durmalarının
nedeni: pytest ile tarayıcı açmadan test edilebilmeleri (tests/test_ui.py).

  • format_percent  -> 0.9558 -> "%95,6"  (Türkçe yazım: % önde, ondalık virgül)
  • is_uncertain    -> model iki sınıf arasında mı kaldı?
  • plate_html      -> tahmini plaka biçiminde gösteren blok (sayfanın imza öğesi)
  • probs_html      -> 8 sınıfın olasılıkları, büyükten küçüğe yatay çubuklar
  • CSS             -> sayfaya bir kez eklenen stil
"""

from __future__ import annotations

from html import escape

# Kararsızlık eşikleri: en yüksek olasılık %60'ın altındaysa YA DA ilk iki sınıf
# arasındaki fark 25 puandan azsa kullanıcıya "model emin değil" notu gösterilir.
# (Test kümesinde en çok HATCHBACK <-> SEDAN karışıyor; bu not o durumu dürüstçe söyler.)
MIN_CONFIDENCE = 0.60
MIN_MARGIN = 0.25


def format_percent(p: float, digits: int = 1) -> str:
    """0-1 arası olasılığı Türkçe yüzde yazımına çevirir: 0.9558 -> '%95,6'."""
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


def plate_html(label: str, confidence: float) -> str:
    """Tahmini plaka gibi çizer: solda mavi şeritte güven, sağda gövde tipi."""
    return (
        '<div class="plate" aria-label="Tahmin: {label}, güven {conf}">'
        '<div class="plate-band"><span class="plate-conf">{conf}</span>'
        '<span class="plate-conf-label">güven</span></div>'
        '<div class="plate-text">{label}</div>'
        "</div>"
    ).format(label=escape(label), conf=format_percent(confidence, 0))


def probs_html(probabilities: dict[str, float], pred_label: str) -> str:
    """Olasılık dağılımını sıralı yatay çubuk listesi olarak üretir.

    Renk tek başına bilgi taşımasın diye tahmin edilen satır hem mavi hem kalın;
    her satırın değeri de yazıyla yanında durur.
    """
    rows = []
    for label, p in ranked(probabilities):
        top = " is-top" if label == pred_label else ""
        rows.append(
            f'<li class="prob{top}">'
            f'<span class="prob-name">{escape(label)}</span>'
            f'<span class="prob-track"><span class="prob-fill" style="width:{p * 100:.2f}%"></span></span>'
            f'<span class="prob-val">{format_percent(p)}</span>'
            "</li>"
        )
    return '<ol class="probs" aria-label="Sınıf olasılıkları">' + "".join(rows) + "</ol>"


CSS = """
<style>
:root {
  --ink: #15191F;
  --muted: #5B6472;
  --line: #DCE1E7;
  --card: #FFFFFF;
  --plate-blue: #1F4BA5;
  --plate-blue-dark: #183C86;
  --bar: #C9D0D9;
  --warn-ink: #7A4600;
  --warn-bg: #FFF4DE;
  --warn-line: #F0C980;
}

/* Sayfa genişliği: çok geniş ekranlarda satırlar dağılmasın */
.block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 4rem; }

/* Başlık bölümü */
.intro h1 {
  font-family: "Barlow Condensed", sans-serif;
  font-weight: 700; font-size: clamp(2.2rem, 4vw, 3.1rem);
  line-height: 1.02; letter-spacing: -0.01em; margin: 0 0 .6rem; padding: 0;
  color: var(--ink); text-wrap: balance;
}
.intro p { color: var(--muted); font-size: 1.06rem; line-height: 1.5; max-width: 60ch; margin: 0; }
.facts { display: flex; flex-wrap: wrap; gap: .5rem 2rem; margin: 1.4rem 0 .4rem; padding: 0; }
.facts div { display: flex; flex-direction: column; }
.facts dt { color: var(--muted); font-size: .85rem; }
.facts dd { margin: 0; font-weight: 600; font-size: 1.05rem; font-variant-numeric: tabular-nums; color: var(--ink); }

/* Bölüm başlıkları (h2) */
.section-title {
  font-family: "Barlow Condensed", sans-serif; font-weight: 600;
  font-size: 1.45rem; margin: 0 0 .75rem; color: var(--ink);
}

/* Plaka: sayfanın imza öğesi */
.plate {
  display: flex; align-items: stretch; background: var(--card);
  border: 3px solid var(--ink); border-radius: 10px; overflow: hidden;
  min-height: 96px; margin: .25rem 0 1rem;
  animation: plate-in .32s cubic-bezier(.2,.7,.2,1) both;
}
.plate-band {
  background: var(--plate-blue); color: #fff; min-width: 88px; padding: .5rem .75rem;
  display: flex; flex-direction: column; justify-content: center; align-items: center; gap: .1rem;
}
.plate-conf { font-family: "Barlow Condensed", sans-serif; font-weight: 700; font-size: 1.6rem;
  line-height: 1; font-variant-numeric: tabular-nums; }
.plate-conf-label { font-size: .8rem; opacity: .9; }
.plate-text {
  flex: 1; display: flex; align-items: center; padding: .25rem 1.25rem;
  font-family: "Barlow Condensed", sans-serif; font-weight: 700;
  font-size: clamp(2rem, 5vw, 3.2rem); letter-spacing: .04em; line-height: 1.05; color: var(--ink);
  word-break: break-word;
}
@keyframes plate-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

.meta { color: var(--muted); font-size: .95rem; margin: 0 0 1.25rem; font-variant-numeric: tabular-nums; }

/* Kararsızlık notu */
.note {
  background: var(--warn-bg); border: 1px solid var(--warn-line); color: var(--warn-ink);
  border-radius: 8px; padding: .8rem 1rem; font-size: .98rem; line-height: 1.45; margin: 0 0 1.25rem;
}

/* Streamlit markdown listelere girinti verir; bizim listeler başlıkla aynı kenarda başlasın */
.probs, .types { padding-left: 0 !important; margin-left: 0 !important; }
.probs li, .types li { margin: 0 !important; padding-left: 0; }

/* Olasılık çubukları */
.probs { list-style: none; margin: 0; padding: 0; display: grid; gap: .55rem; }
.prob { display: grid; grid-template-columns: 9.5rem 1fr 4.2rem; align-items: center; gap: .75rem; }
.prob-name { font-size: .95rem; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prob-track { height: 10px; background: #EDF0F3; border-radius: 5px; overflow: hidden; }
.prob-fill { display: block; height: 100%; background: var(--bar); border-radius: 5px;
  transform-origin: left; animation: grow .45s cubic-bezier(.2,.7,.2,1) both; }
.prob-val { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); font-size: .95rem; }
.prob.is-top .prob-name, .prob.is-top .prob-val { font-weight: 600; color: var(--ink); }
.prob.is-top .prob-fill { background: var(--plate-blue); }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* Boş durum: modelin tanıdığı 8 tip */
.empty p { color: var(--muted); margin: 0 0 1rem; line-height: 1.5; }
.types { display: flex; flex-wrap: wrap; gap: .5rem; list-style: none; margin: 0; padding: 0; }
.types li { border: 1px solid var(--line); background: var(--card); border-radius: 6px;
  padding: .35rem .7rem !important; font-size: .92rem; color: var(--ink); }

/* Ana buton: üzerine gelince koyulaşır, basınca hafifçe çöker */
.stButton > button[kind="primary"] {
  min-height: 48px; font-weight: 600; font-size: 1.02rem;
  transition: background-color .18s ease, transform .08s ease;
}
.stButton > button[kind="primary"]:hover { background-color: var(--plate-blue-dark); border-color: var(--plate-blue-dark); }
.stButton > button[kind="primary"]:active { transform: scale(.98); }
.stButton > button:focus-visible { outline: 3px solid #8DB0F0; outline-offset: 2px; }

/* Yüklenen görsel: kenarlık ve yumuşak köşe */
[data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--line); }

/* Telefon: çubuk satırında isim sütunu daralsın */
@media (max-width: 640px) {
  .block-container { padding-top: 1.5rem; }
  .prob { grid-template-columns: 7.5rem 1fr 3.8rem; gap: .5rem; }
  .plate-band { min-width: 72px; }
}

/* Hareket azaltma tercihi: tüm animasyonları kapat */
@media (prefers-reduced-motion: reduce) {
  .plate, .prob-fill { animation: none; }
  .stButton > button { transition: none; }
}
</style>
"""
