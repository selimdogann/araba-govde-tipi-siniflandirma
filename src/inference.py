"""
inference.py
============
Eğitilmiş modelle TEK BİR GÖRÜNTÜ üzerinde tahmin yapan yeniden kullanılabilir modül.

Hem web arayüzü (app/app.py) hem de komut satırı (predict.py) hem de jürinin
sağlayacağı test scripti bu `Predictor` sınıfını import edip kullanabilir.

Hız notu: Tek görüntü çıkarımı için cihaz varsayılan olarak CPU seçilir; küçük bir
modelde (EfficientNet-B0) tek görüntü CPU'da milisaniyeler sürer ve GPU'ya veri
taşıma gecikmesi olmadığı için arayüzde düşük gecikme sağlar.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src import config
from src.data.dataset import get_eval_transforms
from src.models import load_checkpoint_model


class Predictor:
    """Modeli bir kez yükleyip görüntüleri sınıflandıran tahmin sınıfı."""

    def __init__(
        self,
        checkpoint_path: str | Path = config.CHECKPOINTS_DIR / "best_model.pt",
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Args:
            checkpoint_path: best_model.pt yolu.
            device:          "cpu" (varsayılan, düşük gecikme), "mps" veya "cuda".
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {checkpoint_path}\nÖnce 'python train.py' ile eğitin."
            )
        self.device = torch.device(device)
        # Modeli ve meta veriyi yükle (eval moduna alınmış olarak gelir).
        self.model, ckpt = load_checkpoint_model(str(checkpoint_path), self.device)

        # Meta veri: sınıf adları, gösterim etiketleri, görüntü boyutu.
        self.class_names: list[str] = ckpt.get("class_names", config.CLASS_NAMES)
        self.display_labels: list[str] = ckpt.get(
            "display_labels",
            [config.DISPLAY_NAMES.get(c, c) for c in self.class_names],
        )
        self.img_size: int = ckpt.get("img_size", config.IMG_SIZE)
        # Değerlendirme dönüşümü (eğitimdeki val/test ile AYNI: resize+centercrop+normalize).
        self.transform = get_eval_transforms(self.img_size)

    @torch.no_grad()  # tahminde gradyan gerekmez (hız/bellek)
    def predict(self, image: Image.Image) -> dict:
        """
        Bir PIL görüntüsünü sınıflandırır.

        Args:
            image: PIL.Image nesnesi (herhangi bir modda olabilir; RGB'ye çevrilir).

        Returns:
            {
              "pred_index":   int,                 # tahmin edilen sınıf indeksi (0-7)
              "pred_class":   str,                 # klasör adı (ör. "sedan")
              "pred_label":   str,                 # gösterim etiketi (ör. "SEDAN")
              "confidence":   float,               # en yüksek olasılık (0-1)
              "probabilities":{etiket: olasılık},  # 8 sınıf için olasılık dağılımı
              "inference_ms": float,               # çıkarım süresi (ms)
            }
        """
        # Girişi RGB'ye normalize et (palette/gri görüntüleri de güvenle işlemek için).
        image = image.convert("RGB")

        t0 = time.perf_counter()
        # Dönüşümü uygula ve batch boyutu ekle: (3,H,W) -> (1,3,H,W).
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)                 # ham skorlar
        probs = F.softmax(logits, dim=1)[0]         # olasılığa çevir (toplamı 1)
        conf, idx = torch.max(probs, dim=0)         # en yüksek olasılık ve indeksi
        inference_ms = (time.perf_counter() - t0) * 1000.0

        idx = int(idx.item())
        # Olasılık dağılımını gösterim etiketleriyle eşle.
        probabilities = {
            self.display_labels[i]: float(probs[i].item())
            for i in range(len(self.display_labels))
        }
        return {
            "pred_index": idx,
            "pred_class": self.class_names[idx],
            "pred_label": self.display_labels[idx],
            "confidence": float(conf.item()),
            "probabilities": probabilities,
            "inference_ms": inference_ms,
        }

    def predict_path(self, image_path: str | Path) -> dict:
        """Dosya yolundan görüntü okuyup predict() çağırır (test scripti için pratik)."""
        return self.predict(Image.open(image_path))
