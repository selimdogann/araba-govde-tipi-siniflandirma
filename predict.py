"""
predict.py
==========
Komut satırından TEK bir görüntü için tahmin döndüren basit script.

Jürinin sağlayacağı test scriptiyle entegrasyonu kolaylaştırır: bir görüntü yolu
verilir, tahmin edilen kategori (ve istenirse JSON çıktısı) yazdırılır.

Kullanım:
    python predict.py yol/araba.jpg              # sadece tahmin edilen kategoriyi yazar
    python predict.py yol/araba.jpg --json       # tüm olasılıkları JSON olarak yazar
"""

from __future__ import annotations

import argparse
import json

from src import config
from src.inference import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek görüntü tahmini")
    parser.add_argument("image", type=str, help="Sınıflandırılacak görüntü dosyası yolu")
    parser.add_argument("--checkpoint", type=str,
                        default=str(config.CHECKPOINTS_DIR / "best_model.pt"))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--json", action="store_true", help="Tüm olasılıkları JSON olarak yazdır")
    args = parser.parse_args()

    predictor = Predictor(checkpoint_path=args.checkpoint, device=args.device)
    result = predictor.predict_path(args.image)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # Test scriptlerinin kolayca ayrıştırabilmesi için sade çıktı.
        print(f"{result['pred_label']}  (güven: {result['confidence']*100:.1f}%, "
              f"{result['inference_ms']:.1f} ms)")


if __name__ == "__main__":
    main()
