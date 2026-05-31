"""
model.py
========
Araba gövde tipi sınıflandırması için MODEL MİMARİSİ kurucu modülü.

Neden TRANSFER LEARNING?
  Veri setimiz ~5000 görüntü; sıfırdan derin bir ağ eğitmek için görece azdır.
  ImageNet (1.2M görüntü) üzerinde önceden eğitilmiş bir ağ; kenar, doku, tekerlek,
  cam gibi genel görsel özellikleri zaten öğrenmiştir. Bu ağırlıkları başlangıç
  noktası alıp sadece son katmanı 8 sınıfımıza göre yeniden eğitmek (fine-tuning)
  hem daha hızlı yakınsar hem de DAHA İYİ GENELLEME yapar.

Neden EfficientNet-B0 (varsayılan)?
  ~5.3M parametre (~20 MB) ile 95 MB sınırının çok altında kalır; "compound scaling"
  sayesinde boyutuna göre yüksek doğruluk verir ve inferansı hızlıdır (arayüzde
  hız puanı için önemli). Kod, aşağıdaki 4 mimariyi de destekler; istenirse
  --arch argümanıyla değiştirilebilir.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# Desteklenen mimariler (hepsi 95 MB sınırının altında).
SUPPORTED_ARCHS = ("efficientnet_b0", "mobilenet_v3_large", "mobilenet_v3_small", "resnet18")


def build_model(
    arch: str = "efficientnet_b0",
    num_classes: int = 8,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    """
    Seçilen mimariyi kurup SON KATMANINI (sınıflandırıcı head) 8 sınıfa uyarlar.

    Args:
        arch:        Mimari adı (SUPPORTED_ARCHS içinden).
        num_classes: Çıkış sınıfı sayısı (projede 8).
        dropout:     Sınıflandırıcı head'indeki dropout oranı (overfitting'i azaltır).
        pretrained:  True ise ImageNet ön-eğitimli ağırlıklar yüklenir (transfer learning).

    Returns:
        Eğitime hazır torch.nn.Module.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Desteklenmeyen mimari: {arch}. Seçenekler: {SUPPORTED_ARCHS}")

    # "DEFAULT" -> torchvision'ın o mimari için en güncel ImageNet ağırlıkları.
    weights = "DEFAULT" if pretrained else None

    # -----------------------------------------------------------------------
    # EfficientNet-B0
    # -----------------------------------------------------------------------
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        # Orijinal classifier: Sequential(Dropout, Linear(1280 -> 1000)).
        # Linear'ın giriş boyutunu (1280) alıp head'i yeniden kuruyoruz.
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),     # rastgele nöron söndürme (regularization)
            nn.Linear(in_features, num_classes),     # 8 sınıfa indirgeyen tam bağlı katman
        )

    # -----------------------------------------------------------------------
    # MobileNetV3-Large / Small
    # -----------------------------------------------------------------------
    elif arch in ("mobilenet_v3_large", "mobilenet_v3_small"):
        model = getattr(models, arch)(weights=weights)
        # MobileNetV3 classifier: Sequential(Linear, Hardswish, Dropout, Linear -> 1000).
        # 2. indeksteki Dropout'u kendi oranımızla, 3. indeksteki son Linear'ı 8 çıkışla değiştir.
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        model.classifier[3] = nn.Linear(in_features, num_classes)

    # -----------------------------------------------------------------------
    # ResNet18
    # -----------------------------------------------------------------------
    elif arch == "resnet18":
        model = models.resnet18(weights=weights)
        # ResNet'in son tam bağlı katmanı "fc". Önüne dropout ekleyip 8 sınıfa indir.
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    return model


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    """
    Modelin "gövde" (backbone) katmanlarını dondurur (freeze) veya çözer (unfreeze).

    Transfer learning'de yaygın strateji: önce SADECE yeni eklenen sınıflandırıcı
    head'i eğitmek (backbone dondurulur), sonra tüm ağı küçük bir öğrenme oranıyla
    ince ayar (fine-tune) yapmak. Bu fonksiyon head HARİÇ tüm parametreleri kontrol eder.

    Args:
        model:     build_model ile üretilmiş model.
        arch:      Mimari adı (head'in adını bilmek için).
        trainable: False -> backbone dondurulur; True -> backbone eğitilebilir olur.
    """
    # Her mimaride sınıflandırıcı head'in nitelik (attribute) adı farklıdır.
    head_name = "fc" if arch == "resnet18" else "classifier"
    for name, param in model.named_parameters():
        # İsmi head_name ile başlamayan tüm parametreler "backbone" sayılır.
        if not name.startswith(head_name):
            param.requires_grad = trainable


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Modelin toplam ve eğitilebilir parametre sayısını döndürür.

    Returns:
        (toplam_parametre, egitilebilir_parametre)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint_model(
    ckpt_path: str,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Eğitimde kaydedilen checkpoint'ten modeli YENİDEN KURAR ve ağırlıkları yükler.

    train.py, modeli `state_dict` + meta veri (arch, class_names, img_size ...) olarak
    kaydeder. Bu fonksiyon o meta veriyi okuyup aynı mimariyi kurar, ağırlıkları yükler
    ve modeli DEĞERLENDİRME (eval) moduna alır. evaluate.py ve web arayüzü bunu kullanır.

    Returns:
        (model, checkpoint_dict). checkpoint_dict; class_names, img_size, val_f1 gibi
        meta veriyi içerir.
    """
    # weights_only=False -> kaydedilen sözlükteki meta veriyi de okuyabilmek için.
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(
        arch=checkpoint["arch"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint.get("dropout", 0.0),
        pretrained=False,  # ağırlıkları zaten checkpoint'ten yükleyeceğiz
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()  # inferans için: dropout kapalı, BatchNorm sabit
    return model, checkpoint


def estimate_model_size_mb(model: nn.Module) -> float:
    """
    Modelin diske kaydedildiğinde yaklaşık kaç MB tutacağını tahmin eder.

    Her parametre + buffer'ın byte boyutunu toplayıp MB'a çevirir. Proje sınırı
    95 MB olduğu için bu değeri eğitim öncesi kontrol etmek önemlidir.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)
