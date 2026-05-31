"""
dataset.py
==========
8 sınıflı araba gövde tipi veri setini okuyup modele hazır hale getiren VERİ HATTI.

Sorumlulukları:
  1) Eğitim ve değerlendirme için görüntü dönüşümlerini (transform) tanımlamak:
       - Eğitim    -> boyutlandırma + normalizasyon + GELİŞMİŞ veri çoğaltma (augmentation)
       - Val/Test  -> sadece boyutlandırma + normalizasyon (çoğaltma YOK)
  2) Veriyi sınıf dağılımını koruyarak (stratified) train/val/test olarak bölmek.
  3) PyTorch DataLoader'ları üretmek.

Önemli tasarım kararı: Eğitim verisine veri çoğaltma uygulanır ama val/test verisine
UYGULANMAZ. Bunu sağlamak için aynı klasörü iki ayrı ImageFolder ile (biri eğitim,
biri değerlendirme dönüşümüyle) açıp, indeksleri Subset ile bölüyoruz.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src import config

# Bazı görüntü dosyaları eksik/bozuk byte içerebilir; bu satır PIL'in böyle
# dosyaları hata fırlatmadan (mümkün olduğunca) okumasını sağlar.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ===========================================================================
# 1) GÖRÜNTÜ DÖNÜŞÜMLERİ (TRANSFORMS)
# ===========================================================================
def get_train_transforms(img_size: int = config.IMG_SIZE) -> transforms.Compose:
    """
    EĞİTİM verisi için dönüşüm hattı (veri çoğaltma DAHİL).

    Veri çoğaltma (data augmentation), her epoch'ta görüntüleri rastgele
    değiştirerek modelin EZBER yapmasını (overfitting) zorlaştırır ve
    GENELLEME (generalization) yeteneğini artırır. Proje dokümanında test
    görüntülerinin farklı açı/ışık/arka plandan geleceği belirtildiği için
    bu dönüşümler kritik öneme sahiptir.

    Args:
        img_size: Modelin giriş kenar uzunluğu (kare görüntü). Varsayılan 224.

    Returns:
        Sırayla uygulanacak dönüşümleri içeren transforms.Compose nesnesi.
    """
    return transforms.Compose([
        # Rastgele bir bölgeyi kırpıp img_size'a ölçekler. scale=(0.7, 1.0) ->
        # görüntünün %70-%100'ü arasında bir alan seçilir. Ölçek/konum değişimine
        # karşı dayanıklılık kazandırır.
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        # %50 olasılıkla yatay çevirme. Arabalar sağa/sola dönük olabileceği için
        # anlamlıdır. (Dikey çevirme YAPILMAZ; arabalar ters durmaz.)
        transforms.RandomHorizontalFlip(p=0.5),
        # ±15 derece rastgele döndürme: farklı kamera açılarına karşı dayanıklılık.
        transforms.RandomRotation(degrees=15),
        # Parlaklık/kontrast/doygunluk/ton oynaması: farklı ışık koşullarını taklit eder.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        # Küçük öteleme (translation): nesnenin karede farklı konumda olmasına karşı.
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        # PIL görüntüsünü [0,1] aralığında float tensöre çevirir (C x H x W).
        transforms.ToTensor(),
        # ImageNet istatistikleriyle normalizasyon (transfer learning ile uyumlu).
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        # Tensör üzerinde rastgele bir dikdörtgeni siler (occlusion/tıkanıklık taklidi);
        # modelin tek bir bölgeye aşırı bağımlı olmasını engeller.
        transforms.RandomErasing(p=0.25),
    ])


def get_eval_transforms(img_size: int = config.IMG_SIZE) -> transforms.Compose:
    """
    DOĞRULAMA (validation) / TEST verisi için dönüşüm hattı (çoğaltma YOK).

    Değerlendirme sırasında sonucun DETERMİNİSTİK (her seferinde aynı) olması
    gerekir; bu yüzden rastgelelik içermez. Sadece ortalamak için yeniden
    boyutlandırma, merkezden kırpma ve normalizasyon uygulanır.
    """
    return transforms.Compose([
        # Kısa kenarı img_size+32'ye ölçekler (örn. 256), sonra merkezden img_size kırpar.
        # Bu "resize + center-crop" deseni değerlendirmede standart bir yaklaşımdır.
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


# ===========================================================================
# 2) VERİYİ BÖLME (STRATIFIED TRAIN / VAL / TEST)
# ===========================================================================
def build_datasets(
    data_dir: Path | str = config.DATA_DIR,
    img_size: int = config.IMG_SIZE,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[Subset, Subset, Subset, list[str]]:
    """
    Veri setini eğitim/doğrulama/test olarak SINIF DAĞILIMINI KORUYARAK böler.

    "Stratified" bölme, her alt kümede (train/val/test) sınıf oranlarının
    yaklaşık olarak aynı kalmasını garanti eder; küçük sınıfların bir bölmede
    hiç bulunmaması gibi sorunları önler.

    Args:
        data_dir:   8 sınıf alt klasörünü içeren kök veri dizini.
        img_size:   Görüntü giriş boyutu.
        val_split:  Doğrulama oranı (0-1 arası).
        test_split: Test oranı (0-1 arası).
        seed:       Tekrarlanabilirlik için rastgelelik tohumu.

    Returns:
        (train_ds, val_ds, test_ds, class_names) dörtlüsü.
        train_ds eğitim dönüşümünü; val_ds ve test_ds değerlendirme dönüşümünü kullanır.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Veri dizini bulunamadı: {data_dir}")

    # Aynı klasörü iki kez açıyoruz: farklı alt kümelere farklı dönüşüm uygulamak için.
    # train_base -> eğitim (augmentation'lı), eval_base -> val/test (augmentation'sız).
    train_base = ImageFolder(str(data_dir), transform=get_train_transforms(img_size))
    eval_base = ImageFolder(str(data_dir), transform=get_eval_transforms(img_size))

    # ImageFolder klasör adlarını alfabetik indeksler; config ile tutarlı olduğunu doğrula.
    class_names = train_base.classes
    if class_names != config.CLASS_NAMES:
        raise ValueError(
            "Klasör sınıfları config.CLASS_NAMES ile uyuşmuyor!\n"
            f"  Bulunan : {class_names}\n"
            f"  Beklenen: {config.CLASS_NAMES}"
        )

    # Her örneğin sınıf etiketi (stratify için gerekli).
    targets = train_base.targets
    indices = list(range(len(targets)))

    # 1. bölme: önce test'i ayır (geri kalan = train + val).
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        stratify=targets,         # sınıf dağılımını koru
        random_state=seed,        # tekrarlanabilir
        shuffle=True,
    )
    # 2. bölme: kalanı train ve val olarak ayır.
    # val oranını kalan küme ÜZERİNDEN yeniden hesaplıyoruz ki orijinal val_split'e denk gelsin.
    val_relative = val_split / (1.0 - test_split)
    train_val_targets = [targets[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        stratify=train_val_targets,
        random_state=seed,
        shuffle=True,
    )

    # Subset ile indeks kümelerini ilgili "base" veri setine bağla.
    train_ds = Subset(train_base, train_idx)  # eğitim dönüşümü
    val_ds = Subset(eval_base, val_idx)       # değerlendirme dönüşümü
    test_ds = Subset(eval_base, test_idx)     # değerlendirme dönüşümü

    return train_ds, val_ds, test_ds, class_names


# ===========================================================================
# 3) DATALOADER ÜRETİMİ
# ===========================================================================
def build_dataloaders(
    data_dir: Path | str = config.DATA_DIR,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Eğitim/doğrulama/test için PyTorch DataLoader'larını üretir.

    DataLoader, veri setini 'batch_size'lık gruplara böler, paralel okuma yapar
    ve (eğitimde) her epoch'ta veriyi karıştırır.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir=data_dir,
        img_size=img_size,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    # Karıştırmanın tekrarlanabilir olması için tohumlanmış bir generator.
    generator = torch.Generator().manual_seed(seed)
    # CUDA varsa pin_memory hızlandırır; MPS/CPU'da gereksizdir.
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # eğitimde her epoch'ta karıştır (öğrenmeyi iyileştirir)
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         # değerlendirmede karıştırmaya gerek yok
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names
