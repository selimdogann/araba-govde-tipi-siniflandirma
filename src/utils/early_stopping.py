"""
early_stopping.py
=================
EARLY STOPPING (erken durdurma) mekanizması.

Amaç: Overfitting'i önlemek. Model eğitim verisini ezberlemeye başladığında
doğrulama (validation) metriği artık iyileşmez, hatta kötüleşir. Bu sınıf,
izlenen metrik belirli sayıda epoch (patience) boyunca iyileşmezse eğitimi
durdurmamızı söyler. Böylece gereksiz epoch'larda hem zaman harcanmaz hem de
modelin en iyi (en genelleyen) haline yakın kalınır.
"""

from __future__ import annotations


class EarlyStopping:
    """
    İzlenen bir metriği takip eder; iyileşme durursa eğitimi durdurma sinyali verir.

    Kullanım (her epoch sonunda):
        stopper = EarlyStopping(patience=7, mode="max")  # F1'i maksimize ediyoruz
        improved = stopper.step(val_f1)
        if improved:
            # en iyi modeli kaydet
        if stopper.should_stop:
            break
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max") -> None:
        """
        Args:
            patience:  İyileşme olmadan tolere edilecek epoch sayısı.
            min_delta: "İyileşme" sayılması için gereken minimum değişim miktarı.
            mode:      "max" -> büyük metrik daha iyi (ör. F1/accuracy);
                       "min" -> küçük metrik daha iyi (ör. loss).
        """
        if mode not in ("max", "min"):
            raise ValueError("mode 'max' veya 'min' olmalı")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # En iyi gözlemlenen metrik değeri; mode'a göre +/- sonsuz ile başlatılır.
        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.counter: int = 0          # iyileşmeden geçen epoch sayacı
        self.should_stop: bool = False  # eğitimi durdurma bayrağı

    def _is_improvement(self, value: float) -> bool:
        """Yeni değerin, en iyi değere göre anlamlı bir iyileşme olup olmadığını döndürür."""
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def step(self, value: float) -> bool:
        """
        Yeni metrik değerini işler.

        Returns:
            True  -> bu epoch'ta iyileşme oldu (en iyi modeli kaydetmek için kullan).
            False -> iyileşme yok (sayaç artırılır; patience dolarsa should_stop=True olur).
        """
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0      # iyileşme oldu, sayacı sıfırla
            return True

        # İyileşme yok: sayacı artır, patience aşıldıysa durma bayrağını kaldır.
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
