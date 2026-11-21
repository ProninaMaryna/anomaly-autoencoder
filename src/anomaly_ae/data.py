"""Підготовка даних для задачі виявлення аномалій (novelty detection) на MNIST.

Протокол one-vs-rest:
  * "нормальні" дані — зображення цифр із ``normal_classes``; усі інші цифри — аномалії;
  * навчальна вибірка: лише нормальні зображення з MNIST-train (опційно — з домішкою
    аномалій ``contamination`` для перевірки стійкості);
  * валідаційна вибірка: відкладені нормальні зображення з MNIST-train + аномалії з
    MNIST-train (використовується ТІЛЬКИ для ранньої зупинки, вибору гіперпараметрів
    та порогу);
  * тестова вибірка: увесь MNIST-test (10 000 зображень), не використовується
    для жодних рішень щодо моделі.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST


@dataclass
class Split:
    x: np.ndarray        # (N, 1, 28, 28) float32 у [0, 1]
    y: np.ndarray        # (N,) 1 = аномалія, 0 = норма
    digit: np.ndarray    # (N,) вихідна мітка цифри (для аналізу помилок)


@dataclass
class AnomalyData:
    train: Split
    val: Split
    test: Split
    normal_classes: list[int]


@lru_cache(maxsize=2)
def _load_raw(root: str | Path):
    tr = MNIST(str(root), train=True, download=True)
    te = MNIST(str(root), train=False, download=True)
    to_np = lambda t: (t.numpy().astype(np.float32) / 255.0)[:, None, :, :]
    return to_np(tr.data), tr.targets.numpy(), to_np(te.data), te.targets.numpy()


def make_data(cfg: dict, seed: int) -> AnomalyData:
    d = cfg["data"]
    normal = list(d["normal_classes"])
    rng = np.random.default_rng(seed)
    root = Path(d["root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    x_tr, y_tr, x_te, y_te = _load_raw(str(root))

    is_norm_tr = np.isin(y_tr, normal)
    norm_idx = rng.permutation(np.where(is_norm_tr)[0])
    anom_idx = rng.permutation(np.where(~is_norm_tr)[0])

    n_val_norm = int(round(d["val_fraction"] * len(norm_idx)))
    val_norm, train_norm = norm_idx[:n_val_norm], norm_idx[n_val_norm:]
    if d.get("n_train") is not None:
        train_norm = train_norm[: int(d["n_train"])]

    n_val_anom = int(round(d["val_anomaly_ratio"] * n_val_norm))
    val_anom, anom_pool = anom_idx[:n_val_anom], anom_idx[n_val_anom:]

    c = float(d.get("contamination", 0.0))
    n_cont = int(round(c * len(train_norm) / (1.0 - c))) if c > 0 else 0
    train_idx = np.concatenate([train_norm, anom_pool[:n_cont]])
    train_idx = rng.permutation(train_idx)
    val_idx = np.concatenate([val_norm, val_anom])

    def mk(x, y, idx):
        return Split(x[idx], (~np.isin(y[idx], normal)).astype(np.int64), y[idx])

    test_all = np.arange(len(y_te))
    return AnomalyData(mk(x_tr, y_tr, train_idx), mk(x_tr, y_tr, val_idx),
                       mk(x_te, y_te, test_all), normal)

