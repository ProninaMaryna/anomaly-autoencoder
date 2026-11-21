"""Оцінки аномальності (anomaly score) та метрики."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


@torch.no_grad()
def reconstruction_scores(model: torch.nn.Module, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Anomaly score = середньоквадратична помилка реконструкції кожного зображення."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size])
        rec = model(xb)
        out.append(((rec - xb) ** 2).flatten(1).mean(1).numpy())
    return np.concatenate(out)


def threshold_from_normals(scores_normal: np.ndarray, percentile: float) -> float:
    """Поріг = перцентиль помилок на валідаційних НОРМАЛЬНИХ зразках (контроль FPR)."""
    return float(np.percentile(scores_normal, percentile))


def compute_metrics(y: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    m = {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }
    if threshold is not None:
        pred = (scores > threshold).astype(int)
        m.update({
            "threshold": float(threshold),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "fpr": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        })
    return m


def per_digit_detection(digits: np.ndarray, y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Для кожної цифри: частка зразків, позначених як аномалія, і AUC (ця цифра vs норма)."""
    res = {}
    norm_scores = scores[y == 0]
    for d in np.unique(digits):
        mask = digits == d
        rate = float((scores[mask] > threshold).mean())
        entry = {"flagged_rate": rate, "n": int(mask.sum())}
        if y[mask][0] == 1:
            yy = np.r_[np.zeros(len(norm_scores)), np.ones(mask.sum())]
            ss = np.r_[norm_scores, scores[mask]]
            entry["auc_vs_normal"] = float(roc_auc_score(yy, ss))
        res[int(d)] = entry
    return res
