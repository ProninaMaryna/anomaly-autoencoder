"""Класичні бейзлайни для порівняння: PCA-реконструкція, Isolation Forest, One-Class SVM."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def _flat(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1)


def fit_score(method: str, x_train: np.ndarray, x_eval_list: list[np.ndarray], seed: int,
              params: dict | None = None) -> list[np.ndarray]:
    """Навчає бейзлайн на x_train та повертає anomaly scores (більше = аномальніше)."""
    p = params or {}
    xt = _flat(x_train)
    if method == "pca":
        pca = PCA(n_components=p.get("n_components", 32), random_state=seed).fit(xt)

        def score(x):
            z = pca.transform(_flat(x))
            return ((pca.inverse_transform(z) - _flat(x)) ** 2).mean(1)
    elif method == "iforest":
        model = IsolationForest(n_estimators=p.get("n_estimators", 200), random_state=seed).fit(xt)
        score = lambda x: -model.score_samples(_flat(x))
    elif method == "ocsvm":
        model = OneClassSVM(nu=p.get("nu", 0.1), gamma=p.get("gamma", "scale")).fit(xt)
        score = lambda x: -model.score_samples(_flat(x))
    else:
        raise ValueError(method)
    return [score(x) for x in x_eval_list]
