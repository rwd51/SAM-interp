"""Separation metrics + CKA.

Deliberately light on dependencies: sklearn + numpy only.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.config import SEED


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    between = np.linalg.norm(mu0 - mu1) ** 2
    within  = X[y == 0].var(0).sum() + X[y == 1].var(0).sum() + 1e-8
    return float(between / within)


def separation_metrics(X: np.ndarray, y: np.ndarray) -> dict:
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return dict(
        linear_probe = float(cross_val_score(LogisticRegression(max_iter=500), Xs, y, cv=cv).mean()),
        knn5         = float(cross_val_score(KNeighborsClassifier(n_neighbors=5), Xs, y, cv=cv).mean()),
        silhouette   = float(silhouette_score(Xs, y)),
        fisher       = fisher_ratio(Xs, y),
    )


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear Centered Kernel Alignment. A, B: (n_samples, n_features)."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    hsic = ((A.T @ B) ** 2).sum()
    denom = np.sqrt(((A.T @ A) ** 2).sum()) * np.sqrt(((B.T @ B) ** 2).sum()) + 1e-9
    return float(hsic / denom)
