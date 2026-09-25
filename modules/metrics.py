"""Metrics used by the experiments."""

from __future__ import annotations

import numpy as np

__all__ = ["accuracy", "confusion_matrix", "per_class_recall", "macro_recall",
           "row_cosine", "prototype_margins", "pairwise_prototype_cosine"]


def accuracy(preds: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(preds) == np.asarray(y)).mean())


def confusion_matrix(preds: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y, preds), 1)
    return cm


def per_class_recall(preds: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(preds, y, num_classes)
    support = cm.sum(axis=1)
    return np.divide(np.diag(cm), support, out=np.zeros(num_classes, dtype=np.float64),
                     where=support > 0)


def macro_recall(preds: np.ndarray, y: np.ndarray, num_classes: int) -> float:
    return float(per_class_recall(preds, y, num_classes).mean())


def row_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two matrices with equal rows."""
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return np.sum(An * Bn, axis=1)


def prototype_margins(H: np.ndarray, y: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """cos(h, c_true) - max_{j != true} cos(h, c_j) for every sample."""
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    Pn = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    S = Hn @ Pn.T
    y = np.asarray(y, dtype=np.int64)
    rows = np.arange(len(y))
    true_scores = S[rows, y]
    S[rows, y] = -np.inf
    return true_scores - S.max(axis=1)


def pairwise_prototype_cosine(prototypes: np.ndarray) -> np.ndarray:
    Pn = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    return Pn @ Pn.T
