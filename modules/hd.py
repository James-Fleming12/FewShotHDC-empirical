"""Class-hypervector model matching the HyperLiDAR reference.

Reference semantics (``HDC_utils.Model`` + ``Basic_HD.py`` training loop):

* ``classify_weights``: unnormalised accumulator, ``W_true += h`` per sample
  (the few-shot prototype is just the sum of the support encodings).
* ``classify``: row-normalised copy of ``classify_weights`` used for cosine
  logits, ``logits = normalize(h) @ classify.T``.
* classification loss of a sample (Basic_HD): ``cos(h, c_pred) - cos(h, c_gt)``
  for mistakes (>= 0) and ``0`` otherwise.  Negative of the paper's Eq. (4)
  sign convention; "highest loss" = hardest sample.
* error-driven retraining on mistakes only (reference default):
  ``classify_weights[true] += lr * h`` and ``classify_weights[pred] -= lr * h``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HDCModel", "row_normalize", "class_means", "loss_from_scores"]


def row_normalize(W: np.ndarray) -> np.ndarray:
    """``F.normalize`` along rows (zero rows stay zero)."""
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return (W / np.maximum(norms, 1e-12)).astype(np.float32)


def class_means(H: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    """Row-normalised per-class means of encoded samples (oracle prototypes)."""
    y = np.asarray(y, dtype=np.int64)
    P = np.zeros((num_classes, H.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        mask = y == c
        if mask.any():
            P[c] = H[mask].mean(axis=0)
    return row_normalize(P)


def loss_from_scores(scores: np.ndarray, y: np.ndarray,
                     preds: np.ndarray | None = None):
    """Per-sample classification loss (0 for correct predictions)."""
    if preds is None:
        preds = scores.argmax(axis=1)
    loss = np.zeros(scores.shape[0], dtype=np.float32)
    wrong = preds != y
    if wrong.any():
        rows = np.flatnonzero(wrong)
        loss[rows] = scores[rows, preds[rows]] - scores[rows, y[rows]]
    return loss


class HDCModel:
    """Bipolar class hypervectors with cosine readout."""

    def __init__(self, num_classes: int, hd_dim: int, *, bipolar: bool = False):
        self.num_classes = int(num_classes)
        self.hd_dim = int(hd_dim)
        # Reference: `bipolar_prototypes=False` keeps continuous sums.
        self.bipolar = bool(bipolar)
        self.classify_weights = np.zeros((self.num_classes, self.hd_dim), dtype=np.float32)
        self.classify = np.zeros_like(self.classify_weights)
        self.sample_cnt = np.zeros(self.num_classes, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # target bookkeeping
    # ------------------------------------------------------------------ #
    def commit(self) -> None:
        """Refresh the readout matrix from the accumulator."""
        if self.bipolar:
            W = np.sign(self.classify_weights).astype(np.float32)
            W[W == 0.0] = -1.0  # reference sets zero entries to -1
            self.classify_weights = W
            self.classify = W.copy()
        else:
            self.classify = row_normalize(self.classify_weights)

    def accumulate(self, H: np.ndarray, y: np.ndarray,
                   weights: np.ndarray | None = None, lr: float = 1.0) -> None:
        """Add encoded samples to their class accumulators (`index_add_`)."""
        y = np.asarray(y, dtype=np.int64)
        if weights is None:
            np.add.at(self.classify_weights, y, (lr * H).astype(np.float32))
        else:
            w = np.asarray(weights, dtype=np.float32)[:, None]
            np.add.at(self.classify_weights, y, (lr * w * H).astype(np.float32))
        self.sample_cnt += np.bincount(y, minlength=self.num_classes)
        self.commit()

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def scores(self, H: np.ndarray) -> np.ndarray:
        """Cosine logits against the current (committed) prototypes."""
        Hn = row_normalize(H)
        return (Hn @ self.classify.T).astype(np.float32)

    def predict(self, H: np.ndarray) -> np.ndarray:
        return self.scores(H).argmax(axis=1)

    def accuracy(self, H: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(H) == np.asarray(y)).mean())

    def evaluate(self, H: np.ndarray, y: np.ndarray, batch_size: int = 1024) -> dict:
        """Accuracy + per-sample loss + predictions over a split."""
        y = np.asarray(y, dtype=np.int64)
        n = H.shape[0]
        preds = np.empty(n, dtype=np.int64)
        losses = np.empty(n, dtype=np.float32)
        for s in range(0, n, batch_size):
            S = self.scores(H[s:s + batch_size])
            p = S.argmax(axis=1)
            preds[s:s + batch_size] = p
            losses[s:s + batch_size] = loss_from_scores(S, y[s:s + batch_size], p)
        return {"acc": float((preds == y).mean()), "preds": preds, "losses": losses}

    # ------------------------------------------------------------------ #
    # error-driven update (reference `_train_retrain`)
    # ------------------------------------------------------------------ #
    def apply_error_updates(self, H: np.ndarray, y: np.ndarray, preds: np.ndarray,
                            lr: float = 1.0, update_on_correct: bool = False) -> int:
        """Pull the true class towards mistakes and push the predicted class away.

        Returns the number of updated (misclassified) samples.
        """
        y = np.asarray(y, dtype=np.int64)
        wrong = preds != y
        n_wrong = int(wrong.sum())
        if n_wrong:
            Hw = H[wrong].astype(np.float32)
            yw = y[wrong]
            pw = preds[wrong]
            np.add.at(self.classify_weights, yw, (lr * Hw).astype(np.float32))
            np.add.at(self.classify_weights, pw, (-lr * Hw).astype(np.float32))
        if update_on_correct:
            ok = ~wrong
            if ok.any():
                np.add.at(self.classify_weights, y[ok], (lr * H[ok]).astype(np.float32))
        if n_wrong or update_on_correct:
            self.commit()
        return n_wrong
