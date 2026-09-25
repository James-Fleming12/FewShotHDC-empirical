"""Synthetic Gaussian tasks for the few-shot HDC study.

Each class ``c`` has a centroid ``0.5 + separation * s_c`` with a random sign
pattern ``s_c in {-1, +1}^d``; samples are drawn from
``N(centroid_c, noise^2)`` where ``noise = 2 * separation / snr`` and values are
clipped to ``[0, 1]`` (the encoder input range).  With this parametrisation the
expected per-class pair separation index grows like ``0.5 * d * snr^2``, so
``snr`` controls difficulty independently of the feature count.

Knobs: ``class_counts`` (imbalance), ``label_noise`` (training label flips),
``n_noise_dims`` (irrelevant features).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Dataset", "make_gaussian", "sample_support"]


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    name: str = "gaussian"
    meta: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.x_test.shape[0])

    @property
    def dim(self) -> int:
        return int(self.x_train.shape[1])

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y_train, minlength=self.num_classes)


def make_gaussian(num_classes: int = 5, dim: int = 32, n_train: int = 100,
                  n_test: int = 50, snr: float = 1.0, seed: int = 0,
                  separation: float = 0.15, class_counts=None,
                  label_noise: float = 0.0, n_noise_dims: int = 0,
                  centroids: np.ndarray | None = None,
                  name: str = "gaussian") -> Dataset:
    """Build a balanced (or imbalanced) Gaussian task.

    Pass ``centroids`` (e.g. ``ds.meta["centroids"]``) to draw an independent
    sample from the *same* class distributions, which is how the oracle
    prototypes in Exp 3 are generated.
    """
    rng = np.random.default_rng(seed)
    total_dim = dim + n_noise_dims
    if centroids is None:
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                           size=(num_classes, dim))
        centroids = np.zeros((num_classes, total_dim), dtype=np.float32)
        centroids[:, :dim] = 0.5 + separation * signs
    else:
        centroids = np.asarray(centroids, dtype=np.float32)
        assert centroids.shape[1] == total_dim, (centroids.shape, total_dim)
    noise_std = 2.0 * separation / max(snr, 1e-9)

    if class_counts is None:
        counts = np.full(num_classes, n_train, dtype=np.int64)
    else:
        counts = np.asarray(class_counts, dtype=np.int64)
        assert len(counts) == num_classes

    y_train = np.repeat(np.arange(num_classes, dtype=np.int64), counts)
    x_train = centroids[y_train] + noise_std * rng.standard_normal(
        (len(y_train), total_dim)).astype(np.float32)

    y_test = np.repeat(np.arange(num_classes, dtype=np.int64), n_test)
    x_test = centroids[y_test] + noise_std * rng.standard_normal(
        (len(y_test), total_dim)).astype(np.float32)

    if n_noise_dims > 0:  # irrelevant features: pure class-independent noise
        x_train[:, dim:] = 0.5 + 0.5 * rng.standard_normal((len(y_train), n_noise_dims))
        x_test[:, dim:] = 0.5 + 0.5 * rng.standard_normal((len(x_test), n_noise_dims))

    x_train = np.clip(x_train, 0.0, 1.0).astype(np.float32)
    x_test = np.clip(x_test, 0.0, 1.0).astype(np.float32)

    if label_noise > 0.0:
        n_flip = int(round(label_noise * len(y_train)))
        flip_idx = rng.choice(len(y_train), size=n_flip, replace=False)
        new_labels = rng.integers(0, num_classes - 1, size=n_flip)
        new_labels = np.where(new_labels >= y_train[flip_idx],
                              new_labels + 1, new_labels)  # ensure != current
        y_train = y_train.copy()
        y_train[flip_idx] = new_labels

    meta = {
        "snr": snr,
        "separation": separation,
        "noise_std": noise_std,
        "centroids": centroids,
        "dim_signal": dim,
        "n_noise_dims": n_noise_dims,
        "label_noise": label_noise,
        "class_counts": counts.tolist(),
    }
    return Dataset(x_train, y_train, x_test, y_test, num_classes, name=name, meta=meta)


def sample_support(y: np.ndarray, shots, rng: np.random.Generator) -> np.ndarray:
    """Stratified support indices: up to ``shots`` labelled samples per class.

    ``shots`` may be an int (same for every class) or a per-class sequence.
    """
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    counts = np.full(classes.max() + 1, int(shots), dtype=np.int64) if np.isscalar(shots) \
        else np.asarray(shots, dtype=np.int64)
    idx = []
    for c in classes:
        pool = np.flatnonzero(y == c)
        k = int(min(counts[c], pool.size))
        if k > 0:
            idx.append(rng.choice(pool, size=k, replace=False))
    return np.concatenate(idx).astype(np.int64)
