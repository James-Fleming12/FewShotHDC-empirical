"""HDC encoders matching the HyperLiDAR reference implementation.

Reference: ``Model`` / ``ClassificationDensityModel`` in
``HyperLiDAR_TTA/DensityUnsupHyperLidar/modules/{HDC_utils,HDC_cl}.py``.
Every encoder maps a feature vector to a bipolar hypervector ``{-1, +1}^q``
and is deterministic after ``fit`` (stateless encoding), which lets the
experiments encode a dataset once and index support/buffer subsets without
changing semantics.

* ``RPEncoder``      -- ``hd_encoder="rp"``: linear random projection followed
  by ``hard_quantize`` (sign).  With ``gauss_rp=True`` (the reference default)
  the weights are QR-orthogonalised Gaussian rows when ``q >= d_in``.
* ``IDLevelEncoder`` -- ``hd_encoder="idlevel"``: a per-feature position
  (identity) hypervector is bound (element-wise product) with a quantised
  value/level hypervector, the bound terms are bundled (summed) and
  hard-quantised.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RPEncoder", "IDLevelEncoder", "make_encoder"]


def _level_chain(levels: int, dim: int, rng: np.random.Generator,
                 randomness: float = 0.0) -> np.ndarray:
    """Ordered level hypervectors.

    The first level is a random bipolar vector; each following level flips one
    contiguous block of a random permutation of the dimensions, so adjacent
    levels differ in ~``dim / (levels - 1)`` bits and the extremes are exact
    negatives.  ``randomness`` adds i.i.d. extra bit flips between levels.
    """
    order = rng.permutation(dim)
    base = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)
    out = np.empty((levels, dim), dtype=np.float32)
    h = base.copy()
    step = dim / max(1, levels - 1)
    for i in range(levels):
        out[i] = h
        if i == levels - 1:
            break
        lo, hi = int(round(i * step)), int(round((i + 1) * step))
        h[order[lo:hi]] *= -1.0
        if randomness > 0.0:
            h[rng.random(dim) < randomness] *= -1.0
    return out


class RPEncoder:
    """Random-projection encoder (reference ``hd_encoder='rp'``)."""

    name = "rp"

    def __init__(self, hd_dim: int = 10000, seed: int = 0, gauss_rp: bool = True):
        self.hd_dim = int(hd_dim)
        self.seed = int(seed)
        self.gauss_rp = bool(gauss_rp)
        self.proj_: np.ndarray | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        d_in = X.shape[1]
        rng = np.random.default_rng(self.seed)
        G = rng.standard_normal((self.hd_dim, d_in))
        if self.gauss_rp and self.hd_dim >= d_in:
            # Reference: nn.Linear weight = Q * sqrt(hd_dim) with Q from QR(G).
            Q, _ = np.linalg.qr(G)
            W = Q * np.sqrt(self.hd_dim)
        else:
            W = G / np.sqrt(d_in)
        self.proj_ = W.astype(np.float32)
        return self

    def encode(self, X, chunk: int = 4096) -> np.ndarray:
        if self.proj_ is None:
            raise RuntimeError("call fit() before encode()")
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((X.shape[0], self.hd_dim), dtype=np.float32)
        for s in range(0, X.shape[0], chunk):
            out[s:s + chunk] = np.where(X[s:s + chunk] @ self.proj_.T >= 0.0, 1.0, -1.0)
        return out


class IDLevelEncoder:
    """ID-level encoder (reference ``hd_encoder='idlevel'``).

    ``h = sign(sum_f position[f] * level[bin(x_f)])`` with per-feature
    min/max normalisation to ``[0, 1]`` when ``normalize=True``.
    """

    name = "idlevel"

    def __init__(self, hd_dim: int = 10000, levels: int = 100, seed: int = 0,
                 randomness: float = 0.0, normalize: bool = True):
        self.hd_dim = int(hd_dim)
        self.levels = int(levels)
        self.seed = int(seed)
        self.randomness = float(randomness)
        self.normalize = bool(normalize)
        self.level_hv_: np.ndarray | None = None
        self.position_hv_: np.ndarray | None = None
        self.x_min_: np.ndarray | None = None
        self.x_max_: np.ndarray | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        d_in = X.shape[1]
        if self.levels < 2:
            raise ValueError("levels must be >= 2")
        rng = np.random.default_rng(self.seed)
        self.level_hv_ = _level_chain(self.levels, self.hd_dim, rng, self.randomness)
        self.position_hv_ = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32), size=(d_in, self.hd_dim))
        self.x_min_ = X.min(axis=0)
        self.x_max_ = np.maximum(X.max(axis=0), self.x_min_ + 1e-9)
        return self

    def _bins(self, X: np.ndarray) -> np.ndarray:
        if self.normalize:
            Xn = (X - self.x_min_) / (self.x_max_ - self.x_min_)
        else:
            Xn = X
        Xn = np.clip(Xn, 0.0, 1.0)
        return np.clip(np.round(Xn * (self.levels - 1)).astype(np.int64),
                       0, self.levels - 1)

    def encode(self, X, chunk: int = 512) -> np.ndarray:
        if self.level_hv_ is None:
            raise RuntimeError("call fit() before encode()")
        X = np.asarray(X, dtype=np.float32)
        idx = self._bins(X)
        out = np.empty((X.shape[0], self.hd_dim), dtype=np.float32)
        for s in range(0, X.shape[0], chunk):
            ic = idx[s:s + chunk]
            acc = np.zeros((ic.shape[0], self.hd_dim), dtype=np.float32)
            for f in range(ic.shape[1]):
                acc += self.level_hv_[ic[:, f]] * self.position_hv_[f]
            out[s:s + chunk] = np.where(acc >= 0.0, 1.0, -1.0)
        return out


def make_encoder(kind: str, hd_dim: int, seed: int = 0, **kwargs):
    """Factory for the encoders used in the reference model."""
    if kind == "rp":
        return RPEncoder(hd_dim=hd_dim, seed=seed, **kwargs)
    if kind == "idlevel":
        return IDLevelEncoder(hd_dim=hd_dim, seed=seed, **kwargs)
    raise ValueError(f"unknown encoder kind {kind!r} (options: 'rp', 'idlevel')")
