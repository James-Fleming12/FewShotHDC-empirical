"""Robustness tests: input noise, domain shift, support adaptation,
class-prior shift and long-tail imbalance.

Small fixed-seed versions of `experiments/exp4_robustness.py`; they assert the
qualitative signatures behind the Exp 4 findings.

Run: ``python tests/test_robustness.py`` or ``pytest tests/test_robustness.py``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import run_configs
from modules import RetrainConfig, make_encoder, make_gaussian, sample_support

DATA_CFG = dict(epochs=8, batch_size=256, buffer_fraction=0.05)
C, D = 5, 32


def _setup(seed: int, *, n_train: int = 100, n_test: int = 50, hd_dim: int = 2048,
           **kw):
    ds = make_gaussian(C, D, n_train, n_test, snr=1.0, seed=seed, **kw)
    enc = make_encoder("rp", hd_dim, seed=seed).fit(ds.x_train)
    return ds, enc, (enc.encode(ds.x_train), enc.encode(ds.x_test))


def _row(ds, enc, H, seed, *, config="fewshot", shots=5, support=None,
         support_x=None, extra=None):
    cfg = RetrainConfig(seed=seed, **DATA_CFG)
    rows, _ = run_configs(ds, enc, shots=shots, retrain=cfg, seed=seed,
                          configs=(config,), selection="hard_random",
                          support=support, support_x=support_x, extra=extra, H=H)
    return rows[0]


def _natural_support(y, total, rng):
    """One sample per class, the remainder proportional to class frequency."""
    y = np.asarray(y, dtype=np.int64)
    picks = [np.atleast_1d(rng.choice(np.flatnonzero(y == c))) for c in np.unique(y)]
    remaining = max(0, total - sum(len(p) for p in picks))
    if remaining:
        counts = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float64) + 1.0
        p = counts[y] / counts[y].sum()
        picks.append(rng.choice(len(y), size=remaining, replace=False, p=p))
    return np.concatenate(picks).astype(np.int64)


def _scale_shift(ds, enc, severity, seed):
    """Per-feature gain on the test domain (covariance-style covariate shift)."""
    rng = np.random.default_rng([seed, 17])
    gain = (1.0 + severity * rng.uniform(-1.0, 1.0, size=ds.dim)).astype(np.float32)
    x_shift = np.clip(ds.x_test * gain, 0.0, 1.0).astype(np.float32)
    return gain, enc.encode(x_shift)


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #
def test_input_noise_degrades_full_pipeline():
    drops = []
    for seed in range(2):
        ds, enc, (Htr, Hte) = _setup(seed)
        rng = np.random.default_rng([seed, 5])
        x_noisy = np.clip(ds.x_test + 0.4 * rng.standard_normal(ds.x_test.shape),
                          0.0, 1.0).astype(np.float32)
        clean = _row(ds, enc, (Htr, Hte), seed, config="full")["acc"]
        noisy = _row(ds, enc, (Htr, enc.encode(x_noisy)), seed, config="full")["acc"]
        drops.append(clean - noisy)
    assert np.mean(drops) > 0.1, drops


# --------------------------------------------------------------------------- #
# domain shift
# --------------------------------------------------------------------------- #
def test_domain_shift_hurts_fewshot_more_than_full():
    few, full = [], []
    for seed in range(3):
        ds, enc, (Htr, Hte) = _setup(seed)
        _, Hshift = _scale_shift(ds, enc, severity=2.0, seed=seed)
        few.append(_row(ds, enc, (Htr, Hshift), seed, config="fewshot")["acc"])
        full.append(_row(ds, enc, (Htr, Hshift), seed, config="full")["acc"])
    assert np.mean(full) > np.mean(few) + 0.05, (few, full)


def test_shifted_support_adapts_fewshot_prototypes():
    gains = []
    for seed in range(3):
        ds, enc, (Htr, _) = _setup(seed)
        gain, Hshift = _scale_shift(ds, enc, severity=2.0, seed=seed)
        support = sample_support(ds.y_train, 5, np.random.default_rng(seed + 31))
        source = _row(ds, enc, (Htr, Hshift), seed, support=support)["acc"]
        x_sup = np.clip(ds.x_train[support] * gain, 0.0, 1.0).astype(np.float32)
        shifted = _row(ds, enc, (Htr, Hshift), seed, support=support,
                       support_x=x_sup)["acc"]
        gains.append(shifted - source)
    assert np.mean(gains) > 0.05, gains


# --------------------------------------------------------------------------- #
# class-prior shift
# --------------------------------------------------------------------------- #
def test_prior_shift_is_absorbed_by_prior_free_prototypes():
    """Cosine prototypes carry no class prior: skewing the test stream changes
    the class mix (overall accuracy) but not per-class behaviour, for both the
    few-shot and the full/retrained pipelines (macro recall stays within noise).
    """
    for config in ("fewshot", "full"):
        diffs = []
        for seed in range(3):
            ds, enc, (Htr, Hte) = _setup(seed)
            rng = np.random.default_rng([seed, 41])
            w = np.ones(C)
            w[:2] = 10.0
            counts = rng.multinomial(250, w / w.sum())
            idx = np.concatenate([
                rng.choice(np.flatnonzero(ds.y_test == c), size=int(counts[c]),
                           replace=True)
                for c in range(C)])
            ds_skew = replace(ds, y_test=ds.y_test[idx])
            m0 = _row(ds, enc, (Htr, Hte), seed, config=config)["macro_recall"]
            m1 = _row(ds_skew, enc, (Htr, Hte[idx]), seed, config=config)["macro_recall"]
            diffs.append(m1 - m0)
        assert abs(float(np.mean(diffs))) < 0.06, (config, diffs)


# --------------------------------------------------------------------------- #
# class imbalance
# --------------------------------------------------------------------------- #
def test_imbalance_balanced_support_protects_rare_classes():
    counts = [100, 25, 6, 4, 4]  # ratio 4 long tail
    bal, nat, full = [], [], []
    acc_bal, acc_nat = [], []
    for seed in range(3):
        ds, enc, (Htr, Hte) = _setup(seed, class_counts=counts)
        b = _row(ds, enc, (Htr, Hte), seed,
                 support=sample_support(ds.y_train, 5, np.random.default_rng(seed)))
        n = _row(ds, enc, (Htr, Hte), seed,
                 support=_natural_support(ds.y_train, C * 5, np.random.default_rng(seed + 7)))
        f = _row(ds, enc, (Htr, Hte), seed, config="full")
        bal.append(b["min_class_recall"]); nat.append(n["min_class_recall"])
        full.append(f["min_class_recall"])
        acc_bal.append(b["acc"]); acc_nat.append(n["acc"])
    assert np.mean(bal) > np.mean(nat) + 0.15, (bal, nat)
    assert np.mean(bal) > np.mean(full) + 0.1, (bal, full)
    assert np.mean(acc_nat) < np.mean(acc_bal) - 0.2, (acc_bal, acc_nat)


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__} ok")
    print(f"\nall {len(tests)} robustness tests passed")


if __name__ == "__main__":
    main()
