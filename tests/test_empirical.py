"""Fast empirical tests for the three research questions.

These are small, fixed-seed versions of the full experiments (which live in
``experiments/exp*.py``); they assert the qualitative signatures that motivate
the study, so a run that passes means the pipelines behave as expected before
scaling up.

Run: ``python tests/test_empirical.py`` or ``pytest tests/test_empirical.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import run_configs
from modules import (HDCModel, RetrainConfig, make_encoder, make_gaussian,
                     sample_support)
from modules.hd import class_means
from modules.metrics import row_cosine

DATA_CFG = dict(epochs=8, batch_size=256, buffer_fraction=0.05)
C, D = 5, 32


def _setup(seed: int, *, n_train: int = 100, n_test: int = 50, snr: float = 1.0,
           classes: int = C, dim: int = D, hd_dim: int = 1024):
    ds = make_gaussian(classes, dim, n_train, n_test, snr=snr, seed=seed)
    enc = make_encoder("rp", hd_dim, seed=seed).fit(ds.x_train)
    H = (enc.encode(ds.x_train), enc.encode(ds.x_test))
    return ds, enc, H


def _rows(ds, enc, H, *, shots, seed, configs, selection="hard_random"):
    cfg = RetrainConfig(seed=seed, **DATA_CFG)
    rows, _ = run_configs(ds, enc, shots=shots, retrain=cfg, seed=seed,
                          configs=configs, selection=selection, H=H)
    return rows


def _mean_acc(rows, config):
    return float(np.mean([r["acc"] for r in rows if r["config"] == config]))


def _prototype(enc, ds, H_train, shots, seed):
    sup = sample_support(ds.y_train, shots, np.random.default_rng(seed + 31))
    model = HDCModel(ds.num_classes, enc.hd_dim)
    model.accumulate(H_train[sup], ds.y_train[sup])
    return model.classify


# --------------------------------------------------------------------------- #
# Q1: few-shot ability
# --------------------------------------------------------------------------- #
def test_q1_fewshot_accuracy_grows_with_shots():
    accs = {1: [], 5: [], 20: []}
    for seed in range(3):
        ds, enc, H = _setup(seed)
        for shots in accs:
            rows = _rows(ds, enc, H, shots=shots, seed=seed, configs=("fewshot",))
            accs[shots].append(rows[0]["acc"])
    assert np.mean(accs[20]) > np.mean(accs[1]) + 0.05, accs
    assert np.mean(accs[5]) > np.mean(accs[1]) - 0.03, accs


def test_q1_buffer_matches_full_retrain_at_low_budget():
    full, buf, frac = [], [], []
    for seed in range(3):
        ds, enc, H = _setup(seed)
        rows = _rows(ds, enc, H, shots=20, seed=seed, configs=("full", "buffer"))
        f = [r for r in rows if r["config"] == "full"][0]
        b = [r for r in rows if r["config"] == "buffer"][0]
        full.append(f["acc"]); buf.append(b["acc"]); frac.append(b["n_encoded_rel"])
        assert b["acc"] > 0.75, b  # better than a 5-shot prototype on this task
    assert np.mean(full) - np.mean(buf) < 0.10, (full, buf)
    assert np.mean(frac) < 0.4, frac


# --------------------------------------------------------------------------- #
# Q2: breakage
# --------------------------------------------------------------------------- #
def test_q2_capacity_collapse_at_small_dimension():
    small, large = [], []
    for seed in range(2):
        for q, out in ((64, small), (1024, large)):
            ds, enc, H = _setup(seed, classes=10, hd_dim=q)
            out.append(_rows(ds, enc, H, shots=5, seed=seed, configs=("fewshot",))[0]["acc"])
    assert np.mean(small) < np.mean(large) - 0.1, (small, large)


def test_q2_label_noise_degrades_all_configs():
    clean, noisy = [], []
    for seed in range(2):
        for noise, out in ((0.0, clean), (0.35, noisy)):
            ds = make_gaussian(C, D, 100, 50, snr=1.0, seed=seed, label_noise=noise)
            enc = make_encoder("rp", 512, seed=seed).fit(ds.x_train)
            H = (enc.encode(ds.x_train), enc.encode(ds.x_test))
            rows = _rows(ds, enc, H, shots=5, seed=seed, configs=("full",))
            out.append(rows[0]["acc"])
    assert np.mean(noisy) < np.mean(clean) - 0.05, (clean, noisy)


def test_q2_support_contamination_hurts_fewshot():
    drops = []
    for seed in range(3):
        ds, enc, H = _setup(seed)
        sup = sample_support(ds.y_train, 5, np.random.default_rng(seed))
        clean = _rows(ds, enc, H, shots=5, seed=seed, configs=("fewshot",))
        cfg = RetrainConfig(seed=seed, **DATA_CFG)
        x_bad = ds.x_train[sup].copy()
        rng = np.random.default_rng(seed + 99)
        bad = rng.choice(len(sup), size=len(sup) // 2, replace=False)
        x_bad[bad] = rng.random((len(bad), ds.dim)).astype(np.float32)
        dirty, _ = run_configs(ds, enc, shots=5, retrain=cfg, seed=seed,
                               configs=("fewshot",), support=sup, support_x=x_bad, H=H)
        drops.append(clean[0]["acc"] - dirty[0]["acc"])
    assert float(np.mean(drops)) > 0.05, drops


# --------------------------------------------------------------------------- #
# Q3: prototype sample complexity
# --------------------------------------------------------------------------- #
def _oracle_prototypes(enc, seed, *, snr, classes=C, dim=D, per_class=1500,
                       centroids=None):
    oracle = make_gaussian(classes, dim, per_class, 1, snr=snr, seed=seed + 10_000,
                           centroids=centroids)
    return class_means(enc.encode(oracle.x_train), oracle.y_train, classes)


def test_q3_prototype_fidelity_grows_with_shots():
    sims = {1: [], 32: []}
    for seed in range(3):
        ds, enc, H = _setup(seed, n_train=200)
        P_or = _oracle_prototypes(enc, seed, snr=1.0, centroids=ds.meta["centroids"])
        for shots in sims:
            P = _prototype(enc, ds, H[0], shots, seed)
            sims[shots].append(float(row_cosine(P, P_or).mean()))
    assert np.mean(sims[32]) > np.mean(sims[1]) + 0.05, sims


def test_q3_overlapping_classes_need_more_samples():
    easy, hard = [], []
    for seed in range(3):
        for snr, out in ((2.0, easy), (0.25, hard)):
            ds, enc, H = _setup(seed, n_train=200, snr=snr)
            P_or = _oracle_prototypes(enc, seed, snr=snr, centroids=ds.meta["centroids"])
            # 4 shots: the low-SNR prototype is still far from the oracle, while
            # the high-SNR one is already close (equivalently: overlapping
            # classes need more samples for the same prototype fidelity).
            P = _prototype(enc, ds, H[0], 4, seed)
            out.append(float(row_cosine(P, P_or).mean()))
    assert np.mean(easy) > np.mean(hard) + 0.05, (easy, hard)


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__} ok")
    print(f"\nall {len(tests)} empirical tests passed")


if __name__ == "__main__":
    main()
