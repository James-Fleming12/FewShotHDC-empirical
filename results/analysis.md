# FewShotHDC-empirical -- quick-run analysis

Setup for the numbers below (all produced by the checked-in scripts):

* encoder `rp` (QR-Gaussian random projection, sign-quantised), unless noted;
* `q = 4096` (Exp 1/2), `q = 2048` (Exp 3), `epochs = 10`, `lr = 1`,
  buffer fraction `k = 5%` (hard:random = 50:50);
* 3 seeds per setting; synthetic Gaussian tasks: `C = 5`, `d = 32`,
  `n_train = 100`/class (Exp 1/2), `200`/class (Exp 3); balanced test set;
* configs: **A = few-shot**, **B = full retrain**, **C = buffered retrain**,
  **B0 = full-data means without retraining** (added as reference in Exp 2),
  **C-rand = random-only buffer** (control).

Encode budgets: with `N = 500` training samples, B encodes `N·(1+epochs) = 5500`
samples, C encodes `N + epochs·k·N = 750` (13.6% of B), and a 50-shot A encodes
250 (4.5% of B).

---

## Q1 -- few-shot ability (Exp 1, `exp1_fewshot.csv`)

Mean test accuracy:

| SNR | 1 shot | 2 | 5 | 10 | 20 | 50 | B full | C buffer | chance |
|-----|--------|---|---|----|----|----|--------|----------|--------|
| 0.25 | 0.205 | 0.188 | 0.208 | 0.248 | 0.244 | 0.276 | 0.247 | 0.257 | 0.20 |
| 0.50 | 0.228 | 0.257 | 0.329 | 0.432 | 0.436 | 0.500 | 0.489 | 0.460 | 0.20 |
| 1.00 | 0.407 | 0.617 | 0.733 | 0.808 | 0.849 | 0.865 | 0.871 | 0.872 | 0.20 |

* At SNR = 1, 1-shot already reaches 0.41 (2x chance) and the curve saturates
  onto the full-retrain accuracy around 20-50 shots: the mean prototype is
  sample-efficient.
* At SNR = 0.5, ~50 shots match B; at SNR = 0.25 everything is at chance and
  few-shot never recovers -- a first sign that few-shot ability is bounded by
  class overlap, not by the number of shots (see Q3).
* C matches B within 0.01 at SNR >= 0.5 with 13.6% of the encode budget.
* `idlevel` (position x level, `q = 1024`) gives the same picture, e.g. SNR = 1:
  few-shot 1/5/20/50 = 0.427/0.676/0.823/0.853 vs full 0.852, buffer 0.851.

## Q2 -- where it breaks (Exp 2, `exp2_*.csv`)

### Class overlap (`exp2_overlap.csv`)

| SNR | A few-shot | B full | B0 full-init | C buffer | C-rand |
|-----|-----------|--------|--------------|----------|--------|
| 0.10 | 0.197 | 0.213 | 0.208 | 0.195 | 0.199 |
| 0.35 | 0.260 | 0.327 | 0.369 | 0.344 | 0.363 |
| 0.60 | 0.404 | 0.581 | 0.617 | 0.569 | 0.615 |
| 1.00 | 0.733 | 0.871 | 0.880 | 0.872 | 0.871 |
| 2.00 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |

A degrades earlier and lower than every full-data variant; the gap is largest in
the intermediate regime (0.35-0.6) where a few samples cannot span the class
distribution. This is the main "few-shot breaks" boundary for this task family.

### Label noise (`exp2_label_noise.csv`)

| flip frac | A few-shot | B full | B0 full-init | C hard+random | C-rand |
|-----------|-----------|--------|--------------|---------------|--------|
| 0.00 | 0.733 | 0.871 | 0.880 | 0.872 | 0.871 |
| 0.10 | 0.737 | 0.797 | 0.879 | 0.795 | 0.875 |
| 0.20 | 0.679 | 0.700 | 0.877 | 0.761 | 0.843 |
| 0.40 | 0.489 | 0.485 | 0.851 | 0.449 | 0.752 |

* The strongest breakage found: **retraining on noisy labels destroys accuracy**
  while the plain full-data means (B0) stay at 0.85.
* **Hard-sample mining makes it worse**: C (0.449) < B (0.485) << C-rand (0.752).
  "Hard" samples are precisely the mislabeled ones, so the buffer keeps
  relearning label noise. A random buffer is the robust choice in this regime.

### Encoder capacity (`exp2_capacity.csv`, 10 classes)

| q | A | B | B0 | C |
|---|---|----|----|----|
| 64 | 0.405 | 0.553 | 0.593 | 0.574 |
| 256 | 0.552 | 0.729 | 0.751 | 0.744 |
| 1024 | 0.624 | 0.809 | 0.816 | 0.807 |
| 4096 | 0.634 | 0.816 | 0.826 | 0.817 |

All variants collapse below `q ~ 256` (10 classes x 32 features over-saturate a
small hypervector); the few-shot prototype is the most sensitive (its estimate
noise adds to interference).

### Class imbalance (`exp2_imbalance.csv`, counts [100, 25, 6, 4, 4])

| ratio | A acc | A rare-recall | B acc | B rare | B0 acc | B0 rare | C acc | C rare |
|-------|-------|---------------|-------|--------|--------|---------|-------|--------|
| 1 | 0.733 | 0.58 | 0.871 | 0.78 | 0.880 | 0.81 | 0.872 | 0.78 |
| 2 | 0.749 | 0.62 | 0.749 | 0.29 | 0.800 | 0.49 | 0.753 | 0.29 |
| 4 | 0.769 | 0.56 | 0.641 | 0.17 | 0.731 | 0.35 | 0.641 | 0.17 |

Long tails break the full-data pipelines: rare-class recall collapses already
in the initial means (B0: 0.35 at ratio 4) and further under error-driven
retraining (B/C: 0.17). Balanced few-shot support keeps 0.56 and even wins on
overall accuracy at ratio 4. This is a case where few-shot prototypes are
*more* robust than the full pipeline, because the support is balanced by
construction.

### Support-set contamination (`exp2_contamination.csv`)

| corrupted | A feature outliers | A label flips |
|-----------|--------------------|---------------|
| 0.0 | 0.713 | 0.713 |
| 0.2 | 0.660 | 0.605 |
| 0.4 | 0.624 | 0.487 |

Few-shot prototypes are corrupted by bad support samples; flipping support
labels is ~2x more damaging than feature outliers per corrupted sample (the
prototype is pulled towards another class HV). B/C are unaffected (clean pool).

### Irrelevant features (`exp2_noise_dims.csv`)

| extra dims | A | B | B0 | C |
|------------|----|----|----|----|
| 0 | 0.733 | 0.871 | 0.880 | 0.872 |
| 16 | 0.636 | 0.884 | 0.896 | 0.889 |
| 64 | 0.479 | 0.864 | 0.864 | 0.872 |
| 256 | 0.352 | 0.692 | 0.695 | 0.691 |

Few-shot breaks much earlier under irrelevant-feature dilution (0.64 vs 0.88 at
16 extra dims); full-data variants absorb noise features until `d_noise ~ q`
starts to dilute the signal.

## Q3 -- prototypes that need more samples (Exp 3, `exp3_prototypes.csv`)

cos(few-shot prototype, oracle prototype) -- oracle = class mean over 2000
clean samples drawn from the same centroids:

| shots | SNR 0.25 | SNR 0.5 | SNR 1 | SNR 2 |
|-------|----------|---------|-------|-------|
| 1 | 0.639 | 0.684 | 0.774 | 0.875 |
| 4 | 0.839 | 0.874 | 0.924 | 0.962 |
| 16 | 0.953 | 0.965 | 0.980 | 0.990 |
| 64 | 0.987 | 0.990 | 0.994 | 0.997 |

Mean decision margin `cos(correct) - max cos(wrong)`:

| shots | SNR 0.25 | SNR 0.5 | SNR 1 | SNR 2 |
|-------|----------|---------|-------|-------|
| 1 | -0.061 | -0.048 | -0.006 | +0.057 |
| 8 | -0.032 | -0.012 | +0.032 | +0.084 |
| 64 | -0.011 | +0.001 | +0.036 | +0.085 |

* Prototype fidelity grows quickly (roughly `1 - cos ~ c/n`), but the *usable*
  prototype -- one with a positive decision margin -- needs far more samples as
  overlap increases: 1 shot at SNR 2, 2 at SNR 1, ~32-64 at SNR 0.5, and never
  (within 64 shots) at SNR 0.25. Fidelity saturating near 1 does not imply a
  separable prototype.
* Shots to 80% test accuracy (`exp3_shots_to_target.csv`): 1 at SNR 2, 9.5 at
  SNR 1, unreachable at SNR <= 0.5 (plateaus 0.56/0.34).
* More features help (SNR 1, target 80%): unreachable at `d = 8/16`, 8.9 shots
  at `d = 32`, 2.3 shots at `d = 64`.
* More classes hurt (SNR 1, target 80%): 1.6 shots for `C = 2`, 8.9 for `C = 5`,
  40.3 for `C = 10` -- interference between prototypes is a first-order driver
  of sample complexity.

## Cross-cutting observations

1. **The retraining step rarely earns its cost on stationary synthetic tasks.**
   B/B0 are within noise in the easy/low-noise regimes, and retraining actively
   hurts under label noise and imbalance. Its benefits would have to come from
   non-stationarity (drift, new contexts) that these tests do not model.
2. **Buffer selection is accurate but fragile.** At 13.6% of B's encode budget,
   C tracks B on clean data. Under label noise the hard half is poison; the
   random-only control is much more robust. This suggests mixing in more random
   samples (or validating the hard fraction) when labels may be noisy.
3. **Few-shot failure modes are predictable from geometry**: overlap (SNR),
   support contamination, irrelevant features, and capacity/interference.
   Balanced support versus long-tailed full data can actually make A *better*
   for rare classes.
4. **Prototype fidelity is necessary but not sufficient**: cosine to the oracle
   saturates long before the margin becomes positive for overlapping classes.

## Caveats

* Synthetic Gaussian clusters with a fixed `[0,1]` feature range and a global
  random projection; conclusions about absolute accuracy do not transfer to
  LiDAR features.
* No domain shift / stream non-stationarity, no unlabeled inference-time
  updates, no subcluster/density model -- deliberate scope limits.
* Mini-batch retraining (batch 256) normalises prototypes per batch as in
  `Basic_HD.retrain`; the reference processes one scan per batch, which is
  analogous. Buffer selection is global per epoch on the pooled dataset.

## Reproduction

```bash
python tests/test_sanity.py && python tests/test_empirical.py
python experiments/exp1_fewshot_ability.py --seeds 3 --hd-dim 4096
python experiments/exp2_breakage.py --seeds 3 --hd-dim 4096
python experiments/exp3_prototype_samples.py --seeds 3 --hd-dim 2048 \
    --oracle-per-class 2000 --sweep --sweep-seeds 2
# or everything at once:
python experiments/run_all.py            # quick
python experiments/run_all.py --full     # larger paper settings
```
