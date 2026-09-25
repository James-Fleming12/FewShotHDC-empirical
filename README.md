# FewShotHDC-empirical

An empirical study of **few-shot learning with hyperdimensional computing (HDC)
class prototypes** in a post-deployment adaptation pipeline, on synthetic
datasets, with a target/update implementation that follows the HyperLiDAR
reference model.

The study asks:

1. **How good is few-shot HDC?** What test accuracy does a prototype built from
   `k` labelled samples per class reach, and how does it compare with the full
   retraining pipelines it is meant to replace?
2. **Where does it break?** Class overlap, label noise, hypervector capacity,
   class imbalance, support contamination, irrelevant features.
3. **Which prototype properties need more samples?** Fidelity to the
   infinite-data prototype, decision margins, and how many shots are needed for
   target accuracy as a function of overlap, feature count and class count.
4. **How robust is the pipeline?** Input noise, domain shift (mean, variance,
   per-feature scale), few-shot support adaptation to a shifted domain,
   class-prior shift, and long-tailed class imbalance with balanced vs natural
   support sampling.

All results in this README come from the full runs
(`python experiments/run_all.py --full`): 5 seeds, `q = 10000`, 20 retraining
epochs, 5% buffer; exp3 additionally reports feature/class sweeps. Generated
tables with paired t-tests over seeds:
[`results/summary.md`](results/summary.md); figures: `results/fig_*.png`.

## Problem setting

The deployment pipeline mirrors HyperLiDAR's adaptation flow: a feature
extractor and HDC encoder are fixed, class hypervectors ("prototypes") live on
device, and prototypes are built or adapted from labelled samples. Under this
setting the question is how far cheap prototype construction can replace the
expensive full retraining loop, and where each pipeline fails.

Target model semantics (`modules/hd.py:HDCModel`, matching the reference
`Model`):

* `classify_weights` is the **sum** of encoded samples per class;
* `classify` is its row-normalised copy, used for cosine logits
  `logits = normalize(h) @ classify.T`;
* a misclassified sample has loss `cos(h, c_pred) - cos(h, c_true)` (0 when
  correct), the reference `Basic_HD` convention: highest loss = hardest;
* error-driven retraining moves only misclassified samples,
  `classify_weights[true] += lr·h` and `classify_weights[pred] -= lr·h`, with
  per-batch prototype normalisation.

### The three configurations

| Config | Prototype construction | Encode budget (`N` train samples, `R` epochs, buffer `k`) |
|---|---|---|
| **A** `fewshot` | weighted average of `k` labelled support samples per class (uniform weights; optional per-sample weights) | `C·k` |
| **B** `full` | full-data class means, then `R` error-driven passes over the full pool | `N·(1+R)` |
| **C** `buffer` | one full-data initial pass (accumulate + score + per-sample loss), then `R` passes over a `k%` buffer: `hard_fraction` = highest stored loss, remainder random from the rest; only buffer losses are refreshed (paper Sec. 4.3) | `N + R·k·N` |

Exp 2-4 also report **B0** = full-data means *without* retraining, and a
**C-rand** control = random-only buffer, to separate "retraining" from
"hard-sample mining".

### Reference fidelity and deviations

Matches `HyperLiDAR_TTA/DensityUnsupHyperLidar/modules/{HDC_utils,Basic_HD,HDC_cl}.py`:
sum accumulators + cosine readout; mistake-only updates (`lr = 1` as
`HDC_cl._train_retrain`, `lr = 2` reproduces the double `index_add_` in
`Basic_HD.retrain`); per-batch normalisation; `rp` encoder = QR-orthogonalised
Gaussian projection + `hard_quantize` (reference default), `idlevel` =
position ⊗ level bundling; `selection="wrong_first"` reproduces the reference
`Model.encode` buffer rule.

Deliberate simplifications: the initial pass accumulates a batch before scoring
it (encode-once); buffer selection is global per epoch over the pooled train
set; encodings are precomputed and indexed, so `n_encoded` is the *equivalent*
deployment encode budget (deterministic encoders make this exact); no
subclusters/density model, no unlabeled inference-time updates, no drift.

## Synthetic tasks

`modules/data.py:make_gaussian` builds `C` Gaussian classes with centroids
`0.5 ± separation` (random sign patterns), per-feature SNR `snr` (noise
`2·separation/snr`), optional long-tailed `class_counts`, `label_noise`,
`n_noise_dims` irrelevant features, and optional shared `centroids` (used to
sample Exp 3 oracle prototypes from the *same* distributions). The expected
separability index grows like `0.5·d·snr²`, so difficulty is controllable
independently of feature count. Defaults: `C=5`, `d=32`, 100 train / 50 test
samples per class, values clipped to `[0,1]`.

Metrics: test accuracy, macro recall, rarest-class recall, per-class recall,
prototype cosine to oracle, decision margin (`cos(true) - max cos(wrong)`),
number of encoded samples.

## Layout

```
modules/
  encoders.py   rp / idlevel bipolar encoders (deterministic after fit)
  hd.py         HDCModel: accumulators, cosine readout, loss, error updates
  training.py   initial / full-retrain / buffered-retrain passes, buffer selection
  data.py       synthetic Gaussian tasks (SNR, imbalance, label noise, noise dims)
  metrics.py    accuracy, recall, prototype cosine, margins
experiments/
  common.py                    shared runner (run_configs) + plotting helpers
  exp1_fewshot_ability.py      Q1: accuracy vs shots vs full pipelines + encode budget
  exp2_breakage.py             Q2: overlap / label noise / capacity / imbalance /
                               contamination / irrelevant features
  exp3_prototype_samples.py    Q3: prototype fidelity, margins, shots-to-target
  exp4_robustness.py           Q4: input noise / domain shift / support shift /
                               prior shift / imbalance policies
  make_summaries.py            results/*.csv -> results/summary.md
  run_all.py                   quick or --full batch driver
tests/
  test_sanity.py               encoder/prototype/loss/update/buffer mechanics
  test_empirical.py            Q1-Q3 signature tests (fast, fixed seeds)
  test_robustness.py           Q4 signature tests
results/                       CSVs, figures, generated summary.md
```

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests -q          # all tests (21)
python experiments/run_all.py      # quick suite (~30 s)
python experiments/run_all.py --full
python experiments/make_summaries.py
```

Individual experiments accept knobs such as `--seeds`, `--hd-dim`,
`--encoder rp|idlevel`, `--epochs`, `--buffer-fraction`, `--shots`;
`exp2` adds `--capacity-dims`, `--imbalance-ratios`, `--contam-modes`,
`--noise-dims`; `exp3` adds `--snrs`, `--dim`, `--classes`, `--targets`,
`--sweep*`; `exp4` adds `--noise-sigmas`, `--shift-severities`,
`--prior-skews`, `--imbalance-ratios`.

# Findings (full runs)

Setup unless stated otherwise: `C=5`, `d=32`, 100 train / 50 test per class,
`q=10000`, 20 retraining epochs, `lr=1`, buffer 5% (hard:random 50:50),
5 seeds. Encode budgets at these settings: B = 10500 samples, C = 1000
(9.5% of B), 50-shot A = 250 (2.4% of B).

## Q1 -- few-shot ability (`exp1_fewshot.csv`)

Mean test accuracy (chance = 0.20):

| SNR | 1 shot | 2 | 5 | 10 | 20 | 50 | B full | C buffer |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.205 | 0.192 | 0.219 | 0.259 | 0.269 | 0.286 | 0.270 | 0.260 |
| 0.50 | 0.231 | 0.259 | 0.334 | 0.429 | 0.452 | 0.522 | 0.516 | 0.482 |
| 1.00 | 0.418 | 0.597 | 0.716 | 0.809 | 0.860 | 0.880 | 0.882 | 0.882 |

* At SNR = 1, one shot already reaches 0.42 (2.1x chance) and 20-50 shots land
  on the full-retrain accuracy: the mean prototype is extremely sample
  efficient at this separability.
* At SNR = 0.5, ~50 shots match B; at SNR = 0.25 the task is at chance for
  every pipeline: the few-shot ceiling is set by class overlap, not by sample
  count (Q3 shows why).
* C matches B within 0.004-0.034 at 9.5% of the encode budget. The `idlevel`
  encoder reproduces the same picture at full settings (SNR 1: 50-shot 0.880,
  B 0.884, C 0.877; `rp`: 0.880 / 0.882 / 0.882).

## Q2 -- where it breaks (`exp2_*.csv`)

### Class overlap

| SNR | A few-shot | B full | B0 full-init | C buffer | C-rand |
|---|---|---|---|---|---|
| 0.10 | 0.196 | 0.211 | 0.212 | 0.232 | 0.205 |
| 0.35 | 0.258 | 0.298 | 0.398 | 0.377 | 0.366 |
| 0.60 | 0.400 | 0.608 | 0.635 | 0.592 | 0.616 |
| 1.00 | 0.716 | 0.882 | 0.888 | 0.882 | 0.886 |
| 2.00 | 0.994 | 0.998 | 0.998 | 0.998 | 0.998 |

A loses 12-21 points in the intermediate regime (0.35-0.6) where few samples
cannot span the class distribution, and the gap persists at SNR = 1 (0.716 vs
0.882). Overlap is the primary few-shot failure axis.

### Label noise (flipped training labels)

| flip frac | A few-shot | B full | B0 full-init | C hard+random | C-rand |
|---|---|---|---|---|---|
| 0.00 | 0.716 | 0.882 | 0.888 | 0.882 | 0.886 |
| 0.10 | 0.730 | 0.821 | 0.887 | 0.810 | 0.880 |
| 0.20 | 0.679 | 0.738 | 0.876 | 0.690 | 0.838 |
| 0.40 | 0.507 | 0.497 | 0.854 | 0.446 | 0.596 |

* Retraining destroys accuracy under label noise while the plain full-data
  means (B0) stay at 0.85 -- the strongest breakage found. Paired over all
  noise levels and seeds (`results/summary.md`): B - B0 = -0.124 (p = 1e-4),
  C - B = -0.021 (p = 0.033), A - B = -0.076 (p = 1e-4).
* **Hard-sample mining amplifies noise**: C (0.446) < B (0.497) << C-rand
  (0.596) << B0 (0.854) at 40% flips. "Hard" examples are usually mislabelled
  ones, so the buffer keeps re-learning label noise. A random buffer is the
  robust choice whenever labels may be unreliable.

### Encoder capacity (10 classes)

| q | A | B | B0 | C |
|---|---|---|---|---|
| 64 | 0.418 | 0.533 | 0.599 | 0.551 |
| 256 | 0.570 | 0.740 | 0.761 | 0.742 |
| 1024 | 0.632 | 0.810 | 0.822 | 0.810 |
| 4096 | 0.649 | 0.827 | 0.839 | 0.827 |

All pipelines collapse below `q ~ 256`; few-shot is the most sensitive (its
estimate noise adds to the interference floor). At `q >= 1024` accuracy
saturates.

### Class imbalance (long tail, counts [100, 50, 25, 12, 6] / [100, 25, 6, 4, 4])

| ratio | A acc | A rare | B acc | B rare | B0 acc | B0 rare | C acc | C rare |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.716 | 0.580 | 0.882 | 0.800 | 0.888 | 0.816 | 0.882 | 0.804 |
| 2 | 0.755 | 0.624 | 0.774 | 0.416 | 0.802 | 0.528 | 0.775 | 0.416 |
| 4 | 0.730 | 0.520 | 0.643 | 0.220 | 0.695 | 0.336 | 0.646 | 0.220 |

Rare-class recall collapses for the full-data pipelines -- already in the
initial means (B0 0.34 at ratio 4) and further under retraining (B/C 0.22) --
while balanced few-shot support holds 0.52 and even wins overall accuracy at
ratio 4 (0.730 vs 0.643). Balanced support is the property that protects rare
classes (see Q4 for natural support).

### Support-set contamination

| corrupted | A feature outliers | A label flips |
|---|---|---|
| 0.0 | 0.738 | 0.738 |
| 0.2 | 0.682 | 0.610 |
| 0.4 | 0.595 | 0.484 |

Flipping support labels is ~2x more damaging than feature outliers per
corrupted sample (the prototype is pulled towards another class HV). B/C are
unaffected because their pool is clean.

### Irrelevant features

| extra dims | A | B | B0 | C |
|---|---|---|---|---|
| 0 | 0.716 | 0.882 | 0.888 | 0.882 |
| 16 | 0.638 | 0.874 | 0.884 | 0.872 |
| 64 | 0.481 | 0.874 | 0.881 | 0.875 |
| 256 | 0.338 | 0.724 | 0.731 | 0.727 |

Few-shot breaks much earlier under feature dilution; full-data variants absorb
noise dimensions until `d_noise ~ q`.

## Q3 -- prototypes that need more samples (`exp3_*.csv`)

cos(few-shot prototype, oracle prototype) (oracle = class mean over 5000
clean samples from the same centroids):

| shots | SNR 0.25 | SNR 0.5 | SNR 1 | SNR 2 |
|---|---|---|---|---|
| 1 | 0.632 | 0.676 | 0.765 | 0.871 |
| 4 | 0.847 | 0.877 | 0.924 | 0.963 |
| 16 | 0.952 | 0.964 | 0.979 | 0.990 |
| 64 | 0.988 | 0.991 | 0.995 | 0.998 |

Mean decision margin `cos(correct) - max cos(wrong)`:

| shots | SNR 0.25 | SNR 0.5 | SNR 1 | SNR 2 |
|---|---|---|---|---|
| 1 | -0.061 | -0.047 | -0.001 | +0.059 |
| 8 | -0.029 | -0.011 | +0.032 | +0.085 |
| 64 | -0.010 | +0.001 | +0.038 | +0.088 |

Shots to target accuracy (log-interpolated means):

| target | SNR 0.5 | SNR 1 | SNR 2 | d=32 | d=64 | C=2 | C=5 | C=10 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 24.9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.2 |
| 0.8 | n/a | 7.1 | 1.0 | 6.9 | 2.1 | 1.5 | 6.9 | 22.4 |
| 0.9 | n/a | n/a | 1.0 | n/a | 4.5 | 8.4 | n/a | n/a |

* **Fidelity saturates long before usability.** At SNR 0.25, cosine to the
  oracle reaches 0.95+ at 16 shots but the margin stays negative even at 64
  shots: a prototype can look exactly like the infinite-data prototype and
  still be unseparated from its neighbours.
* Sample complexity for a *positive margin*: 1 shot at SNR 2, 2 at SNR 1,
  ~32-64 at SNR 0.5, never within 64 at SNR 0.25.
* More features help (shots-to-0.8: 6.9 at d=32 -> 2.1 at d=64; unreachable at
  d<=16), more classes hurt (1.5 -> 6.9 -> 22.4 shots for C = 2 -> 5 -> 10):
  prototype interference is a first-order driver of sample complexity.

## Q4 -- robustness (`exp4_*.csv`)

### Input noise (test-time Gaussian perturbation, clean training)

| sigma | A | B | B0 | C | C-rand |
|---|---|---|---|---|---|
| 0.00 | 0.716 | 0.882 | 0.888 | 0.882 | 0.886 |
| 0.20 | 0.630 | 0.800 | 0.806 | 0.794 | 0.807 |
| 0.40 | 0.495 | 0.639 | 0.654 | 0.633 | 0.635 |
| 0.80 | 0.346 | 0.394 | 0.418 | 0.404 | 0.415 |

All pipelines degrade smoothly; A is consistently the weakest because its
prototypes are noisier (paired over all noise levels: A - B = -0.141,
p < 1e-4), and the full-data variants are indistinguishable from each other
(C - B = -0.001, p = 0.53): retraining is not a robustness mechanism here.

### Domain shift (test-domain transform, graded severity)

| shift | sev | A | B | B0 | C |
|---|---|---|---|---|---|
| mean | 2 | 0.706 | 0.852 | 0.862 | 0.858 |
| mean | 4 | 0.645 | 0.799 | 0.821 | 0.800 |
| noise | 2 | 0.394 | 0.483 | 0.490 | 0.488 |
| scale | 2 | 0.448 | 0.614 | 0.598 | 0.610 |
| scale | 4 | 0.398 | 0.502 | 0.497 | 0.502 |

Translation is a mild shift for sign-linear encoders (shared additive term);
variance inflation and per-feature gains are the harsh axes. Again A degrades
first, while B/B0/C track each other.

### Few-shot support adaptation to a shifted domain

| shift | sev | source-support A | shifted-support A | B (source-trained) | C |
|---|---|---|---|---|---|
| scale | 1 | 0.579 | 0.688 | 0.726 | 0.730 |
| scale | 2 | 0.434 | 0.595 | 0.568 | 0.574 |
| scale | 4 | 0.338 | 0.476 | 0.432 | 0.426 |
| mean | 4 | 0.666 | 0.758 | 0.762 | 0.762 |

**Deployment-domain labels beat source-domain data volume**: at scale shift 2,
25 target-domain support samples lift A from 0.434 to 0.595 -- above the
source-trained full-data pipelines (0.568/0.574) -- and at severity 4 they stay
ahead (0.476 vs 0.432). This is the direct synthetic analogue of HyperLiDAR's
few-shot post-deployment adaptation.

### Class-prior shift (test stream dominated by two classes)

| prior skew | A | B | B0 | C |
|---|---|---|---|---|
| 0.0 acc | 0.730 | 0.890 | 0.899 | 0.893 |
| 0.9 acc | 0.718 | 0.885 | 0.891 | 0.879 |
| 0.0 macro recall | 0.734 | 0.891 | 0.899 | 0.893 |
| 0.9 macro recall | 0.764 | 0.914 | 0.924 | 0.911 |

Cosine prototypes are **prior-free**: a 10x class-prior skew changes the stream
accuracy by at most ~1-2 points and leaves per-class behaviour unchanged (macro
recall is stable or rises with the mix). No prior calibration mechanism is
needed; the caveat is that prototype classifiers cannot exploit priors either.

### Imbalance: balanced vs natural support

| ratio | A bal acc | A nat acc | A bal rare | A nat rare | B rare | B0 rare |
|---|---|---|---|---|---|---|
| 1 | 0.746 | 0.700 | 0.588 | 0.420 | 0.800 | 0.816 |
| 2 | 0.746 | 0.466 | 0.596 | 0.016 | 0.416 | 0.528 |
| 4 | 0.748 | 0.334 | 0.572 | 0.000 | 0.220 | 0.336 |
| 8 | 0.709 | 0.254 | 0.508 | 0.004 | 0.176 | 0.312 |

Sampling the support *naturally* (proportional to class frequencies) is
catastrophic under a long tail: overall accuracy drops to chance at ratio 8
and rare-class recall goes to zero. Balanced support keeps both flat. The
few-shot recipe's rare-class advantage is entirely a property of *how the
support is sampled* (also visible in the balanced-support vs full-pipeline
comparison in Q2).

## Main takeaways

1. **Few-shot HDC works when classes are separable, and saturates onto full
   retraining there.** At SNR = 1, 1 shot gives 0.42 and 20-50 shots match the
   10,500-sample full pipeline (0.880 vs 0.882) using 2.4% of its encode
   budget. Below SNR ~ 0.5 the ceiling is overlap, not sample count.
2. **Buffered retraining is an accurate and cheap stand-in for full retraining
   on clean, stationary data**: within 0.4-3.4 points at 9.5% of the encode
   budget, across every stress test except label noise.
3. **Retraining rarely earns its cost on stationary synthetic tasks.** Paired
   over seeds, B is significantly *worse* than B0 (no retraining) in every
   Exp-2 stress test (mean differences -0.007 to -0.124, p <= 0.034): the
   error-driven pull/push on already-fit prototypes mostly encodes noise. Its
   value would have to come from non-stationarity, which these tests do not
   model.
4. **Hard-sample mining and label noise are incompatible.** At 40% flipped
   labels the hard+random buffer reaches 0.446 while a random buffer reaches
   0.596 and the un-retrained full means stay at 0.854. Any deployment using
   hard mining should validate the hard fraction against label quality.
5. **Balanced support is the key to long tails.** Full-data pipelines collapse
   rare-class recall (0.22 at ratio 4; B0 0.34) while balanced few-shot support
   holds 0.52-0.57 and wins overall accuracy at ratio 4. Natural support
   sampling destroys even few-shot prototypes (rare recall 0.00, accuracy
   0.25-0.33). Prototype quality under imbalance is a property of the support
   distribution, not of the HDC readout.
6. **Failure ordering is consistent across axes**: few-shot prototypes are the
   most fragile under feature noise, domain shift, irrelevant features and
   support contamination; full-data variants differ mostly through
   susceptibility to label noise. All pipelines collapse at small `q`, at
   severe overlap (SNR <= 0.1) and under strong within-class shifts.
7. **Prototype fidelity is necessary but not sufficient.** Cosine to the oracle
   reaches 0.95+ long before the decision margin becomes positive. Shots to
   80% accuracy fall with feature count (6.9 -> 2.1 for d = 32 -> 64) and rise
   with class count (1.5 -> 6.9 -> 22.4 for C = 2 -> 5 -> 10).
8. **Target-domain labels are the most valuable resource under shift.** A
   25-sample target-domain support beats source-trained full-data pipelines
   under moderate/strong per-feature shift, matching the HyperLiDAR adaptation
   premise. Prototypes are prior-free, so class-prior shift needs no
   correction (and cannot be exploited either).

## Deviations and limitations

* **Synthetic data only**: isotropic-ish Gaussian clusters with a global random
  projection on `[0,1]` features. Absolute accuracies do not transfer to LiDAR
  backbones; the mechanisms (prototype noise, interference, support sampling)
  do.
* **No non-stationarity / drift**, no unlabeled inference-time updates, no
  subcluster/density prototypes. These are exactly where retraining and
  hard-sample mining should start to pay off, and they are the natural next
  experiments.
* **Buffer selection is global per epoch** over the pooled synthetic train set,
  not per scan with `PERCENTAGE`; `wrong_first` reproduces the reference rule.
* **Initial-pass scoring** differs slightly: the reference scores with the
  weights available at that batch, we score each batch right after its own
  accumulation.
* **Encodings are precomputed** and indexed, so the reported encode budget is
  exact but wall-clock adaptation latency is not measured.
* **`idlevel` normalisation** uses train min/max bounds (reference expects
  backbone features already normalised); shifted test inputs are clipped to
  `[0,1]` for both encoders, which can soften extreme shifts.

## Reproducibility

* Python 3.14, NumPy 2.5, pandas 3.0, matplotlib 3.11; CPU only.
* Full suite: **~4.5 minutes** (exp1 15 s, exp2 76 s, exp3 84 s, exp4 84 s);
  quick suite ~30 s.
* 5 seeds per cell (exp3 sweep: 3); all dataset/encoder/training seeds derive
  from the cell seed and are recorded in the CSVs.
* Raw tables: `results/exp*.csv`; generated summary
  [`results/summary.md`](results/summary.md); figures `results/fig_*.png`.
* Tests: `python -m pytest tests -q` (21 tests: mechanics, Q1-Q3 signatures,
  Q4 robustness signatures).
