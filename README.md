# FewShotHDC-empirical

Empirical probes of **few-shot learning with hyperdimensional computing (HDC)
prototypes**, using synthetic datasets and a target/update implementation that
follows the HyperLiDAR reference model.

The repo answers three questions:

1. **How good is few-shot HDC?** How does a prototype built from `k` labelled
   samples per class compare with the full retraining pipeline?
2. **Where does it break?** Class overlap, label noise, hypervector capacity,
   class imbalance, support contamination, irrelevant features.
3. **Which prototype properties need more samples?** Prototype fidelity to the
   infinite-data prototype, decision margins, and shots-to-target scaling with
   overlap, feature count and class count.

A short write-up of the quick-run results is in
[`results/analysis.md`](results/analysis.md); figures and CSVs are in
[`results/`](results/).

## The three configurations

All configurations share the same target model
(`modules/hd.py:HDCModel`): class hypervectors are **sum accumulators**
(`classify_weights`, the reference `Model.classify_weights`) and predictions
use cosine similarity against their row-normalised copies
(`logits = normalize(h) @ classify.T`, the reference `Model.forward`).

| Config | Training data | Prototype construction | Encode budget |
|--------|---------------|------------------------|---------------|
| **A** `fewshot` | `k` labelled samples/class | `c_y = normalize(sum of support encodings)` (uniform "weighted average"; per-sample weights optional) | `C·k` |
| **B** `full` | full training pool | full-data class means, then `epochs` error-driven passes: for each mistake `c_true += lr·h`, `c_pred -= lr·h` (normalise per batch, like `Basic_HD.retrain`) | `N·(1+epochs)` |
| **C** `buffer` | full pool + buffers | one full-data initial pass (accumulate + score + loss), then `epochs` passes over `k%` of samples: `hard_fraction` of the buffer = highest stored loss, rest random from the remainder; buffer losses refreshed after each pass, others stay stale (paper Sec. 4.3) | `N + epochs·k·N` |

The classification loss used for hard-sample mining is the reference
`Basic_HD` definition, `cos(h, c_pred) - cos(h, c_true)` for mistakes (0 for
correct predictions). It is the negative of the paper's Eq. (4) sign, chosen so
that "highest loss = hardest sample".

Exp 2 also reports a **B0** reference (full-data means *without* retraining) and
a **C-rand** control (random-only buffer) to separate the effect of retraining
from the effect of hard-sample mining.

## Fidelity to the reference implementation

The target/update semantics follow
`Old/HyperLiDAR_TTA/DensityUnsupHyperLidar/modules/{HDC_utils,Basic_HD,HDC_cl}.py`:

* sum-of-encodings accumulators + cosine readout (`Model.classify_weights` /
  `Model.classify`);
* retraining updates only misclassified samples (reference default);
  `lr = 1` matches `HDC_cl._train_retrain`, `lr = 2` matches the double
  `index_add_` in `Basic_HD.retrain`;
* per-batch prototype normalisation before scoring;
* encoders: `rp` (QR-orthogonalised Gaussian projection + `hard_quantize`,
  reference default `gauss_rp=True`) and `idlevel` (position ⊗ level, bundle,
  `hard_quantize`).

Deliberate simplifications (documented in code):

* the initial pass accumulates all samples and scores each batch after its own
  accumulation (encode-once); the reference scores with the weights available at
  that batch;
* buffer selection is global over the pooled synthetic train set per epoch,
  rather than per scan/batch with `PERCENTAGE`; `selection="wrong_first"` in
  `modules/training.py` reproduces the reference "wrong samples first, random
  fill" rule;
* encodings are precomputed once and indexed, so the reported
  `n_encoded` / `n_encoded_rel` is the *equivalent deployment encode budget*
  (encoding is deterministic, so this is semantically identical and much
  faster);
* no subclusters/density model, no inference-time (unlabeled) updates, no
  domain shift.

## Repository layout

```
modules/
  encoders.py   # rp / idlevel encoders (bipolar, deterministic)
  hd.py         # HDCModel: accumulators, cosine readout, loss, error updates
  training.py   # initial / full-retrain / buffered-retrain passes + buffer selection
  data.py       # synthetic Gaussian tasks (SNR, imbalance, label noise, noise dims)
  metrics.py    # accuracy, per-class/macro recall, prototype cosine, margins
experiments/
  common.py                    # shared runner + reporting helpers
  exp1_fewshot_ability.py      # Q1: accuracy vs shots vs full pipelines + budgets
  exp2_breakage.py             # Q2: six stress tests
  exp3_prototype_samples.py    # Q3: prototype fidelity / margins / sample complexity
  run_all.py                   # quick or --full batch driver
tests/
  test_sanity.py               # encoder/prototype/loss/update/buffer mechanics
  test_empirical.py            # fast fixed-seed versions of the three questions
results/                       # generated CSVs, figures, analysis.md
```

## Quickstart

```bash
pip install -r requirements.txt

# mechanics + fast empirical checks
python tests/test_sanity.py          # or: pytest tests -q
python tests/test_empirical.py

# experiments (quick defaults, a few seconds each)
python experiments/run_all.py
# or individually
python experiments/exp1_fewshot_ability.py --seeds 3 --hd-dim 4096
python experiments/exp2_breakage.py --seeds 3 --hd-dim 4096
python experiments/exp3_prototype_samples.py --seeds 3 --hd-dim 2048 \
    --oracle-per-class 2000 --sweep --sweep-seeds 2

# larger paper-style sweeps
python experiments/run_all.py --full
```

Every experiment writes one CSV per setting plus figures to `results/`.
Useful knobs: `--seeds`, `--hd-dim`, `--encoder rp|idlevel`, `--epochs`,
`--buffer-fraction`, `--shots`; `exp2` also exposes `--capacity-dims`,
`--imbalance-ratios`, `--contam-modes feature|label`, `--noise-dims`;
`exp3` exposes `--snrs`, `--dim`, `--classes`, `--targets` and the sweep lists.

## Synthetic tasks

`modules/data.py:make_gaussian` builds `C` Gaussian classes with centroids
`0.5 ± separation`, per-feature signal-to-noise ratio `snr`, optional
long-tailed `class_counts`, `label_noise`, `n_noise_dims` irrelevant features,
and optional shared `centroids` (used to sample oracle prototypes from the
*same* class distributions). The separability index grows like `0.5·d·snr²`,
so difficulty can be controlled independently of the feature count.

## What the quick runs show (details in `results/analysis.md`)

* **Q1**: at SNR = 1, 1-shot reaches 0.41 vs 0.20 chance and 20-50 shots match
  full retraining (0.87); the buffered pipeline matches full retraining with
  ~14% of the encode budget. At SNR ≤ 0.25 everything is at chance.
* **Q2**: few-shot degrades earlier under overlap, irrelevant features and
  support corruption; retraining on label noise *hurts* (and hard-sample mining
  hurts most: 0.45 vs 0.75 for a random buffer at 40% flips); long-tailed pools
  collapse rare-class recall for the full pipelines (0.17) while balanced
  few-shot support keeps 0.56; small `q` causes interference collapse for all.
* **Q3**: prototype cosine to the oracle saturates quickly (≈0.99 at 64 shots)
  but the decision margin only becomes positive with overlap-dependent sample
  counts (1 shot at SNR 2 → 2 at SNR 1 → ~32-64 at SNR 0.5 → never at SNR 0.25
  within 64); shots-to-80%-accuracy scales down with feature count and up with
  class count (1.6 / 8.9 / 40.3 shots for 2 / 5 / 10 classes at SNR 1).

## Natural extensions

* drift / stream non-stationarity (where retraining should start to pay off);
* inference-time unlabeled updates (`Model.inference_update`-style pulls) and
  subcluster prototypes;
* support-set weighting/robustification (margin or confidence weights are
  already supported by `HDCModel.accumulate`);
* mixed buffer schedules (hard fraction decay, loss re-estimation on a random
  probe) to keep hard mining useful under label noise.
