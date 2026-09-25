"""Exp 4: robustness to input noise, domain shift, prior shift and imbalance.

Five stress tests, all comparing the three configurations (few-shot A, full
retrain B, buffered retrain C, plus the full-data-mean reference B0):

* ``feature_noise``:  Gaussian noise added to the test inputs (measurement
  noise at deployment; models are trained on clean data).
* ``domain_shift``:   covariate shift of the test domain: (a) mean shift along
  a random direction, (b) variance inflation (extra Gaussian noise), graded in
  units of the training within-class noise std.
* ``support_shift``:  few-shot support drawn from the shifted domain while the
  test set is shifted too (does per-deployment support adaptation recover the
  accuracy the source-support prototypes lose?).
* ``prior_shift``:    test-time class-prior skew (two classes dominate the
  test stream); macro recall is the metric that matters.
* ``imbalance``:      long-tailed training pools with *balanced* vs *natural*
  few-shot support sampling.

Outputs: one CSV + figure per subtest in ``results/``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, basic_rows, encode_dataset, line_plot, run_configs
from modules import RetrainConfig, make_encoder, make_gaussian, sample_support


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subtests", nargs="+",
                    default=["feature_noise", "domain_shift", "support_shift",
                             "prior_shift", "imbalance"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--levels", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-fraction", type=float, default=0.05)
    ap.add_argument("--snr", type=float, default=1.0)
    ap.add_argument("--encoder", default="rp")
    ap.add_argument("--noise-sigmas", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--shift-severities", nargs="+", type=float,
                    default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--prior-skews", nargs="+", type=float, default=[0.0, 0.5, 0.9])
    ap.add_argument("--prior-n", type=int, default=250,
                    help="size of the prior-shifted test stream")
    ap.add_argument("--imbalance-ratios", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--min-class-count", type=int, default=4)
    ap.add_argument("--tag", type=str, default="")
    return ap


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _retrain(args, seed: int) -> RetrainConfig:
    return RetrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                         buffer_fraction=args.buffer_fraction, seed=seed)


def _base(args, seed: int, name: str, **ds_kw):
    ds = make_gaussian(args.classes, args.dim, args.n_train, args.n_test,
                       snr=args.snr, seed=seed, name=name, **ds_kw)
    enc = make_encoder(args.encoder, args.hd_dim, seed=seed,
                       **({"levels": args.levels} if args.encoder == "idlevel" else {}))
    enc.fit(ds.x_train)
    return ds, enc, enc.encode(ds.x_train), enc.encode(ds.x_test)


def _save(df: pd.DataFrame, name: str, tag: str) -> None:
    path = RESULTS / f"{name}{tag}.csv"
    df.to_csv(path, index=False)
    print(f"saved {path}")


def _summary(df: pd.DataFrame, index, value="acc") -> pd.DataFrame:
    return df.pivot_table(index=index, columns="config", values=value,
                          aggfunc="mean").round(3)


def _with_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cfg, mode = df["config"].astype(str), df.get("support_mode", pd.Series("", index=df.index))
    df["policy"] = np.where((cfg == "fewshot") & (mode != ""),
                            "fewshot_" + mode.astype(str), cfg)
    return df


def _natural_support(y: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    """Long-tail support: one sample per class, the rest ∝ class frequency."""
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    picks = [np.atleast_1d(rng.choice(np.flatnonzero(y == c))) for c in classes]
    remaining = max(0, total - sum(len(p) for p in picks))
    if remaining > 0:
        counts = np.bincount(y, minlength=int(classes.max()) + 1).astype(np.float64) + 1.0
        w = counts[y]
        p = w / w.sum()
        extra = rng.choice(len(y), size=remaining, replace=False, p=p)
        picks.append(extra)
    return np.concatenate(picks).astype(np.int64)


# --------------------------------------------------------------------------- #
# subtests
# --------------------------------------------------------------------------- #
def subtest_feature_noise(args):
    rows = []
    for sigma in args.noise_sigmas:
        for seed in range(args.seeds):
            ds, enc, Htr, Hte = _base(args, seed, "feature_noise")
            rng = np.random.default_rng([seed, int(round(sigma * 1000))])
            x_noisy = ds.x_test if sigma <= 0 else np.clip(
                ds.x_test + sigma * rng.standard_normal(ds.x_test.shape), 0.0, 1.0).astype(np.float32)
            rows += basic_rows(ds, enc, _retrain(args, seed), args.shots, seed,
                               {"test_noise": sigma}, H=(Htr, enc.encode(x_noisy)),
                               selections=("hard_random", "random"))
            print(f"[feature_noise sigma={sigma:g} seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp4_feature_noise", args.tag)
    print(_summary(df, ["test_noise"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    line_plot(axes[0], df, "test_noise", "acc",
              xlabel="test-input noise sigma", chance=1 / args.classes,
              title="accuracy under measurement noise")
    line_plot(axes[1], df, "test_noise", "macro_recall",
              xlabel="test-input noise sigma", title="macro recall")
    fig.tight_layout()
    out = RESULTS / f"fig_exp4_feature_noise{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def _shift_transform(ds, kind: str, severity: float, rng: np.random.Generator):
    """Return a test-domain transform and its description for one shift kind."""
    if kind == "mean":  # translation along a random direction
        direction = rng.standard_normal(ds.dim).astype(np.float32)
        direction /= max(np.linalg.norm(direction), 1e-9)
        step = severity * float(ds.meta["noise_std"])
        return lambda x: x + step * direction
    if kind == "noise":  # variance inflation
        step = severity * float(ds.meta["noise_std"])
        return lambda x: x + step * rng.standard_normal(x.shape).astype(np.float32)
    if kind == "scale":  # per-feature gain (covariance shift)
        gain = (1.0 + severity * rng.uniform(-1.0, 1.0, size=ds.dim)).astype(np.float32)
        return lambda x: x * gain
    raise ValueError(kind)


def subtest_domain_shift(args):
    rows = []
    for kind in ("mean", "noise", "scale"):
        for severity in args.shift_severities:
            for seed in range(args.seeds):
                ds, enc, Htr, Hte = _base(args, seed, "domain_shift")
                rng = np.random.default_rng([seed, 17])
                transform = _shift_transform(ds, kind, severity, rng)
                x_shift = np.clip(transform(ds.x_test), 0.0, 1.0).astype(np.float32)
                rows += basic_rows(ds, enc, _retrain(args, seed), args.shots, seed,
                                   {"shift_kind": kind, "shift_severity": severity},
                                   H=(Htr, enc.encode(x_shift)))
                print(f"[domain_shift {kind} sev={severity:g} seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp4_domain_shift", args.tag)
    print(_summary(df, ["shift_kind", "shift_severity"]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    for ax, kind in zip(axes, ("mean", "noise", "scale")):
        line_plot(ax, df[df.shift_kind == kind], "shift_severity", "acc",
                  xlabel=f"shift severity ({kind})", logx=False,
                  chance=1 / args.classes, title=f"{kind} shift")
    fig.tight_layout()
    out = RESULTS / f"fig_exp4_domain_shift{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def subtest_support_shift(args):
    """Few-shot support from the shifted domain vs source support."""
    rows = []
    for kind in ("mean", "scale"):
        for severity in args.shift_severities:
            for seed in range(args.seeds):
                ds, enc, Htr, Hte = _base(args, seed, "support_shift")
                rng = np.random.default_rng([seed, 23])
                transform = _shift_transform(ds, kind, severity, rng)
                x_test_shift = np.clip(transform(ds.x_test), 0.0, 1.0).astype(np.float32)
                Hte_shift = enc.encode(x_test_shift)
                cfg = _retrain(args, seed)
                support = sample_support(ds.y_train, args.shots,
                                         np.random.default_rng(seed + 31))
                x_sup_shift = np.clip(transform(ds.x_train[support]), 0.0, 1.0).astype(np.float32)
                extra = {"shift_kind": kind, "shift_severity": severity}
                # full-data references see the shifted test domain too
                r, _ = run_configs(ds, enc, shots=args.shots, retrain=cfg, seed=seed,
                                   configs=("full_init", "full", "buffer"),
                                   selection="hard_random",
                                   extra={**extra, "support_domain": "none"},
                                   H=(Htr, Hte_shift))
                rows += r
                r, _ = run_configs(ds, enc, shots=args.shots, retrain=cfg, seed=seed,
                                   configs=("fewshot",), support=support,
                                   extra={**extra, "support_domain": "source"},
                                   H=(Htr, Hte_shift))
                rows += r
                r, _ = run_configs(ds, enc, shots=args.shots, retrain=cfg, seed=seed,
                                   configs=("fewshot",), support=support,
                                   support_x=x_sup_shift,
                                   extra={**extra, "support_domain": "shifted"},
                                   H=(Htr, Hte_shift))
                rows += r
                print(f"[support_shift {kind} sev={severity:g} seed={seed}] done",
                      flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp4_support_shift", args.tag)
    fs = df[df.config == "fewshot"]
    print(fs.pivot_table(index=["shift_kind", "shift_severity"],
                         columns="support_domain", values="acc", aggfunc="mean").round(3))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, kind in zip(axes, ("mean", "scale")):
        sub = fs[fs.shift_kind == kind]
        line_plot(ax, sub, "shift_severity", "acc", hue="support_domain", logx=False,
                  xlabel=f"test-domain {kind} shift (severity)",
                  chance=1 / args.classes,
                  title=f"support adaptation ({kind} shift)")
        ref = df[(df.shift_kind == kind)]
        for cfg, color, ls in (("full", "#d65f5f", "-"), ("buffer", "#6acc65", "--")):
            v = ref[ref.config == cfg].acc
            if len(v):
                ax.axhline(v.mean(), color=color, ls=ls, label=f"{cfg} (source-trained)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / f"fig_exp4_support_shift{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def subtest_prior_shift(args):
    """Class-prior shift at test time: two classes dominate the stream."""
    rows = []
    for skew in args.prior_skews:
        for seed in range(args.seeds):
            ds, enc, Htr, Hte = _base(args, seed, "prior_shift")
            rng = np.random.default_rng([seed, 41])
            w = np.ones(ds.num_classes)
            w[:2] = 1.0 + 9.0 * skew
            p_class = w / w.sum()
            counts = rng.multinomial(args.prior_n, p_class)
            idx = []
            for c in range(ds.num_classes):
                pool = np.flatnonzero(ds.y_test == c)
                idx.append(rng.choice(pool, size=int(counts[c]), replace=True))
            idx = np.concatenate(idx) if idx else np.array([], dtype=np.int64)
            ds_shift = replace(ds, y_test=ds.y_test[idx])
            rows += basic_rows(ds_shift, enc, _retrain(args, seed), args.shots, seed,
                               {"prior_skew": skew}, H=(Htr, Hte[idx]),
                               selections=("hard_random",))
            print(f"[prior_shift skew={skew:g} seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp4_prior_shift", args.tag)
    print("accuracy:")
    print(_summary(df, ["prior_skew"]))
    print("macro recall:")
    print(_summary(df, ["prior_skew"], value="macro_recall"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    line_plot(axes[0], df, "prior_skew", "acc",
              xlabel="prior skew (two dominant classes)", chance=1 / args.classes,
              title="stream accuracy")
    line_plot(axes[1], df, "prior_skew", "macro_recall", ylim=(0, 1.02),
              xlabel="prior skew (two dominant classes)", title="macro recall")
    fig.tight_layout()
    out = RESULTS / f"fig_exp4_prior_shift{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def subtest_imbalance(args):
    """Long-tailed pools: balanced vs natural few-shot support sampling."""
    rows = []
    for ratio in args.imbalance_ratios:
        counts = [max(args.min_class_count, int(round(args.n_train / ratio ** c)))
                  for c in range(args.classes)]
        for seed in range(args.seeds):
            ds, enc, Htr, Hte = _base(args, seed, "imbalance", class_counts=counts)
            cfg = _retrain(args, seed)
            rng = np.random.default_rng(seed + 53)
            sup_bal = sample_support(ds.y_train, args.shots, rng)
            sup_nat = _natural_support(ds.y_train, args.shots * args.classes,
                                       np.random.default_rng(seed + 71))
            extra = {"imbalance_ratio": ratio, "counts": str(counts)}
            r, _ = run_configs(ds, enc, shots=args.shots, retrain=cfg, seed=seed,
                               configs=("full_init", "full", "buffer"),
                               selection="hard_random",
                               extra={**extra, "support_mode": "none"}, H=(Htr, Hte))
            rows += r
            for mode, sup in (("balanced", sup_bal), ("natural", sup_nat)):
                r, _ = run_configs(ds, enc, shots=args.shots, retrain=cfg, seed=seed,
                                   configs=("fewshot",), support=sup,
                                   extra={**extra, "support_mode": mode}, H=(Htr, Hte))
                rows += r
            print(f"[imbalance ratio={ratio:g} counts={counts} seed={seed}] done",
                  flush=True)
    df = _with_policy(pd.DataFrame(rows))
    _save(df, "exp4_imbalance", args.tag)
    print("accuracy (by policy):")
    print(df.pivot_table(index="imbalance_ratio", columns="policy", values="acc",
                         aggfunc="mean").round(3))
    print("rarest-class recall (by policy):")
    print(df.pivot_table(index="imbalance_ratio", columns="policy",
                         values="min_class_recall", aggfunc="mean").round(3))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    line_plot(axes[0], df, "imbalance_ratio", "acc", hue="policy", logx=True,
              xlabel="class-count ratio between neighbours", chance=1 / args.classes,
              title="overall accuracy")
    line_plot(axes[1], df, "imbalance_ratio", "min_class_recall", hue="policy", logx=True,
              xlabel="class-count ratio between neighbours", title="rarest-class recall")
    fig.tight_layout()
    out = RESULTS / f"fig_exp4_imbalance{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


SUBTESTS = {
    "feature_noise": subtest_feature_noise,
    "domain_shift": subtest_domain_shift,
    "support_shift": subtest_support_shift,
    "prior_shift": subtest_prior_shift,
    "imbalance": subtest_imbalance,
}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    for name in args.subtests:
        if name not in SUBTESTS:
            raise SystemExit(f"unknown subtest {name!r} (options: {list(SUBTESTS)})")
        print(f"\n===== {name} =====", flush=True)
        SUBTESTS[name](args)


if __name__ == "__main__":
    main()
