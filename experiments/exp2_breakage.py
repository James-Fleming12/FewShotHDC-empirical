"""Exp 2 -- scenarios where few-shot / buffered HDC breaks.

Six synthetic stress tests, all compared across the three configurations
(few-shot A, full retrain B, buffered retrain C; C also in a random-only
buffer control where relevant):

* ``overlap``       -- class overlap (SNR sweep)
* ``label_noise``   -- fraction of flipped training labels
* ``capacity``      -- hypervector dimension vs number of classes (interference)
* ``imbalance``     -- long-tailed class counts (overall vs rare-class recall)
* ``contamination`` -- gross outliers inside the few-shot support set
* ``noise_dims``    -- irrelevant features added to every sample

Outputs: one CSV + figure per subtest in ``results/``.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, encode_dataset, run_configs
from modules import RetrainConfig, make_encoder, make_gaussian, sample_support

CONFIG_COLORS = {"fewshot": "#4878cf", "full": "#d65f5f", "buffer": "#6acc65",
                 "buffer_random": "#b47cc7", "full_init": "#8c8c8c"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subtests", nargs="+",
                    default=["overlap", "label_noise", "capacity", "imbalance",
                             "contamination", "noise_dims"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--levels", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--buffer-fraction", type=float, default=0.05)
    ap.add_argument("--overlap-snrs", nargs="+", type=float,
                    default=[0.1, 0.2, 0.35, 0.6, 1.0, 2.0])
    ap.add_argument("--noise-levels", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--capacity-dims", nargs="+", type=int,
                    default=[64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--capacity-classes", type=int, default=10)
    ap.add_argument("--imbalance-ratios", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    ap.add_argument("--min-class-count", type=int, default=4)
    ap.add_argument("--contam-fracs", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.4])
    ap.add_argument("--contam-modes", nargs="+", default=["feature", "label"])
    ap.add_argument("--noise-dims", nargs="+", type=int, default=[0, 16, 64, 256])
    ap.add_argument("--encoder", default="rp")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tag", type=str, default="")
    return ap


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _retrain(args, seed: int) -> RetrainConfig:
    return RetrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                         buffer_fraction=args.buffer_fraction, seed=seed)


def _basic_rows(ds, enc, cfg, shots, seed, extra, H, selections=("hard_random",)):
    rows, _ = run_configs(ds, enc, shots=shots, retrain=cfg, seed=seed,
                          configs=("fewshot", "full_init", "full", "buffer"),
                          selection="hard_random", extra=extra, H=H)
    out = list(rows)
    for sel in selections:
        if sel == "hard_random":
            continue
        r, _ = run_configs(ds, enc, shots=shots, retrain=cfg, seed=seed,
                           configs=("buffer",), selection=sel, extra=extra, H=H)
        out += r
    return out


def _line_plot(ax, df, x, y, hue="config", ylim=(0, 1.02), logx=False,
               ylabel="test accuracy", xlabel=None, chance=None, title=None):
    for key, sub in df.groupby(hue):
        agg = sub.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
        err = np.where(agg["count"] > 1,
                       1.96 * agg["std"].fillna(0.0) / np.sqrt(agg["count"]), 0.0)
        ax.errorbar(agg[x], agg["mean"], yerr=err, marker="o", capsize=3,
                    label=key, color=CONFIG_COLORS.get(key))
    if chance is not None:
        ax.axhline(chance, color="gray", ls=":", lw=1)
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)


def _summary(df: pd.DataFrame, group_cols, value="acc") -> pd.DataFrame:
    return df.pivot_table(index=group_cols, columns="config", values=value,
                          aggfunc="mean").round(3)


def _save(df, name, tag):
    path = RESULTS / f"{name}{tag}.csv"
    df.to_csv(path, index=False)
    print(f"saved {path}")


# --------------------------------------------------------------------------- #
# subtests
# --------------------------------------------------------------------------- #
def subtest_overlap(args):
    rows = []
    for snr in args.overlap_snrs:
        for seed in range(args.seeds):
            ds = make_gaussian(args.classes, args.dim, args.n_train, args.n_test,
                               snr=snr, seed=seed, name="overlap")
            enc = make_encoder(args.encoder, args.hd_dim, seed=seed,
                               **({"levels": args.levels} if args.encoder == "idlevel" else {}))
            H = encode_dataset(ds, enc)
            rows += _basic_rows(ds, enc, _retrain(args, seed), args.shots, seed,
                                {"snr": snr}, H, selections=("hard_random", "random"))
            print(f"[overlap snr={snr:g} seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp2_overlap", args.tag)
    print(_summary(df, ["snr"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    _line_plot(axes[0], df, "snr", "acc", ylabel="test accuracy",
               xlabel="SNR (per-feature class separation / noise)", logx=True,
               chance=1 / args.classes, title="class overlap")
    _line_plot(axes[1], df, "snr", "min_class_recall", ylabel="rarest-class recall",
               xlabel="SNR", logx=True, title="class overlap (rare class)")
    fig.tight_layout()
    out = RESULTS / f"fig_exp2_overlap{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def subtest_label_noise(args):
    rows = []
    for noise in args.noise_levels:
        for seed in range(args.seeds):
            ds = make_gaussian(args.classes, args.dim, args.n_train, args.n_test,
                               snr=1.0, seed=seed, label_noise=noise, name="label_noise")
            enc = make_encoder(args.encoder, args.hd_dim, seed=seed,
                               **({"levels": args.levels} if args.encoder == "idlevel" else {}))
            H = encode_dataset(ds, enc)
            rows += _basic_rows(ds, enc, _retrain(args, seed), args.shots, seed,
                                {"label_noise": noise}, H,
                                selections=("hard_random", "random"))
            print(f"[label_noise {noise:g} seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp2_label_noise", args.tag)
    print(_summary(df, ["label_noise"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    _line_plot(axes[0], df, "label_noise", "acc",
               xlabel="fraction of flipped training labels", chance=1 / args.classes,
               title="label noise")
    _line_plot(axes[1], df, "label_noise", "min_class_recall",
               xlabel="fraction of flipped training labels", title="rarest-class recall")
    fig.tight_layout()
    out = RESULTS / f"fig_exp2_label_noise{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def subtest_capacity(args):
    rows = []
    for q in args.capacity_dims:
        for seed in range(args.seeds):
            ds = make_gaussian(args.capacity_classes, args.dim, args.n_train,
                               args.n_test, snr=1.0, seed=seed, name="capacity")
            enc = make_encoder(args.encoder, q, seed=seed,
                               **({"levels": args.levels} if args.encoder == "idlevel" else {}))
            H = encode_dataset(ds, enc)
            rows += _basic_rows(ds, enc, _retrain(args, seed), args.shots, seed,
                                {"capacity_dim": q, "capacity_classes": args.capacity_classes},
                                H, selections=("hard_random",))
            print(f"[capacity q={q} seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp2_capacity", args.tag)
    print(_summary(df, ["capacity_dim"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    _line_plot(axes[0], df, "capacity_dim", "acc", logx=True,
               xlabel="hypervector dimension q", chance=1 / args.capacity_classes,
               title=f"encoder capacity ({args.capacity_classes} classes)")
    _line_plot(axes[1], df, "capacity_dim", "min_class_recall", logx=True,
               xlabel="hypervector dimension q", title="rarest-class recall")
    fig.tight_layout()
    out = RESULTS / f"fig_exp2_capacity{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def subtest_imbalance(args):
    rows = []
    for ratio in args.imbalance_ratios:
        counts = [max(args.min_class_count, int(round(args.n_train / ratio ** c)))
                  for c in range(args.classes)]
        for seed in range(args.seeds):
            ds = make_gaussian(args.classes, args.dim, n_train=args.n_train,
                               n_test=args.n_test, snr=1.0, seed=seed,
                               class_counts=counts, name="imbalance")
            enc = make_encoder(args.encoder, args.hd_dim, seed=seed,
                               **({"levels": args.levels} if args.encoder == "idlevel" else {}))
            H = encode_dataset(ds, enc)
            rows += _basic_rows(ds, enc, _retrain(args, seed), args.shots, seed,
                                {"imbalance_ratio": ratio, "counts": str(counts)}, H,
                                selections=("hard_random",))
            print(f"[imbalance ratio={ratio:g} counts={counts} seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp2_imbalance", args.tag)
    print(_summary(df, ["imbalance_ratio"]))
    print("\nmin-class recall:")
    print(_summary(df, ["imbalance_ratio"], value="min_class_recall"))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    _line_plot(axes[0], df, "imbalance_ratio", "acc",
               xlabel="class-count ratio between neighbours", chance=1 / args.classes,
               title="overall accuracy")
    _line_plot(axes[1], df, "imbalance_ratio", "min_class_recall",
               xlabel="class-count ratio between neighbours", title="rarest-class recall")
    fig.tight_layout()
    out = RESULTS / f"fig_exp2_imbalance{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def subtest_contamination(args):
    """Corrupted support samples: feature outliers or flipped support labels."""
    rows = []
    for seed in range(args.seeds):
        ds = make_gaussian(args.classes, args.dim, args.n_train, args.n_test,
                           snr=1.0, seed=seed, name="contamination")
        enc = make_encoder(args.encoder, args.hd_dim, seed=seed,
                           **({"levels": args.levels} if args.encoder == "idlevel" else {}))
        H = encode_dataset(ds, enc)
        cfg = _retrain(args, seed)
        support = sample_support(ds.y_train, args.shots, np.random.default_rng(seed + 7))
        y_sup = ds.y_train[support]
        # reference pipelines: untouched by support corruption
        r, _ = run_configs(ds, enc, shots=args.shots, retrain=cfg, seed=seed,
                           configs=("full_init", "full", "buffer"),
                           selection="hard_random",
                           extra={"contam_frac": np.nan, "contam_mode": "none"}, H=H)
        rows += r
        x_sup = ds.x_train[support].copy()
        rng = np.random.default_rng(seed + 99)
        order = rng.permutation(len(support))
        for mode in args.contam_modes:
            for frac in args.contam_fracs:
                n_bad = int(round(frac * len(support)))
                bad = order[:n_bad]
                x_corrupt, y_corrupt = None, None
                if mode == "feature":
                    x_corrupt = x_sup.copy()
                    if n_bad:
                        x_corrupt[bad] = rng.random((n_bad, ds.dim)).astype(np.float32)
                elif mode == "label":
                    y_corrupt = y_sup.copy()
                    for j in bad:
                        choices = np.delete(np.arange(ds.num_classes), y_corrupt[j])
                        y_corrupt[j] = rng.choice(choices)
                else:
                    raise ValueError(mode)
                r, _ = run_configs(ds, enc, shots=args.shots, retrain=cfg, seed=seed,
                                   configs=("fewshot",), support=support,
                                   support_x=x_corrupt, support_y=y_corrupt,
                                   extra={"contam_frac": frac, "contam_mode": mode}, H=H)
                rows += r
        print(f"[contamination seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp2_contamination", args.tag)
    fs = df[df.config == "fewshot"]
    print(_summary(fs.dropna(subset=["contam_frac"]), ["contam_mode", "contam_frac"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for mode, sub in fs.groupby("contam_mode"):
        agg = sub.groupby("contam_frac")["acc"].agg(["mean", "std", "count"]).reset_index()
        err = np.where(agg["count"] > 1,
                       1.96 * agg["std"].fillna(0.0) / np.sqrt(agg["count"].clip(lower=1)), 0.0)
        ax.errorbar(agg["contam_frac"], agg["mean"], yerr=err, marker="o", capsize=3,
                    label=f"few-shot ({mode} corruption)")
    ax.axhline(1 / args.classes, color="gray", ls=":", lw=1, label="chance")
    for cfg, color, ls in (("full", "#d65f5f", "-"), ("buffer", "#6acc65", "--")):
        v = df[df.config == cfg].acc
        if len(v):
            ax.axhline(v.mean(), color=color, ls=ls, label=f"{cfg} (clean train pool)")
    ax.set_xlabel("fraction of corrupted support samples")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("support-set contamination (few-shot)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / f"fig_exp2_contamination{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def subtest_noise_dims(args):
    rows = []
    for n_noise in args.noise_dims:
        for seed in range(args.seeds):
            ds = make_gaussian(args.classes, args.dim, args.n_train, args.n_test,
                               snr=1.0, seed=seed, n_noise_dims=n_noise,
                               name="noise_dims")
            enc = make_encoder(args.encoder, args.hd_dim, seed=seed,
                               **({"levels": args.levels} if args.encoder == "idlevel" else {}))
            H = encode_dataset(ds, enc)
            rows += _basic_rows(ds, enc, _retrain(args, seed), args.shots, seed,
                                {"n_noise_dims": n_noise}, H, selections=("hard_random",))
            print(f"[noise_dims {n_noise} seed={seed}] done", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "exp2_noise_dims", args.tag)
    print(_summary(df, ["n_noise_dims"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    _line_plot(axes[0], df, "n_noise_dims", "acc",
               xlabel="number of irrelevant features", chance=1 / args.classes,
               title="irrelevant feature dilution")
    _line_plot(axes[1], df, "n_noise_dims", "min_class_recall",
               xlabel="number of irrelevant features", title="rarest-class recall")
    fig.tight_layout()
    out = RESULTS / f"fig_exp2_noise_dims{args.tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


SUBTESTS = {
    "overlap": subtest_overlap,
    "label_noise": subtest_label_noise,
    "capacity": subtest_capacity,
    "imbalance": subtest_imbalance,
    "contamination": subtest_contamination,
    "noise_dims": subtest_noise_dims,
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
