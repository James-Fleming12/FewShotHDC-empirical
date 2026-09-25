"""Exp 1: how far does few-shot HDC get?

Question 1: what few-shot ability does the HDC prototype have, and how does it
compare with the full retraining pipelines?

Protocol
--------
For every target SNR and seed:

* build a ``C``-class Gaussian task (``dim`` features, ``n_train`` samples per
  class) and encode it once with the HDC encoder;
* config A (``fewshot``): prototypes = (weighted) average of ``shots`` labelled
  support samples per class, no retraining;
* config B (``full``): prototypes = full-data class means, then ``epochs``
  error-driven retraining passes over the full training pool;
* config C (``buffer``): same initial full-data pass, then ``epochs`` passes
  over a 5% hard+random buffer (Sec. 4.3);
* evaluate every model on the held-out test split and record the number of
  encoded samples (the deployment encode budget).

Outputs
-------
``results/exp1_fewshot.csv``, ``results/fig_exp1_shots_*.png`` (accuracy vs
shots), ``results/fig_exp1_cost*.png`` (accuracy vs encode budget),
``results/fig_exp1_trace*.png`` + ``results/exp1_buffer_trace.csv`` (one
representative buffer trace).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from common import RESULTS, run_configs
from common import encode_dataset, save_csv
from modules import RetrainConfig, make_encoder, make_gaussian


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--snrs", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    ap.add_argument("--encoders", nargs="+", default=["rp"],
                    help="rp | idlevel (multiple values produce a comparison)")
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--levels", type=int, default=64, help="idlevel levels")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-fraction", type=float, default=0.05)
    ap.add_argument("--tag", type=str, default="")
    return ap


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _err(ax, x, agg, **kw):
    yerr = np.where(agg["count"] > 1,
                    1.96 * agg["std"].fillna(0.0) / np.sqrt(agg["count"].clip(lower=1)), 0.0)
    ax.errorbar(x, agg["mean"], yerr=yerr, capsize=3, **kw)


def plot_shots(df: pd.DataFrame, encoders: list[str], snrs: list[float],
               num_classes: int, tag: str) -> None:
    for enc in encoders:
        sub_enc = df[df.encoder == enc]
        fig, axes = plt.subplots(1, len(snrs), figsize=(5.2 * len(snrs), 4.2),
                                 squeeze=False)
        for ax, snr in zip(axes.ravel(), snrs):
            sub = sub_enc[sub_enc.snr == snr]
            fs = sub[sub.config == "fewshot"]
            agg = fs.groupby("shots")["acc"].agg(["mean", "std", "count"])
            _err(ax, agg.index.values, agg, color="#4878cf", marker="o",
                 label="few-shot (A)")
            for cfg, color, ls in (("full", "#d65f5f", "-"),
                                   ("buffer", "#6acc65", "--")):
                vals = sub[sub.config == cfg]["acc"]
                if len(vals):
                    ax.axhline(vals.mean(), color=color, ls=ls, label=f"{cfg} (B/C)")
                    if len(vals) > 1:
                        ax.fill_between(agg.index.values, vals.mean() - vals.std(ddof=1),
                                        vals.mean() + vals.std(ddof=1), color=color, alpha=0.15)
            ax.axhline(1.0 / num_classes, color="gray", ls=":", lw=1, label="chance")
            ax.set_xscale("log", base=2)
            ax.set_xticks(fs.shots.unique())
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax.set_xlabel("labelled shots per class")
            ax.set_ylabel("test accuracy")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"SNR = {snr:g}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Few-shot ability vs full retraining ({enc} encoder)", y=1.02)
        fig.tight_layout()
        out = RESULTS / f"fig_exp1_shots_{enc}{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


def plot_cost(df: pd.DataFrame, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for enc in df.encoder.unique():
        sub = df[df.encoder == enc]
        for cfg, marker, color in (("fewshot", "o", "#4878cf"), ("full", "s", "#d65f5f"),
                                   ("buffer", "^", "#6acc65")):
            s = sub[sub.config == cfg]
            if not len(s):
                continue
            agg = s.groupby(["snr", "shots", "config"], dropna=False).agg(
                n_encoded=("n_encoded", "mean"), acc=("acc", "mean")).reset_index()
            ax.scatter(agg.n_encoded, agg.acc, marker=marker, s=32, alpha=0.75,
                       label=f"{cfg} ({enc})", color=color)
    ax.set_xscale("log")
    ax.set_xlabel("samples encoded (deployment budget, log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Accuracy vs encode budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / f"fig_exp1_cost{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_trace(trace_rows: list[dict], tag: str) -> None:
    df = pd.DataFrame(trace_rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    axes[0].plot(df.epoch, df.train_acc, marker="o", label="train (buffer)")
    axes[0].plot(df.epoch, df.test_acc, marker="s", label="test")
    axes[0].set_xlabel("retraining epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("buffer retraining accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(df.epoch, df.buffer_mean_loss_before, marker="o", color="#d65f5f")
    axes[1].set_xlabel("retraining epoch")
    axes[1].set_ylabel("mean stored loss of buffer")
    axes[1].set_title("selected buffer hardness")
    axes[1].grid(alpha=0.3)
    axes[2].plot(df.epoch, df.n_encoded_cum, marker="o", color="#6acc65")
    axes[2].set_xlabel("retraining epoch")
    axes[2].set_ylabel("cumulative encoded samples")
    axes[2].set_title("encode budget")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / f"fig_exp1_trace{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    out_csv = RESULTS / f"exp1_buffer_trace{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {out} and {out_csv}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    rows: list[dict] = []
    trace_rows: list[dict] = []
    trace_done = False

    for enc_kind in args.encoders:
        for snr in args.snrs:
            for seed in range(args.seeds):
                ds = make_gaussian(num_classes=args.classes, dim=args.dim,
                                   n_train=args.n_train, n_test=args.n_test,
                                   snr=snr, seed=seed)
                enc = make_encoder(enc_kind, args.hd_dim, seed=seed,
                                   **({"levels": args.levels} if enc_kind == "idlevel" else {}))
                H = encode_dataset(ds, enc)
                cfg = RetrainConfig(epochs=args.epochs, lr=args.lr,
                                    batch_size=args.batch_size,
                                    buffer_fraction=args.buffer_fraction, seed=seed)
                extra = {"snr": snr}
                # B and C do not depend on the support size: run once.
                r, hist = run_configs(ds, enc, shots=args.shots[-1], retrain=cfg,
                                      seed=seed, configs=("full", "buffer"),
                                      extra=extra, H=H)
                rows += r
                by_cfg = {row["config"]: row["acc"] for row in r}
                if not trace_done and seed == 0:
                    trace_rows = [dict(snr=snr, encoder=enc_kind, **h)
                                  for h in hist["buffer"]]
                    trace_done = True
                # A for every support size.
                for shots in args.shots:
                    r, _ = run_configs(ds, enc, shots=shots, retrain=cfg,
                                       seed=seed, configs=("fewshot",),
                                       extra=extra, H=H)
                    rows += r
                print(f"[{enc_kind} snr={snr:g} seed={seed}] "
                      f"full={by_cfg['full']:.3f} buffer={by_cfg['buffer']:.3f}",
                      flush=True)

    df = pd.DataFrame(rows)
    path = save_csv(rows, f"exp1_fewshot{args.tag}")
    print(f"\nsaved {path}")

    piv = df.pivot_table(index=["encoder", "snr", "shots", "config"],
                         values="acc", aggfunc="mean")
    print("\nMean test accuracy:")
    print(piv.round(3).to_string())

    plot_shots(df, args.encoders, args.snrs, args.classes, args.tag)
    plot_cost(df, args.tag)
    plot_trace(trace_rows, args.tag)

    # compact summary: few-shot gap vs full retrain at the extreme support sizes
    for enc_kind in args.encoders:
        for snr in args.snrs:
            sub = df[(df.encoder == enc_kind) & (df.snr == snr)]
            full = sub[sub.config == "full"].acc.mean()
            buf = sub[sub.config == "buffer"].acc.mean()
            vals = {s: sub[(sub.config == "fewshot") & (sub.shots == s)].acc.mean()
                    for s in args.shots}
            print(f"[{enc_kind} snr={snr:g}] full={full:.3f} buffer={buf:.3f} "
                  f"| few-shot " + " ".join(f"{s}:{v:.3f}" for s, v in vals.items()))


if __name__ == "__main__":
    main()
