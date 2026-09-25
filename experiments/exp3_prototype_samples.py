"""Exp 3: which prototype properties need more samples?

Question 3: how many labelled samples does an HDC class prototype need to
become (a) a good estimate of the infinite-data prototype and (b) a good
classifier, and how does that sample complexity scale with class overlap,
feature count and class count?

Measurements per ``(snr, shots, seed)``:

* ``cos_oracle``   : cosine between the few-shot prototype and the "oracle"
  prototype computed from a large clean sample of the same class;
* ``cos_centroid`` : cosine to the encoding of the noiseless class centroid
  (an encoder-side ideal that ignores within-class noise);
* ``margin``       : mean test-set margin ``cos(h,c_true) - max_{j!=true} cos``;
* ``acc``          : test accuracy, with the oracle-prototype accuracy and the
  full-data-prototype accuracy as plateaus.

The script also sweeps ``dim`` and ``num_classes`` to estimate the number of
shots needed to reach a target accuracy (log-interpolated).

Outputs: ``results/exp3_prototypes.csv``, ``results/exp3_sweeps.csv``,
``results/exp3_shots_to_target.csv`` and three figures.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, encode_dataset
from modules import HDCModel, make_encoder, make_gaussian, sample_support
from modules.hd import class_means, row_normalize
from modules.metrics import per_class_recall, prototype_margins, row_cosine

SNR_COLORS = {0.25: "#4878cf", 0.5: "#d65f5f", 1.0: "#6acc65", 2.0: "#b47cc7",
              4.0: "#e8a33d"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--snrs", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--hd-dim", type=int, default=2048)
    ap.add_argument("--levels", type=int, default=64)
    ap.add_argument("--encoder", default="rp")
    ap.add_argument("--oracle-per-class", type=int, default=2000)
    ap.add_argument("--targets", nargs="+", type=float, default=[0.5, 0.8, 0.9])
    ap.add_argument("--sweep", action="store_true",
                    help="run the dim / class-count sample-complexity sweeps")
    ap.add_argument("--sweep-snrs", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--sweep-dims", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--sweep-classes", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--sweep-shots", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--sweep-seeds", type=int, default=2)
    ap.add_argument("--tag", type=str, default="")
    return ap


# --------------------------------------------------------------------------- #
# one prototype curve
# --------------------------------------------------------------------------- #
def prototype_curve(ds, enc, H, shots_list, oracle_per_class: int,
                    seed: int, extra: dict | None = None) -> list[dict]:
    """Few-shot prototype quality vs support size (one dataset/encoder)."""
    H_train, H_test = H
    C, q = ds.num_classes, enc.hd_dim

    # oracle prototypes: class means over a large, clean sample drawn from the
    # same class centroids (not a new random task!)
    oracle_ds = make_gaussian(C, ds.meta["dim_signal"], n_train=oracle_per_class,
                              n_test=1, snr=ds.meta["snr"], seed=seed + 10_000,
                              separation=ds.meta["separation"],
                              n_noise_dims=ds.meta.get("n_noise_dims", 0),
                              centroids=ds.meta["centroids"])
    P_oracle = class_means(enc.encode(oracle_ds.x_train), oracle_ds.y_train, C)
    # encoder-side ideal: encoding of the noiseless class centroid
    P_centroid = row_normalize(enc.encode(ds.meta["centroids"]))
    # full-data prototype (all available labelled training samples)
    P_full = class_means(H_train, ds.y_train, C)

    oracle_acc = _prototype_acc(P_oracle, H_test, ds.y_test)
    full_acc = _prototype_acc(P_full, H_test, ds.y_test)

    rows = []
    for shots in shots_list:
        support = sample_support(ds.y_train, shots, np.random.default_rng(seed + 31))
        model = HDCModel(C, q)
        model.accumulate(H_train[support], ds.y_train[support])
        P = model.classify
        res = model.evaluate(H_test, ds.y_test)
        margins = prototype_margins(H_test, ds.y_test, P)
        rec = per_class_recall(res["preds"], ds.y_test, C)
        cos_or = row_cosine(P, P_oracle)
        cos_cent = row_cosine(P, P_centroid)
        row = {
            "seed": seed,
            "num_classes": C,
            "dim": ds.meta["dim_signal"],
            "n_train": ds.n_train,
            "hd_dim": q,
            "encoder": enc.name,
            "shots": shots,
            "n_support": int(len(support)),
            "acc": res["acc"],
            "acc_oracle": oracle_acc,
            "acc_full_proto": full_acc,
            "macro_recall": float(rec.mean()),
            "cos_oracle_mean": float(cos_or.mean()),
            "cos_oracle_min": float(cos_or.min()),
            "cos_centroid_mean": float(cos_cent.mean()),
            "margin_mean": float(margins.mean()),
            "margin_positive_frac": float((margins > 0).mean()),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def _prototype_acc(P: np.ndarray, H: np.ndarray, y: np.ndarray) -> float:
    return float(((row_normalize(H) @ P.T).argmax(1) == y).mean())


# --------------------------------------------------------------------------- #
# shots-to-target
# --------------------------------------------------------------------------- #
def shots_to_target(df: pd.DataFrame, group_cols, target: float,
                    shots_col: str = "shots") -> pd.DataFrame:
    """Minimal ``shots`` whose mean accuracy reaches ``target`` (log interpolated)."""
    out = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        agg = sub.groupby(shots_col)["acc"].mean().sort_index()
        shots, accs = agg.index.to_numpy(dtype=float), agg.to_numpy(dtype=float)
        value = np.nan
        if np.nanmax(accs) >= target:
            i = int(np.argmax(accs >= target))
            if i == 0:
                value = shots[0]
            else:
                x0, x1 = np.log2(shots[i - 1]), np.log2(shots[i])
                y0, y1 = accs[i - 1], accs[i]
                x = x0 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0)
                value = float(2 ** x)
        out.append({**dict(zip(group_cols, keys)), "target": target,
                    "shots_to_target": value})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _agg_line(ax, df, x, y, group, label_fmt="{}", color_map=None, marker="o",
              linestyle="-", ylabel=None, xlabel=None, logx=False, label=True):
    for key, sub in df.groupby(group):
        agg = sub.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
        err = np.where(agg["count"] > 1,
                       1.96 * agg["std"].fillna(0.0) / np.sqrt(agg["count"]), 0.0)
        color = color_map.get(key) if color_map else None
        ax.errorbar(agg[x], agg["mean"], yerr=err, marker=marker, capsize=3,
                    color=color, linestyle=linestyle,
                    label=label_fmt.format(key) if label else None)
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(alpha=0.3)


def plot_curves(df: pd.DataFrame, tag: str, encoder: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    _agg_line(axes[0], df, "shots", "acc", "snr", label_fmt="SNR {}",
              color_map=SNR_COLORS, ylabel="test accuracy", xlabel="shots per class",
              logx=True)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("few-shot accuracy")
    for snr, sub in df.groupby("snr"):
        axes[0].axhline(sub.acc_oracle.mean(), color=SNR_COLORS.get(snr),
                        ls=":", lw=1, alpha=0.8)
    axes[0].legend(fontsize=8, title="dotted = oracle proto", title_fontsize=7)
    _agg_line(axes[1], df, "shots", "cos_oracle_mean", "snr", label_fmt="SNR {}",
              color_map=SNR_COLORS, ylabel="cos(prototype, oracle)",
              xlabel="shots per class", logx=True)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("prototype fidelity")
    _agg_line(axes[2], df, "shots", "margin_mean", "snr", label_fmt="SNR {}",
              color_map=SNR_COLORS, ylabel="mean test margin",
              xlabel="shots per class", logx=True)
    axes[2].set_title("decision margin")
    fig.suptitle(f"Prototype quality vs support size ({encoder} encoder)", y=1.02)
    fig.tight_layout()
    out = RESULTS / f"fig_exp3_curves_{encoder}{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_convergence(df: pd.DataFrame, tag: str, encoder: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    d = df.copy()
    d["gap"] = np.maximum(1.0 - d.cos_oracle_mean, 1e-6)
    for key, sub in d.groupby("snr"):
        agg = sub.groupby("shots")["gap"].mean()
        axes[0].plot(agg.index, agg.values, marker="o", color=SNR_COLORS.get(key),
                     label=f"SNR {key}")
    ref_x = np.array(sorted(d.shots.unique()), dtype=float)
    if len(ref_x) > 1:
        axes[0].plot(ref_x, 0.1 * ref_x[0] / ref_x, "k--", lw=1, label="~1/n reference")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("shots per class")
    axes[0].set_ylabel("1 - cos(prototype, oracle)")
    axes[0].set_title("prototype estimation error")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    for key, sub in d.groupby("snr"):
        agg = sub.groupby("shots")[["cos_centroid_mean", "cos_oracle_mean"]].mean()
        axes[1].plot(agg.index, agg.cos_oracle_mean - agg.cos_centroid_mean, marker="o",
                     color=SNR_COLORS.get(key), label=f"SNR {key}")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("shots per class")
    axes[1].set_ylabel("cos(oracle) - cos(centroid)")
    axes[1].set_title("within-class noise vs encoder-ideal gap")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / f"fig_exp3_convergence_{encoder}{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_shots_to_target(stt: pd.DataFrame, tag: str) -> None:
    if stt.empty:
        return
    kinds = stt.sweep_kind.unique()
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.6 * len(kinds), 4.4), squeeze=False)
    for ax, kind in zip(axes.ravel(), kinds):
        sub = stt[stt.sweep_kind == kind]
        for target, s in sub.groupby("target"):
            agg = s.groupby("sweep_value")["shots_to_target"].mean()
            ax.plot(agg.index, agg.values, marker="o", label=f"acc >= {target:g}")
        ax.set_xlabel({"snr": "SNR", "dim": "feature count", "classes": "num classes"}[kind])
        ax.set_ylabel("shots per class")
        ax.set_title(f"shots to target ({kind})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / f"fig_exp3_shots_to_target{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _encoder(args, seed):
    return make_encoder(args.encoder, args.hd_dim, seed=seed,
                        **({"levels": args.levels} if args.encoder == "idlevel" else {}))


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    rows: list[dict] = []
    sweep_rows: list[dict] = []

    for snr in args.snrs:
        for seed in range(args.seeds):
            ds = make_gaussian(args.classes, args.dim, args.n_train, args.n_test,
                               snr=snr, seed=seed, name="prototype")
            enc = _encoder(args, seed)
            H = encode_dataset(ds, enc)
            rows += prototype_curve(ds, enc, H, args.shots, args.oracle_per_class,
                                    seed, extra={"snr": snr, "sweep_kind": "main",
                                                 "sweep_value": snr})
            print(f"[main snr={snr:g} seed={seed}] done", flush=True)

    df = pd.DataFrame(rows)
    path = RESULTS / f"exp3_prototypes{args.tag}.csv"
    df.to_csv(path, index=False)
    print(f"saved {path}")

    if args.sweep:
        for kind in ("snr", "dim", "classes"):
            values = {"snr": args.sweep_snrs, "dim": args.sweep_dims,
                      "classes": args.sweep_classes}[kind]
            for value in values:
                for seed in range(args.sweep_seeds):
                    C = args.classes if kind != "classes" else int(value)
                    dim = args.dim if kind != "dim" else int(value)
                    snr = float(value) if kind == "snr" else args.sweep_snrs[-1]
                    ds = make_gaussian(C, dim, args.n_train, args.n_test, snr=snr,
                                       seed=seed, name="prototype_sweep")
                    enc = _encoder(args, seed)
                    H = encode_dataset(ds, enc)
                    sweep_rows += prototype_curve(
                        ds, enc, H, args.sweep_shots, args.oracle_per_class, seed,
                        extra={"snr": snr, "sweep_kind": kind, "sweep_value": value})
                print(f"[sweep {kind}={value}] done", flush=True)
        sdf = pd.DataFrame(sweep_rows)
        spath = RESULTS / f"exp3_sweeps{args.tag}.csv"
        sdf.to_csv(spath, index=False)
        print(f"saved {spath}")

    # ---- shots-to-target tables -------------------------------------------
    stt_parts = []
    for target in args.targets:
        main = shots_to_target(df, ["snr"], target)
        main["sweep_kind"] = "snr"
        main["sweep_value"] = main["snr"]
        stt_parts.append(main)
        if sweep_rows:
            sdf = pd.DataFrame(sweep_rows)
            for kind in ("dim", "classes"):
                sub = sdf[sdf.sweep_kind == kind]
                part = shots_to_target(sub, ["sweep_kind", "sweep_value"], target)
                part["sweep_kind"] = kind
                stt_parts.append(part)
    stt = pd.concat(stt_parts, ignore_index=True)
    tpath = RESULTS / f"exp3_shots_to_target{args.tag}.csv"
    stt.to_csv(tpath, index=False)
    print(f"saved {tpath}")
    print(stt.round(2).to_string(index=False))

    plot_curves(df, args.tag, args.encoder)
    plot_convergence(df, args.tag, args.encoder)
    plot_shots_to_target(stt, args.tag)

    plateaus = df.groupby("snr")[["acc_oracle", "acc_full_proto", "acc"]].mean()
    print("\naccuracy plateaus (mean over seeds):")
    print(plateaus.round(3).to_string())


if __name__ == "__main__":
    main()
