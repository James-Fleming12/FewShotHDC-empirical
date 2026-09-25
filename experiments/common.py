"""Shared helpers for the FewShotHDC experiments."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import (HDCModel, RetrainConfig, fit_buffered_retrain, fit_full_retrain,
                     make_encoder, sample_support)
from modules.metrics import macro_recall, per_class_recall

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

__all__ = [
    "RESULTS", "ROOT", "encode_dataset", "run_configs", "evaluate_model",
    "save_csv", "mean_std", "fmt_ci", "CI", "CONFIG_COLORS", "basic_rows", "line_plot",
]

# --------------------------------------------------------------------------- #
# running configurations
# --------------------------------------------------------------------------- #
def encode_dataset(ds, encoder):
    """Encode train/test splits once (idlevel fits its bounds on x_train)."""
    encoder.fit(ds.x_train)
    return encoder.encode(ds.x_train), encoder.encode(ds.x_test)


def evaluate_model(model: HDCModel, H, y, num_classes: int) -> dict:
    res = model.evaluate(H, y)
    rec = per_class_recall(res["preds"], y, num_classes)
    return {
        "acc": res["acc"],
        "macro_recall": float(rec.mean()),
        "min_class_recall": float(rec.min()),
        "per_class_recall": rec.tolist(),
    }


def _base_row(ds, encoder, shots, seed, extra):
    row = {
        "dataset": ds.name,
        "num_classes": int(ds.num_classes),
        "dim": ds.dim,
        "n_train": ds.n_train,
        "n_test": ds.n_test,
        "snr": float(ds.meta.get("snr", np.nan)),
        "label_noise": float(ds.meta.get("label_noise", 0.0)),
        "hd_dim": int(encoder.hd_dim),
        "encoder": encoder.name,
        "shots": shots,
        "seed": seed,
    }
    if extra:
        row.update(extra)
    return row


def run_configs(ds, encoder, *, shots, retrain: RetrainConfig, seed: int = 0,
                configs=("fewshot", "full", "buffer"), selection: str = "hard_random",
                support: np.ndarray | None = None, support_x: np.ndarray | None = None,
                support_y: np.ndarray | None = None,
                support_weights: np.ndarray | None = None, extra: dict | None = None,
                H: tuple[np.ndarray, np.ndarray] | None = None):
    """Run the requested configurations on one dataset.

    Returns ``(rows, histories)``; histories only populated for full/buffer.
    ``H`` allows reusing pre-encoded splits (must come from ``encode_dataset``
    with the same encoder instance).
    """
    H_train, H_test = H if H is not None else encode_dataset(ds, encoder)
    if support is None:
        support = sample_support(ds.y_train, shots, np.random.default_rng(seed))
    support = np.asarray(support, dtype=np.int64)
    if support_x is None:
        H_support = H_train[support]
    else:
        H_support = encoder.encode(np.asarray(support_x, dtype=np.float32))
    y_support = ds.y_train[support] if support_y is None else np.asarray(support_y)

    rows, histories = [], {}
    full_budget = ds.n_train * (1 + retrain.epochs)

    for cfg_name in configs:
        model = HDCModel(ds.num_classes, encoder.hd_dim)
        history = None
        if cfg_name == "fewshot":
            model.accumulate(H_support, y_support, weights=support_weights)
            n_encoded = int(len(support))
            init_acc = np.nan
        elif cfg_name == "full_init":
            # reference: plain full-data class means, before any retraining
            model.accumulate(H_train, ds.y_train)
            n_encoded = int(ds.n_train)
            init_acc = np.nan
        elif cfg_name in ("full", "buffer"):
            if cfg_name == "full":
                history = fit_full_retrain(model, H_train, ds.y_train, retrain, H_test, ds.y_test)
            else:
                history = fit_buffered_retrain(
                    model, H_train, ds.y_train,
                    replace(retrain, selection=selection, seed=seed), H_test, ds.y_test)
            n_encoded = int(history[-1]["n_encoded_cum"])
            init_acc = float(history[0]["test_acc"])
            histories[cfg_name if selection == "hard_random" else f"{cfg_name}_{selection}"] = history
        else:
            raise ValueError(cfg_name)

        metrics = evaluate_model(model, H_test, ds.y_test, ds.num_classes)
        row = _base_row(ds, encoder, shots, seed, extra)
        row.update({
            "config": cfg_name if cfg_name != "buffer" else (
                "buffer" if selection == "hard_random" else f"buffer_{selection}"),
            "selection": selection if cfg_name == "buffer" else "",
            "n_support": int(len(support)),
            "n_encoded": n_encoded,
            "n_encoded_rel": n_encoded / max(1, full_budget),
            "test_acc_init": init_acc,
            "buffer_fraction": retrain.buffer_fraction if cfg_name == "buffer" else 0.0,
            "epochs": 0 if cfg_name in ("fewshot", "full_init") else retrain.epochs,
            **metrics,
        })
        rows.append(row)
    return rows, histories


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def save_csv(rows, name: str) -> Path:
    df = pd.DataFrame(rows)
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def mean_std(df: pd.DataFrame, group_cols, col: str) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={"mean": f"{col}_mean", "std": f"{col}_std",
                               "count": f"{col}_n"})


def fmt_ci(series, digits: int = 3) -> str:
    arr = np.asarray(series, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "n/a"
    if arr.size == 1:
        return f"{arr[0]:.{digits}f}"
    return f"{arr.mean():.{digits}f}±{arr.std(ddof=1):.{digits}f}"


def CI(series) -> float:
    """Half-width of a 95% CI over independent seeds."""
    arr = np.asarray(series, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    return 1.96 * arr.std(ddof=1) / np.sqrt(arr.size)


# --------------------------------------------------------------------------- #
# shared run/plot recipes for the stress-test experiments
# --------------------------------------------------------------------------- #
CONFIG_COLORS = {
    "fewshot": "#4878cf", "full": "#d65f5f", "buffer": "#6acc65",
    "buffer_random": "#b47cc7", "full_init": "#8c8c8c",
    "fewshot_balanced": "#4878cf", "fewshot_natural": "#9ab8e0",
    "source": "#4878cf", "shifted": "#6acc65",
}


def basic_rows(ds, enc, cfg, shots, seed, extra, H, selections=("hard_random",)):
    """few-shot + full-init + full + buffered (+ extra buffer controls) rows."""
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


def line_plot(ax, df, x, y, hue="config", ylim=(0, 1.02), logx=False,
              ylabel="test accuracy", xlabel=None, chance=None, title=None):
    """Mean line + 95% CI error bars per ``hue`` group over seeds."""
    for key, sub in df.groupby(hue):
        agg = sub.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
        err = np.where(agg["count"] > 1,
                       1.96 * agg["std"].fillna(0.0) / np.sqrt(agg["count"].clip(lower=1)), 0.0)
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
