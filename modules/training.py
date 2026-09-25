"""Training configurations for the few-shot HDC study.

Three pipelines share the same target model (:class:`modules.hd.HDCModel`):

``few-shot``   prototypes are the sum of the encoded labelled support samples
               (the "weighted average" with uniform weights); no retraining.
``full``       initial full-data pass sets the class accumulators to the
               dataset means, then ``epochs`` error-driven retraining passes
               over the full training pool.
``buffered``   the same initial full-data pass, then ``epochs`` retraining
               passes over a ``buffer_fraction`` subset composed of the
               hardest samples (highest previous loss) plus random samples;
               per-sample losses of buffer samples are refreshed after each
               pass, all other losses keep their stored value (Sec. 4.3).

Samples are already encoded (``H`` matrices) in every entry point.  Encoding a
dataset once and indexing subsets is semantically identical to re-encoding the
subsets, and makes it easy to compare the *encode budget* (number of samples
that would have to be encoded in deployment) across configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hd import HDCModel, loss_from_scores

__all__ = [
    "RetrainConfig",
    "initial_pass",
    "retrain_epoch",
    "select_buffer",
    "fit_fewshot",
    "fit_full_retrain",
    "fit_buffered_retrain",
]


@dataclass
class RetrainConfig:
    """Shared protocol for the full / buffered retraining pipelines."""

    epochs: int = 10
    lr: float = 1.0
    batch_size: int = 256
    update_on_correct: bool = False
    buffer_fraction: float = 0.05
    hard_fraction: float = 0.5
    selection: str = "hard_random"  # hard_random | loss_first | wrong_first | random
    seed: int = 0


# --------------------------------------------------------------------------- #
# single passes
# --------------------------------------------------------------------------- #
def initial_pass(model: HDCModel, H: np.ndarray, y: np.ndarray,
                 batch_size: int = 256, compute_loss: bool = True) -> dict:
    """First post-deployment pass over the full dataset.

    Each batch is accumulated into ``classify_weights`` and then scored with
    the freshly committed prototypes (encode-once semantics: every sample is
    encoded a single time, exactly like the reference initial pass).
    """
    y = np.asarray(y, dtype=np.int64)
    n = H.shape[0]
    losses = np.zeros(n, dtype=np.float32)
    n_wrong = 0
    for s in range(0, n, batch_size):
        Hb, yb = H[s:s + batch_size], y[s:s + batch_size]
        model.accumulate(Hb, yb)
        S = model.scores(Hb)
        p = S.argmax(axis=1)
        if compute_loss:
            losses[s:s + batch_size] = loss_from_scores(S, yb, p)
        n_wrong += int((p != yb).sum())
    return {"acc": 1.0 - n_wrong / n, "losses": losses, "n_wrong": n_wrong,
            "n": n, "mean_loss": float(losses.mean())}


def retrain_epoch(model: HDCModel, H: np.ndarray, y: np.ndarray,
                  cfg: RetrainConfig, indices: np.ndarray | None = None,
                  compute_loss: bool = True) -> dict:
    """One error-driven retraining pass over ``indices`` (or all samples).

    Prototypes are normalised at every batch start (reference behaviour);
    only misclassified samples move the accumulators by default.
    """
    y = np.asarray(y, dtype=np.int64)
    idx = np.arange(H.shape[0]) if indices is None else np.asarray(indices)
    n = len(idx)
    losses = np.zeros(n, dtype=np.float32)
    preds = np.empty(n, dtype=np.int64)
    n_wrong = 0
    for s in range(0, n, cfg.batch_size):
        sel = idx[s:s + cfg.batch_size]
        Hb, yb = H[sel], y[sel]
        model.commit()  # normalise before scoring (reference Basic_HD.retrain)
        S = model.scores(Hb)
        p = S.argmax(axis=1)
        preds[s:s + cfg.batch_size] = p
        if compute_loss:
            losses[s:s + cfg.batch_size] = loss_from_scores(S, yb, p)
        n_wrong += model.apply_error_updates(Hb, yb, p, cfg.lr, cfg.update_on_correct)
    model.commit()
    return {"acc": 1.0 - n_wrong / n, "losses": losses, "preds": preds,
            "n_wrong": n_wrong, "n": n, "mean_loss": float(losses.mean())}


# --------------------------------------------------------------------------- #
# buffer selection (paper Sec. 4.3 / reference `Model.encode` selection)
# --------------------------------------------------------------------------- #
def select_buffer(losses: np.ndarray, cfg: RetrainConfig,
                  rng: np.random.Generator) -> np.ndarray:
    """Select the next retraining buffer.

    ``hard_random`` (paper): top ``hard_fraction`` of the buffer by stored
    loss + uniformly random samples from the rest.
    ``wrong_first`` (reference code): all samples with non-zero loss first,
    random fill afterwards.
    ``loss_first``: top-loss samples only.
    ``random``: uniform control baseline.
    """
    losses = np.asarray(losses, dtype=np.float64)
    n = len(losses)
    n_buf = int(min(n, max(2, round(cfg.buffer_fraction * n))))

    if cfg.selection == "random":
        return rng.choice(n, size=n_buf, replace=False)
    if cfg.selection == "loss_first":
        noise = 1e-9 * rng.random(n)  # random tie-break
        return np.argsort(-(losses + noise))[:n_buf]

    if cfg.selection == "wrong_first":
        wrong = np.flatnonzero(losses > 0)
        if wrong.size >= n_buf:
            return rng.choice(wrong, size=n_buf, replace=False)
        rest = np.setdiff1d(np.arange(n), wrong, assume_unique=False)
        fill = rng.choice(rest, size=n_buf - wrong.size, replace=False)
        return np.concatenate([wrong, fill])

    if cfg.selection == "hard_random":
        n_hard = int(round(cfg.hard_fraction * n_buf))
        n_hard = min(n_hard, n_buf)
        noise = 1e-9 * rng.random(n)
        order = np.argsort(-(losses + noise))
        hard = order[:n_hard]
        rest = order[n_hard:]
        rand = rng.choice(rest, size=n_buf - n_hard, replace=False)
        return np.concatenate([hard, rand])

    raise ValueError(f"unknown selection {cfg.selection!r}")


# --------------------------------------------------------------------------- #
# configurations
# --------------------------------------------------------------------------- #
def fit_fewshot(model: HDCModel, H_support: np.ndarray, y_support: np.ndarray,
                weights: np.ndarray | None = None) -> dict:
    """Config A: prototypes = (weighted) average of the labelled support."""
    model.accumulate(H_support, np.asarray(y_support, dtype=np.int64), weights=weights)
    return {"n_encoded": int(H_support.shape[0]), "sample_cnt": model.sample_cnt.copy()}


def _row(epoch: int, phase: str, n_epoch: int, n_cum: int, n_total: int,
         stats: dict, test: tuple[np.ndarray, np.ndarray] | None,
         model: HDCModel, extra: dict | None = None) -> dict:
    row = {
        "epoch": epoch,
        "phase": phase,
        "n_encoded_epoch": int(n_epoch),
        "n_encoded_cum": int(n_cum),
        "n_encoded_frac": n_cum / n_total,
        "train_acc": stats["acc"],
        "mean_loss": stats["mean_loss"],
        "n_wrong": stats["n_wrong"],
    }
    if test is not None:
        row["test_acc"] = model.accuracy(test[0], test[1])
    if extra:
        row.update(extra)
    return row


def fit_full_retrain(model: HDCModel, H: np.ndarray, y: np.ndarray,
                     cfg: RetrainConfig, H_test: np.ndarray | None = None,
                     y_test: np.ndarray | None = None) -> list[dict]:
    """Config B: full-data means + ``cfg.epochs`` full-data retraining passes."""
    test = (H_test, y_test) if H_test is not None else None
    n = H.shape[0]
    hist: list[dict] = []
    init = initial_pass(model, H, y, cfg.batch_size, compute_loss=False)
    hist.append(_row(0, "init", n, n, n, init, test, model,
                     {"buffer_size": n, "n_hard": 0, "n_random": 0}))
    for e in range(1, cfg.epochs + 1):
        st = retrain_epoch(model, H, y, cfg, indices=None, compute_loss=True)
        hist.append(_row(e, "full", n, (1 + e) * n, n, st, test, model,
                         {"buffer_size": n, "n_hard": 0, "n_random": 0}))
    return hist


def fit_buffered_retrain(model: HDCModel, H: np.ndarray, y: np.ndarray,
                         cfg: RetrainConfig, H_test: np.ndarray | None = None,
                         y_test: np.ndarray | None = None) -> list[dict]:
    """Config C: full-data initial pass + buffered retraining passes (Sec. 4.3)."""
    test = (H_test, y_test) if H_test is not None else None
    rng = np.random.default_rng(cfg.seed)
    n = H.shape[0]
    hist: list[dict] = []
    init = initial_pass(model, H, y, cfg.batch_size, compute_loss=True)
    losses = init["losses"]
    hist.append(_row(0, "init", n, n, n, init, test, model,
                     {"buffer_size": n, "n_hard": 0, "n_random": 0,
                      "buffer_mean_loss_before": float(init["mean_loss"])}))
    for e in range(1, cfg.epochs + 1):
        buf = select_buffer(losses, cfg, rng)
        if cfg.selection == "hard_random":
            n_hard = min(int(round(cfg.hard_fraction * len(buf))), len(buf))
        elif cfg.selection == "wrong_first":
            n_hard = int((losses[buf] > 0).sum())
        elif cfg.selection == "loss_first":
            n_hard = len(buf)
        else:  # random control: hard/random split is undefined
            n_hard = 0
        mean_before = float(losses[buf].mean())
        st = retrain_epoch(model, H, y, cfg, indices=buf, compute_loss=True)
        losses[buf] = st["losses"]  # refresh only buffer losses; others stay stale
        hist.append(_row(e, "buffer", len(buf), n + e * len(buf), n, st, test, model,
                         {"buffer_size": int(len(buf)),
                          "n_hard": n_hard, "n_random": int(len(buf) - n_hard),
                          "buffer_mean_loss_before": mean_before}))
    return hist
