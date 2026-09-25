"""FewShotHDC empirical building blocks.

Modules
-------
encoders : `rp` / `idlevel` encoders matching the HyperLiDAR reference model.
hd       : class-hypervector model (`classify_weights` sum accumulator + cosine).
training : few-shot / full-retrain / buffered-retrain configurations.
data     : synthetic Gaussian tasks with controllable SNR, imbalance, noise.
metrics  : accuracy / recall / prototype-geometry helpers.
"""

from .data import Dataset, make_gaussian, sample_support
from .encoders import IDLevelEncoder, RPEncoder, make_encoder
from .hd import HDCModel
from .training import (RetrainConfig, fit_buffered_retrain, fit_fewshot,
                       fit_full_retrain)

__all__ = [
    "Dataset",
    "make_gaussian",
    "sample_support",
    "RPEncoder",
    "IDLevelEncoder",
    "make_encoder",
    "HDCModel",
    "RetrainConfig",
    "fit_fewshot",
    "fit_full_retrain",
    "fit_buffered_retrain",
]
