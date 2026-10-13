"""Outlier-resistant L1-norm Tucker decomposition.

Python implementation of L1-HOSVD, L1-HOOI and L1T-L1F ("Improved L1-Tucker
via L1-Fitting", Mozaffari, Markopoulos and Prater-Bennette, EUSIPCO 2021),
with HOSVD and HOOI baselines.
"""

from .l1 import l1hooi, l1hooi_random, l1hosvd, l1pca, l1t_l1f
from .l1fit import l1_regression
from .l2 import hooi, hosvd
from .metrics import reconstruction_error, subspace_error
from .result import TuckerResult
from .tensor import fold, kron_except, mode_product, multi_mode_product, unfold

__version__ = "1.0.0"

__all__ = [
    "TuckerResult",
    "fold",
    "hooi",
    "hosvd",
    "kron_except",
    "l1_regression",
    "l1hooi",
    "l1hooi_random",
    "l1hosvd",
    "l1pca",
    "l1t_l1f",
    "mode_product",
    "multi_mode_product",
    "reconstruction_error",
    "subspace_error",
    "unfold",
]
