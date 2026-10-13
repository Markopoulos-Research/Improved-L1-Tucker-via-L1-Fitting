"""Container for a Tucker decomposition."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .tensor import multi_mode_product


@dataclass
class TuckerResult:
    """Tucker decomposition ``X ~ core x_1 U_1 ... x_N U_N``.

    Attributes
    ----------
    factors : list of ndarray
        Orthonormal bases ``U_n`` of shape ``(D_n, d_n)``.
    core : ndarray
        Core tensor ``X x_1 U_1^T ... x_N U_N^T`` of shape ``(d_1, ..., d_N)``.
    history : dict
        Per-iteration diagnostics recorded by the algorithm (see each
        algorithm's docstring).
    first_sweep : TuckerResult, optional
        For :func:`~l1tucker.l1hooi`, the state at the end of the first sweep
        over the modes, so that the T=1 variant comes for free with a longer run.
    """

    factors: List[np.ndarray]
    core: np.ndarray
    history: dict = field(default_factory=dict)
    first_sweep: Optional["TuckerResult"] = None

    def reconstruct(self):
        """Low-rank reconstruction ``core x_1 U_1 ... x_N U_N``."""
        return multi_mode_product(self.core, self.factors)

    @property
    def l1_metric(self):
        """L1-Tucker objective: sum of absolute values of the core."""
        return float(np.abs(self.core).sum())
