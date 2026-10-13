"""Error metrics used in the paper."""

import numpy as np


def subspace_error(true_factors, est_factors):
    """Mean Aggregated Normalized Subspace Squared Error (MANSSE), Eq. (13).

    ``(1/N) sum_n ||U_n U_n^T - V_n V_n^T||_F^2 / (2 d_n)``, which is 0 for
    identical subspaces and 1 for orthogonal ones.
    """
    if len(true_factors) != len(est_factors):
        raise ValueError("true_factors and est_factors must have the same length")
    errs = []
    for U, V in zip(true_factors, est_factors):
        if U.shape != V.shape:
            raise ValueError(f"factor shapes differ: {U.shape} vs {V.shape}")
        diff = U @ U.T - V @ V.T
        errs.append(np.sum(diff ** 2) / (2 * U.shape[1]))
    return float(np.mean(errs))


def reconstruction_error(X, X_hat):
    """Normalized squared reconstruction error ``||X - X_hat||_F^2 / ||X||_F^2``."""
    return float(np.sum((X - X_hat) ** 2) / np.sum(X ** 2))
