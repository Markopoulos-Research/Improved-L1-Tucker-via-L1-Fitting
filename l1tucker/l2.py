"""Standard (L2-norm) Tucker baselines: HOSVD and HOOI."""

import numpy as np

from .result import TuckerResult
from .tensor import mode_product, multi_mode_product, unfold


def _leading_left_singular_vectors(M, rank):
    evals, evecs = np.linalg.eigh(M @ M.T)
    return evecs[:, ::-1][:, :rank]


def hosvd(X, ranks, sequential=True):
    """Higher-Order SVD.

    With ``sequential=True`` (default) this is the sequentially truncated
    HOSVD, as computed by ``hosvd`` in the MATLAB Tensor Toolbox: each basis
    is computed from the data already projected on the previous bases.
    """
    X = np.asarray(X, dtype=float)
    factors = []
    Y = X
    for n, r in enumerate(ranks):
        U = _leading_left_singular_vectors(unfold(Y if sequential else X, n), r)
        factors.append(U)
        if sequential:
            Y = mode_product(Y, U.T, n)
    core = Y if sequential else multi_mode_product(X, factors, transpose=True)
    return TuckerResult(factors, core)


def hooi(X, ranks, init=None, tol=1e-8, maxit=100):
    """Higher-Order Orthogonal Iterations (ALS for L2 Tucker).

    Mirrors ``tucker_als`` of the MATLAB Tensor Toolbox: each sweep updates the
    bases in increasing mode order and the iterations stop when the change in
    fit ``1 - ||X - X_hat||_F / ||X||_F`` drops below ``tol``.

    Parameters
    ----------
    init : list of ndarray, optional
        Initial bases. Defaults to the HOSVD solution. As in ``tucker_als``,
        the initial basis of the first mode is not used.

    ``history["fit"]`` holds the fit after each sweep.
    """
    X = np.asarray(X, dtype=float)
    N = X.ndim
    factors = [np.array(U, dtype=float) for U in (hosvd(X, ranks).factors if init is None else init)]
    norm_x = np.linalg.norm(X)
    fit = 0.0
    history = {"fit": []}
    for it in range(maxit):
        fit_old = fit
        for n in range(N):
            Y = multi_mode_product(X, factors, transpose=True, skip=n)
            factors[n] = _leading_left_singular_vectors(unfold(Y, n), ranks[n])
        core = mode_product(Y, factors[N - 1].T, N - 1)
        residual = np.sqrt(max(norm_x ** 2 - np.sum(core ** 2), 0.0))
        fit = 1 - residual / norm_x
        history["fit"].append(fit)
        if it > 0 and abs(fit_old - fit) < tol:
            break
    return TuckerResult(factors, core, history)
