"""Basic tensor operations.

All matricizations follow the Kolda & Bader convention used by the MATLAB
Tensor Toolbox (``tenmat``): the mode-n unfolding of a tensor of shape
``(D_1, ..., D_N)`` is a ``D_n x prod(D_m, m != n)`` matrix whose columns are
ordered with the remaining indices in increasing mode order, the first one
varying fastest. With this convention

    unfold(G x_1 U_1 ... x_N U_N, n) = U_n @ unfold(G, n) @ kron_except(U, n).T
"""

import numpy as np


def unfold(X, mode):
    """Mode-``mode`` matricization of ``X`` (same column order as MATLAB ``tenmat``)."""
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1), order="F")


def fold(M, mode, shape):
    """Inverse of :func:`unfold`: rebuild a tensor of ``shape`` from its mode-``mode`` unfolding."""
    full_shape = [shape[mode]] + [s for n, s in enumerate(shape) if n != mode]
    return np.moveaxis(np.reshape(M, full_shape, order="F"), 0, mode)


def mode_product(X, M, mode):
    """Mode-``mode`` product ``X x_mode M``, i.e. ``unfold(Y, mode) = M @ unfold(X, mode)``."""
    return np.moveaxis(np.tensordot(M, X, axes=(1, mode)), 0, mode)


def multi_mode_product(X, matrices, transpose=False, skip=None):
    """Apply ``X x_1 M_1 x_2 M_2 ... x_N M_N``.

    Parameters
    ----------
    X : ndarray
        Input tensor.
    matrices : sequence of ndarray
        One matrix per mode of ``X``.
    transpose : bool
        If True, multiply by ``M_n.T`` instead of ``M_n`` (e.g. to compute the
        Tucker core ``X x_n U_n.T``).
    skip : int, optional
        A mode to leave untouched.
    """
    for n, M in enumerate(matrices):
        if n == skip:
            continue
        X = mode_product(X, M.T if transpose else M, n)
    return X


def kron_except(matrices, mode):
    """``M_N kron ... kron M_{mode+1} kron M_{mode-1} kron ... kron M_1``.

    The Kronecker order matches the column order of :func:`unfold`.
    """
    W = np.ones((1, 1))
    for n in reversed(range(len(matrices))):
        if n != mode:
            W = np.kron(W, matrices[n])
    return W
