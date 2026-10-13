"""L1-norm regression (least absolute deviations) by linear programming."""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def l1_regression(W, x, rtol=1e-7):
    """Solve ``min_g ||W @ g - x||_1``.

    The problem is solved through its LP dual,

        max_y  x @ y   s.t.  W.T @ y = 0,  -1 <= y <= 1,

    whose equality multipliers give the regression coefficients. The dual has
    only ``W.shape[1]`` equality constraints, so it is much faster to solve
    than the primal when ``W`` is tall, which is always the case here. If the
    recovered coefficients do not attain the dual optimum (which can only
    happen through solver tolerances), the primal LP is solved instead.

    Parameters
    ----------
    W : ndarray of shape (P, p)
    x : ndarray of shape (P,)

    Returns
    -------
    g : ndarray of shape (p,)
    """
    P, p = W.shape
    res = linprog(-x, A_eq=W.T, b_eq=np.zeros(p), bounds=(-1, 1), method="highs")
    if res.success:
        g = -res.eqlin.marginals
        optimum = -res.fun
        if np.abs(W @ g - x).sum() <= optimum + rtol * max(1.0, abs(optimum)):
            return g
    return _l1_regression_primal(W, x)


def _l1_regression_primal(W, x):
    # min 1'(z+ + z-)  s.t.  W g - z+ + z- = x,  z+, z- >= 0,  g free
    P, p = W.shape
    eye = sparse.identity(P, format="csr")
    A_eq = sparse.hstack([sparse.csr_matrix(W), -eye, eye], format="csr")
    c = np.concatenate([np.zeros(p), np.ones(2 * P)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * P)
    res = linprog(c, A_eq=A_eq, b_eq=x, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("L1 regression LP failed: " + res.message)
    return res.x[:p]
