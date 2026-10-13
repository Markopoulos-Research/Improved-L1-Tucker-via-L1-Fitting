"""L1-norm Tucker decomposition: L1-PCA, L1-HOSVD, L1-HOOI and L1T-L1F.

L1-Tucker seeks orthonormal bases that maximize the L1 norm of the core,

    max_{U_n^T U_n = I}  || X x_1 U_1^T ... x_N U_N^T ||_1,

which makes the decomposition far less sensitive to outliers than standard
(L2) Tucker.
"""

import numpy as np

from .l1fit import l1_regression
from .l2 import hosvd
from .metrics import reconstruction_error, subspace_error
from .result import TuckerResult
from .tensor import kron_except, multi_mode_product, unfold


def _rel_change(new, old):
    return np.inf if old == 0 else abs(new - old) / abs(old)


def _procrustes(M):
    # argmax_{Q^T Q = I} tr(Q^T M) = U V^T for the thin SVD M = U S V^T
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def _check_ranks(X, ranks):
    if len(ranks) != X.ndim:
        raise ValueError(f"expected {X.ndim} ranks, got {len(ranks)}")
    for n, (D, d) in enumerate(zip(X.shape, ranks)):
        if not 1 <= d <= D:
            raise ValueError(f"rank {d} of mode {n} must be between 1 and {D}")


def _check_factors(X, ranks, factors):
    if len(factors) != X.ndim:
        raise ValueError(f"expected {X.ndim} initial factors, got {len(factors)}")
    for n, U in enumerate(factors):
        if U.shape != (X.shape[n], ranks[n]):
            raise ValueError(f"initial factor {n} has shape {U.shape}, expected {(X.shape[n], ranks[n])}")


class _Tracker:
    """Records the L1 metric (and optional errors) after every basis update."""

    def __init__(self, X, X_clean, U_true):
        self.X, self.X_clean, self.U_true = X, X_clean, U_true
        self.history = {"l1_metric": [], "subspace_error": [], "reconstruction_error": [], "mode": [], "sweep": []}

    def record(self, factors, mode=None, sweep=None):
        core = multi_mode_product(self.X, factors, transpose=True)
        h = self.history
        h["l1_metric"].append(float(np.abs(core).sum()))
        if self.U_true is not None:
            h["subspace_error"].append(subspace_error(self.U_true, factors))
        if self.X_clean is not None:
            h["reconstruction_error"].append(reconstruction_error(self.X_clean, multi_mode_product(core, factors)))
        h["mode"].append(mode)
        h["sweep"].append(sweep)

    def snapshot(self):
        return {k: list(v) for k, v in self.history.items()}


def l1pca(A, init=None, tol=1e-8, maxit=1000, callback=None):
    """Approximate L1-PCA by fixed-point iterations.

    Approximately solves ``max_{Q^T Q = I} ||A^T Q||_1`` for ``Q`` of the same
    shape as ``init``, alternating ``B = sign(A^T Q)`` and
    ``Q = U V^T`` with ``A B = U S V^T``. The metric is non-decreasing over
    the iterations. Iterations stop when its relative change is at most
    ``tol`` or after ``maxit`` iterations.

    Parameters
    ----------
    A : ndarray of shape (D, M)
        Data matrix (columns are samples).
    init : ndarray of shape (D, d) or int
        Initial basis, or the number of components ``d`` to start from the
        ``d`` leading left singular vectors of ``A``.
    callback : callable, optional
        Called as ``callback(Q)`` after every update of ``Q``.

    Returns
    -------
    Q : ndarray of shape (D, d)
    metric : list of float
        ``||A^T Q||_1`` before each update.
    """
    if np.isscalar(init) or init is None:
        d = 1 if init is None else int(init)
        Q = np.linalg.svd(A, full_matrices=False)[0][:, :d]
    else:
        Q = np.array(init, dtype=float)
    prev = 0.0
    metric = []
    for _ in range(maxit):
        AQ = A.T @ Q
        metric.append(float(np.abs(AQ).sum()))
        Q = _procrustes(A @ np.sign(AQ))
        if callback is not None:
            callback(Q)
        if _rel_change(metric[-1], prev) <= tol:
            break
        prev = metric[-1]
    return Q, metric


def l1hosvd(X, ranks, init=None, tol=1e-8, maxit=1000, X_clean=None, U_true=None):
    """L1-HOSVD: each basis is the L1-PCA of the corresponding unfolding of ``X``.

    Parameters
    ----------
    X : ndarray
        Data tensor of shape ``(D_1, ..., D_N)``.
    ranks : sequence of int
        Tucker ranks ``(d_1, ..., d_N)``.
    init : list of ndarray, optional
        Initial bases for L1-PCA. Defaults to the HOSVD solution.
    tol, maxit : float, int
        Stopping rule of each L1-PCA.
    X_clean : ndarray, optional
        Noise-free tensor. If given, ``history["reconstruction_error"]`` is
        recorded after every basis update.
    U_true : list of ndarray, optional
        True bases. If given, ``history["subspace_error"]`` is recorded after
        every basis update.

    Returns
    -------
    TuckerResult
        ``history["l1_metric"]`` holds the L1 norm of the core after every
        basis update (first entry: at ``init``), ``history["mode"]`` the mode
        that was updated.
    """
    X = np.asarray(X, dtype=float)
    _check_ranks(X, ranks)
    factors = [np.array(U, dtype=float) for U in (hosvd(X, ranks).factors if init is None else init)]
    _check_factors(X, ranks, factors)
    tracker = _Tracker(X, X_clean, U_true)
    tracker.record(factors)
    for n in range(X.ndim):
        if ranks[n] == X.shape[n]:
            factors[n] = np.eye(ranks[n])
            continue

        def update(Q, n=n):
            factors[n] = Q
            tracker.record(factors, mode=n)

        factors[n], _ = l1pca(unfold(X, n), factors[n], tol=tol, maxit=maxit, callback=update)
    core = multi_mode_product(X, factors, transpose=True)
    return TuckerResult(factors, core, tracker.history)


def _projected_unfolding(X, factors, n, proj):
    """Matrix whose L1-PCA gives the mode-``n`` basis in one L1-HOOI step.

    ``proj="L2"``: ``[X x_{m != n} U_m^T]_(n)``, the L2 projection used by
    L1-HOOI.

    ``proj="L1"``: ``[G_n x_{m != n} U_m]_(n)``, where the core ``G_n`` is the
    L1-norm fit of ``X`` on the current bases of the other modes, Eqs. (4)-(7)
    of the paper. Each row of ``[G_n]_(n)`` is one L1 regression, Eq. (7).
    """
    if proj == "L2":
        return unfold(multi_mode_product(X, factors, transpose=True, skip=n), n)
    W = kron_except(factors, n)
    G = np.array([l1_regression(W, x) for x in unfold(X, n)])
    return G @ W.T


def _setup(X, ranks, init, proj, T, tol):
    """Shared argument handling of the L1-HOOI variants."""
    X = np.asarray(X, dtype=float)
    _check_ranks(X, ranks)
    if proj not in ("L1", "L2"):
        raise ValueError(f"proj must be 'L1' or 'L2', got {proj!r}")
    if T is None:
        T = 10 if proj == "L1" else np.inf
    if init is None:
        init = l1hosvd(X, ranks, tol=tol).factors
    factors = [np.array(U, dtype=float) for U in init]
    _check_factors(X, ranks, factors)
    return X, factors, T


def l1hooi(X, ranks, init=None, proj="L2", T=None, order=None, tol=1e-8, maxit=1000, rng=None,
           X_clean=None, U_true=None):
    """L1-HOOI, with either L2 projection (L1-HOOI) or L1 fitting (L1T-L1F).

    Starting from ``init``, the bases are updated one mode at a time. The
    mode-``n`` basis is the L1-PCA (started at the current basis) of the
    matrix returned by the projection step:

    * ``proj="L2"``: projection of the data on the bases of the other modes
      (L1-HOOI of Chachlakis et al.).
    * ``proj="L1"``: L1-norm fitting of the data on the bases of the other
      modes, mapped back to the data space (L1T-L1F of Mozaffari et al.,
      EUSIPCO 2021). See also :func:`l1t_l1f`.

    A sweep updates every mode once. Sweeps stop when the relative change of
    the L1 metric between two sweeps is below ``tol``, after ``T`` sweeps, or
    after ``maxit`` sweeps.

    Parameters
    ----------
    X : ndarray
        Data tensor of shape ``(D_1, ..., D_N)``.
    ranks : sequence of int
        Tucker ranks ``(d_1, ..., d_N)``.
    init : list of ndarray, optional
        Initial bases. Defaults to the L1-HOSVD solution (itself initialized
        at HOSVD).
    proj : {"L2", "L1"}
        Projection step, see above.
    T : int, optional
        Maximum number of sweeps. Defaults to 10 for ``proj="L1"`` and to no
        limit for ``proj="L2"``.
    order : {None, "random"} or sequence of int
        Order of the mode updates within a sweep: increasing (default), a new
        random permutation at every sweep, or a fixed permutation.
    tol : float
        Tolerance for both the sweeps and the inner L1-PCA.
    maxit : int
        Maximum number of sweeps and of inner L1-PCA iterations.
    rng : int or numpy.random.Generator, optional
        Seed or generator for ``order="random"``.
    X_clean, U_true : optional
        Noise-free tensor and true bases, for the error histories (see
        :func:`l1hosvd`).

    Returns
    -------
    TuckerResult
        ``history`` holds ``l1_metric``, ``subspace_error`` and
        ``reconstruction_error`` after every inner L1-PCA update (first entry:
        at ``init``), with the updated ``mode`` and ``sweep``.
        ``first_sweep`` holds the result after the first sweep (the T=1
        configuration).
    """
    X, factors, T = _setup(X, ranks, init, proj, T, tol)
    N = X.ndim
    if order is None:
        order = list(range(N))
    elif isinstance(order, str):
        if order != "random":
            raise ValueError(f"order must be None, 'random' or a permutation, got {order!r}")
    else:
        order = [int(n) for n in order]
        if sorted(order) != list(range(N)):
            raise ValueError(f"order must be a permutation of 0..{N - 1}, got {order}")
    rng = np.random.default_rng(rng)

    tracker = _Tracker(X, X_clean, U_true)
    tracker.record(factors, sweep=0)
    first_sweep = None
    prev = 0.0
    sweep = 0
    while sweep < maxit:
        sweep += 1
        for n in (rng.permutation(N) if order == "random" else order):
            if ranks[n] == X.shape[n]:
                factors[n] = np.eye(ranks[n])
            else:
                A = _projected_unfolding(X, factors, n, proj)

                def update(Q, n=n, sweep=sweep):
                    factors[n] = Q
                    tracker.record(factors, mode=int(n), sweep=sweep)

                factors[n], _ = l1pca(A, factors[n], tol=tol, maxit=maxit, callback=update)
            core = multi_mode_product(X, factors, transpose=True)
        metric = float(np.abs(core).sum())
        if sweep == 1:
            first_sweep = TuckerResult(list(factors), core, tracker.snapshot())
        if _rel_change(metric, prev) < tol or sweep >= T:
            break
        prev = metric
    return TuckerResult(factors, core, tracker.history, first_sweep)


def l1hooi_random(X, ranks, init=None, proj="L2", T=None, tol=1e-8, maxit=1000, rng=None,
                  X_clean=None, U_true=None):
    """L1-HOOI with one randomly selected mode per iteration.

    Instead of sweeping over all modes, each iteration updates a single mode,
    drawn uniformly among the modes other than the one updated last, exactly
    as :func:`l1hooi` would update it (L1-PCA of the ``proj`` projection).
    Iterations stop when the relative change of the L1 metric between two
    consecutive updates is below ``tol``, after ``T`` updates, or after
    ``maxit`` updates.

    Parameters are those of :func:`l1hooi`, except that ``T`` counts
    single-mode updates. As in the original MATLAB code, ``T`` defaults to 10
    for ``proj="L1"`` and to no limit for ``proj="L2"``.

    Returns
    -------
    TuckerResult
        ``history["sweep"]`` holds the iteration (single-mode update) index.
        ``first_sweep`` holds the result at the first iteration at which every
        mode has been updated at least once.
    """
    X, factors, T = _setup(X, ranks, init, proj, T, tol)
    N = X.ndim
    if N < 2:
        raise ValueError("l1hooi_random needs a tensor with at least 2 modes")
    rng = np.random.default_rng(rng)

    tracker = _Tracker(X, X_clean, U_true)
    tracker.record(factors, sweep=0)
    pending = {n for n in range(N) if ranks[n] < X.shape[n]}
    first_sweep = None
    prev = 0.0
    last = None
    for it in range(1, maxit + 1):
        if last is None:
            n = int(rng.integers(N))
        else:
            n = int(rng.integers(N - 1))
            n += n >= last
        last = n
        if ranks[n] == X.shape[n]:
            factors[n] = np.eye(ranks[n])
        else:
            A = _projected_unfolding(X, factors, n, proj)

            def update(Q, n=n, it=it):
                factors[n] = Q
                tracker.record(factors, mode=n, sweep=it)

            factors[n], _ = l1pca(A, factors[n], tol=tol, maxit=maxit, callback=update)
            pending.discard(n)
        core = multi_mode_product(X, factors, transpose=True)
        metric = float(np.abs(core).sum())
        if first_sweep is None and not pending:
            first_sweep = TuckerResult(list(factors), core, tracker.snapshot())
        if _rel_change(metric, prev) < tol or it >= T:
            break
        prev = metric
    if first_sweep is None:
        first_sweep = TuckerResult(list(factors), core, tracker.snapshot())
    return TuckerResult(factors, core, tracker.history, first_sweep)


def l1t_l1f(X, ranks, T=1, init=None, **kwargs):
    """L1T-L1F: L1-Tucker followed by L1-Fitting (Algorithm 1 of the paper).

    Starts from the L1-HOSVD bases and, for each mode in turn, L1-fits the
    data on the bases of the other modes and updates the basis by L1-PCA of
    the fitted data. ``T`` > 1 repeats these steps (the paper reports T=10).

    This is ``l1hooi(X, ranks, proj="L1", T=T, ...)``; any other keyword
    argument of :func:`l1hooi` is accepted.
    """
    return l1hooi(X, ranks, init=init, proj="L1", T=T, **kwargs)
