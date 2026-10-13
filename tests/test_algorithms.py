import numpy as np
import pytest

from l1tucker import (hooi, hosvd, l1_regression, l1hooi, l1hooi_random, l1hosvd, l1pca, l1t_l1f,
                      reconstruction_error, subspace_error)
from l1tucker.l1fit import _l1_regression_primal
from l1tucker.synthetic import low_rank_tensor, outlier_to_noise_ratio, sparse_outliers

DIMS = (5, 8, 5, 8, 5)
RANKS = (3, 3, 2, 2, 2)


def corrupted(seed, outlier_std=16.0):
    rng = np.random.default_rng(seed)
    X, _, U = low_rank_tensor(DIMS, RANKS, core_std=3.0, rng=rng)
    X_corr = X + rng.standard_normal(DIMS) + sparse_outliers(DIMS, outlier_std, count=10, rng=rng)
    return X, U, X_corr


def test_l1_regression_dual_matches_primal():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((300, 6))
    x = W @ rng.standard_normal(6) + rng.standard_normal(300)
    g = l1_regression(W, x)
    g_primal = _l1_regression_primal(W, x)
    assert np.abs(W @ g - x).sum() == pytest.approx(np.abs(W @ g_primal - x).sum(), rel=1e-9)


def test_l1_regression_ignores_sparse_outliers():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((200, 5))
    g0 = rng.standard_normal(5)
    x = W @ g0
    x[rng.choice(200, 10, replace=False)] += 100 * rng.standard_normal(10)
    np.testing.assert_allclose(l1_regression(W, x), g0, atol=1e-8)


def test_l1pca_is_monotone_and_orthonormal():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((10, 40))
    Q, metric = l1pca(A, 3)
    assert np.all(np.diff(metric) >= -1e-9)
    np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert np.abs(A.T @ Q).sum() >= metric[-1] - 1e-9


def test_l2_baselines_recover_clean_tensor():
    X, U, _ = corrupted(3)
    for res in (hosvd(X, RANKS), hooi(X, RANKS)):
        assert subspace_error(U, res.factors) < 1e-20
        assert reconstruction_error(X, res.reconstruct()) < 1e-20


@pytest.mark.parametrize("proj", ["L2", "L1"])
def test_l1hooi_recovers_clean_tensor(proj):
    X, U, _ = corrupted(4)
    res = l1hooi(X, RANKS, proj=proj, U_true=U)
    assert subspace_error(U, res.factors) < 1e-12
    assert len(res.history["subspace_error"]) == len(res.history["l1_metric"])
    for V in res.factors:
        np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-12)


def test_l1hooi_l2proj_metric_is_monotone():
    _, _, X_corr = corrupted(5)
    res = l1hooi(X_corr, RANKS, proj="L2")
    assert np.all(np.diff(res.history["l1_metric"]) >= -1e-9 * res.history["l1_metric"][-1])


def test_optional_arguments_can_be_omitted():
    # The MATLAB L1HOOI required X_clean and Un_true despite documenting them as optional.
    _, _, X_corr = corrupted(6)
    res = l1t_l1f(X_corr, RANKS)
    assert res.history["subspace_error"] == [] and res.history["reconstruction_error"] == []
    assert res.core.shape == RANKS


def test_full_rank_mode_gets_identity():
    X, _, _ = corrupted(7)
    ranks = (5, 3, 2, 2, 2)
    for res in (l1hosvd(X, ranks), l1hooi(X, ranks, proj="L1", T=1)):
        np.testing.assert_array_equal(res.factors[0], np.eye(5))


def test_first_sweep_equals_t1_run():
    _, U, X_corr = corrupted(8)
    init = l1hosvd(X_corr, RANKS).factors
    full = l1t_l1f(X_corr, RANKS, T=3, init=init)
    single = l1t_l1f(X_corr, RANKS, T=1, init=init)
    for A, B in zip(full.first_sweep.factors, single.factors):
        np.testing.assert_allclose(A, B, atol=1e-10)


def test_random_order_and_custom_order():
    _, _, X_corr = corrupted(9)
    a = l1hooi(X_corr, RANKS, order="random", rng=0)
    b = l1hooi(X_corr, RANKS, order="random", rng=0)
    for A, B in zip(a.factors, b.factors):
        np.testing.assert_array_equal(A, B)
    c = l1hooi(X_corr, RANKS, order=[4, 3, 2, 1, 0], T=1)
    assert [m for m in c.history["mode"] if m is not None][0] == 4
    with pytest.raises(ValueError):
        l1hooi(X_corr, RANKS, order=[0, 0, 1, 2, 3])


@pytest.mark.parametrize("proj", ["L2", "L1"])
def test_l1hooi_random(proj):
    X, U, X_corr = corrupted(10)
    a = l1hooi_random(X_corr, RANKS, proj=proj, rng=1)
    b = l1hooi_random(X_corr, RANKS, proj=proj, rng=1)
    for A, B in zip(a.factors, b.factors):
        np.testing.assert_array_equal(A, B)
    # one mode per iteration, never the same mode twice in a row
    modes = {}
    for m, it in zip(a.history["mode"][1:], a.history["sweep"][1:]):
        assert modes.setdefault(it, m) == m
    seq = [modes[it] for it in sorted(modes)]
    assert all(p != q for p, q in zip(seq, seq[1:]))
    if proj == "L1":
        assert len(seq) <= 10  # T defaults to 10 single-mode updates
    # first_sweep is taken right after the last unvisited mode got its first update
    k = len(set(a.first_sweep.history["sweep"][1:]))
    assert set(seq[:k]) == set(range(len(DIMS))) and seq[k - 1] not in seq[:k - 1]
    clean = l1hooi_random(X, RANKS, proj=proj, rng=2)
    assert subspace_error(U, clean.factors) < 1e-12


def test_l1t_l1f_is_robust_to_outliers():
    # Paper, Fig. 2 at ONR = 0.32: HOOI breaks down, L1T-L1F does not.
    errs = {"hooi": [], "l1t_l1f": []}
    for seed in range(3):
        _, U, X_corr = corrupted(100 + seed)
        errs["hooi"].append(subspace_error(U, hooi(X_corr, RANKS).factors))
        errs["l1t_l1f"].append(subspace_error(U, l1t_l1f(X_corr, RANKS).factors))
    assert np.mean(errs["l1t_l1f"]) < 0.5 * np.mean(errs["hooi"])


def test_onr_matches_paper():
    assert outlier_to_noise_ratio(DIMS, 16, 1, count=10) == pytest.approx(0.32)
