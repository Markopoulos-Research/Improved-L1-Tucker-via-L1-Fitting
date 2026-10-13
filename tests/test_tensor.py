import numpy as np

from l1tucker import fold, kron_except, mode_product, multi_mode_product, unfold


def test_unfold_matches_matlab_tenmat():
    # X = reshape(1:24, [2 3 4]) in MATLAB; tenmat(X, 2) has rows j and
    # columns (i, k) with i varying fastest.
    X = np.arange(1, 25).reshape((2, 3, 4), order="F")
    M = unfold(X, 1)
    assert M.shape == (3, 8)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert M[j, i + 2 * k] == X[i, j, k]


def test_fold_inverts_unfold():
    X = np.random.default_rng(0).standard_normal((3, 4, 5, 2))
    for n in range(X.ndim):
        np.testing.assert_array_equal(fold(unfold(X, n), n, X.shape), X)


def test_mode_product_definition():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3, 4, 5))
    M = rng.standard_normal((2, 4))
    Y = mode_product(X, M, 1)
    assert Y.shape == (3, 2, 5)
    np.testing.assert_allclose(unfold(Y, 1), M @ unfold(X, 1))


def test_tucker_unfolding_identity():
    rng = np.random.default_rng(2)
    G = rng.standard_normal((2, 3, 2, 2))
    Us = [rng.standard_normal((D, d)) for D, d in zip((4, 5, 3, 6), G.shape)]
    X = multi_mode_product(G, Us)
    for n in range(X.ndim):
        np.testing.assert_allclose(unfold(X, n), Us[n] @ unfold(G, n) @ kron_except(Us, n).T, atol=1e-12)
