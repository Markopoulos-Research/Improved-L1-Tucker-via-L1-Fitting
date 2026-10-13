"""Synthetic data for the subspace-estimation experiment of the paper (Sec. IV-A)."""

import numpy as np

from .tensor import multi_mode_product


def random_orthonormal_bases(dims, ranks, rng=None):
    """One random ``D_n x d_n`` matrix with orthonormal columns per mode."""
    rng = np.random.default_rng(rng)
    return [np.linalg.qr(rng.standard_normal((D, d)))[0] for D, d in zip(dims, ranks)]


def low_rank_tensor(dims, ranks, core_std=1.0, rng=None):
    """Tucker-structured tensor ``G x_1 U_1 ... x_N U_N`` with Gaussian core.

    Returns ``(X, core, factors)``.
    """
    rng = np.random.default_rng(rng)
    core = core_std * rng.standard_normal(tuple(ranks))
    factors = random_orthonormal_bases(dims, ranks, rng)
    return multi_mode_product(core, factors), core, factors


def outlier_mask(shape, count=None, probability=None, rng=None):
    """Boolean mask of outlier locations.

    Exactly one of ``count`` (number of corrupted entries, drawn without
    replacement) or ``probability`` (independent per-entry corruption
    probability) must be given.
    """
    if (count is None) == (probability is None):
        raise ValueError("give exactly one of count or probability")
    rng = np.random.default_rng(rng)
    if probability is not None:
        return rng.random(shape) <= probability
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    mask[rng.choice(mask.size, size=count, replace=False)] = True
    return mask.reshape(shape)


def sparse_outliers(shape, std, count=None, probability=None, rng=None):
    """Sparse tensor whose nonzero entries are drawn from ``N(0, std^2)``."""
    rng = np.random.default_rng(rng)
    mask = outlier_mask(shape, count=count, probability=probability, rng=rng)
    return mask * (std * rng.standard_normal(shape))


def outlier_to_noise_ratio(shape, outlier_std, noise_std, count=None, probability=None):
    """Outlier-to-Noise Ratio ``E||O||_F^2 / E||N||_F^2``, Eq. (12)."""
    if (count is None) == (probability is None):
        raise ValueError("give exactly one of count or probability")
    size = int(np.prod(shape))
    expected_outliers = count if count is not None else size * probability
    return expected_outliers * outlier_std ** 2 / (size * noise_std ** 2)
