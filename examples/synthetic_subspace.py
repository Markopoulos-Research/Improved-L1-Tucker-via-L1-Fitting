"""Reproduce the synthetic subspace-estimation study of the paper (Sec. IV-A, Fig. 2).

A Tucker tensor of size 5 x 8 x 5 x 8 x 5 with ranks (3, 3, 2, 2, 2) and a
N(0, 9) core is corrupted by N(0, 1) noise and 10 sparse N(0, sigma_o^2)
outliers. The true bases are fixed, the noise and outliers are redrawn in
every trial, and the mean MANSSE of each method is reported against the
Outlier-to-Noise Ratio (ONR).

    python examples/synthetic_subspace.py --trials 100 --out results/fig2

writes ``fig2.csv`` and, if matplotlib is installed, ``fig2.png``. The paper
reports 1000 trials per point (the original MATLAB script defaults to 100).
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from l1tucker import hooi, hosvd, l1hooi, l1hosvd, subspace_error  # noqa: E402
from l1tucker.synthetic import low_rank_tensor, outlier_to_noise_ratio, sparse_outliers  # noqa: E402

DIMS = (5, 8, 5, 8, 5)
RANKS = (3, 3, 2, 2, 2)
CORE_STD = 3.0
NOISE_STD = 1.0
N_OUTLIERS = 10
TOL = 1e-8
MAXIT = 100
METHODS = ["HOSVD", "HOOI", "L1-HOSVD", "L1-HOOI", "L1T-L1F", "L1T-L1F (T=10)"]


def run_trial(job):
    seed, i, outlier_std, trial = job
    X, _, U_true = low_rank_tensor(DIMS, RANKS, CORE_STD, rng=seed)  # fixed across trials
    rng = np.random.default_rng([seed, i, trial])
    X_corr = X + NOISE_STD * rng.standard_normal(DIMS) + sparse_outliers(DIMS, outlier_std, count=N_OUTLIERS, rng=rng)

    # Same initialization as the MATLAB experiment: HOSVD for the L2 methods,
    # L1-HOSVD started at HOSVD for the L1 methods.
    U_hosvd = hosvd(X_corr, RANKS).factors
    U_l1hosvd = l1hosvd(X_corr, RANKS, init=U_hosvd, tol=TOL, maxit=1000).factors
    l1f = l1hooi(X_corr, RANKS, init=U_l1hosvd, proj="L1", T=10, tol=TOL, maxit=MAXIT)
    estimates = {
        "HOSVD": U_hosvd,
        "HOOI": hooi(X_corr, RANKS, init=U_hosvd, tol=TOL, maxit=MAXIT).factors,
        "L1-HOSVD": U_l1hosvd,
        "L1-HOOI": l1hooi(X_corr, RANKS, init=U_l1hosvd, proj="L2", tol=TOL, maxit=MAXIT).factors,
        "L1T-L1F": l1f.first_sweep.factors,
        "L1T-L1F (T=10)": l1f.factors,
    }
    return i, {name: subspace_error(U_true, U) for name, U in estimates.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--outlier-stds", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 14, 16])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results/fig2")
    args = parser.parse_args()

    stds = args.outlier_stds
    onr = [outlier_to_noise_ratio(DIMS, s, NOISE_STD, count=N_OUTLIERS) for s in stds]
    jobs = [(args.seed, i, s, t) for i, s in enumerate(stds) for t in range(args.trials)]
    errors = [{m: [] for m in METHODS} for _ in stds]

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for done, (i, errs) in enumerate(pool.map(run_trial, jobs, chunksize=4), 1):
            for m in METHODS:
                errors[i][m].append(errs[m])
            if done % max(1, len(jobs) // 20) == 0:
                print(f"{done}/{len(jobs)} trials, {time.time() - start:.0f} s", flush=True)

    mean = [{m: float(np.mean(e[m])) for m in METHODS} for e in errors]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma_o", "ONR"] + METHODS)
        for s, r, row in zip(stds, onr, mean):
            writer.writerow([s, f"{r:.4f}"] + [f"{row[m]:.4f}" for m in METHODS])

    print(f"\nMean MANSSE over {args.trials} trials")
    print(f"{'ONR':>6} " + " ".join(f"{m:>15}" for m in METHODS))
    for r, row in zip(onr, mean):
        print(f"{r:6.3f} " + " ".join(f"{row[m]:15.4f}" for m in METHODS))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in METHODS:
        ax.plot(onr, [row[m] for row in mean], marker="o", ms=4, ls="--" if "T=10" in m else "-", label=m)
    ax.set_xlabel("ONR")
    ax.set_ylabel("MANSSE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)
    print(f"\nWrote {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()
