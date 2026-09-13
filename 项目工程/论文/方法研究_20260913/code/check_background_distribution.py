"""Numerical check of the oracle marginal-background property, not a proof."""

from paired_loss import ROOT, gaussian_knockoff, contexts_for
from common import save_json
import numpy as np

rng = np.random.default_rng(83153)
n, p = 100000, 6
covariance = .6 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
x = rng.multivariate_normal(np.zeros(p), covariance, n)
xk, meta = gaussian_knockoff(x, covariance, 17303)
inverse = np.linalg.inv(covariance)
noise_cov = 2 * meta["s"] * np.eye(p) - meta["s"]**2 * inverse
chol = np.linalg.cholesky((noise_cov + noise_cov.T) / 2)
noise = np.random.default_rng(17303).normal(size=x.shape)
reference = np.einsum("ij,jk->ik", x, np.eye(p) - meta["s"] * inverse, optimize=False)
reference += np.einsum("ij,kj->ik", noise, chol, optimize=False)
blas_error = float(np.abs(xk - reference).max())
assert np.isfinite(xk).all() and blas_error < 1e-12
contexts = contexts_for(x, xk, draws=8, seed=88661)
rows = []
for i, context in enumerate(contexts):
    error = float(np.abs(np.cov(context.T) - covariance).max())
    mean_error = float(np.abs(context.mean(0)).max())
    assert error < .035 and mean_error < .02
    rows.append({"context": i, "covariance_max_error": error, "mean_max_error": mean_error})

midpoint = (x + xk) / 2
midpoint_target = covariance - .5 * meta["s"] * np.eye(p)
midpoint_error = float(np.abs(np.cov(midpoint.T) - midpoint_target).max())
assert midpoint_error < .035
assert np.abs(np.cov(midpoint.T) - covariance).max() > .1
save_json(ROOT / "results/background_distribution_check.json", {
    "passed": True, "rows": n, "features": p, "seed": 83153,
    "blas_vs_explicit_einsum_max_error": blas_error,
    "random_contexts": rows,
    "midpoint_covariance_expected": midpoint_target,
    "midpoint_expected_covariance_error": midpoint_error,
    "midpoint_departure_from_original_covariance": float(np.abs(np.cov(midpoint.T) - covariance).max()),
    "scope": "exact Gaussian oracle; averaged over random masks, not an Olist validity certificate",
    "after_results_theoretical_verification_only": True,
})
print("Oracle background distribution check passed")
