"""Small, inspectable optimal-transport tools for the DRP lesson."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, kron, vstack, csr_matrix
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import norm


def gaussian_distances(mu1, sigma1, mu2, sigma2):
    """Population W1 and W2 on R. Arguments are standard deviations."""
    if min(sigma1, sigma2) < 0:
        raise ValueError("Standard deviations must be nonnegative")
    m, s = abs(mu1 - mu2), abs(sigma1 - sigma2)
    w1 = m if s == 0 else (s * np.sqrt(2 / np.pi) * np.exp(-m*m/(2*s*s))
                           + m * (2 * norm.cdf(m/s) - 1))
    return float(w1), float(np.hypot(m, s))


def empirical_wp_equal(x, y, p=2):
    """Exact 1D empirical Wp for equally weighted samples of equal size."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
        raise ValueError("Use nonempty 1D samples of equal size")
    if p < 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Require p >= 1 and finite samples")
    return float(np.mean(abs(np.sort(x)-np.sort(y))**p)**(1/p))


def probability_weights(w):
    w = np.asarray(w, float)
    if w.ndim != 1 or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, with positive sum")
    return w / w.sum()


def exact_ot(x, y, a=None, b=None, p=1):
    """Small discrete balanced OT LP. Returns Wp, plan, cost matrix.

    Supports have shape (number of atoms, dimension); costs are Euclidean**p.
    'Exact' means unregularized optimization, subject to solver tolerances.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) == 0:
        raise ValueError("Use nonempty supports with matching feature dimensions")
    if p < 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Require p >= 1 and finite coordinates")
    n, m = len(x), len(y)
    a = probability_weights(np.ones(n) if a is None else a)
    b = probability_weights(np.ones(m) if b is None else b)
    if len(a) != n or len(b) != m:
        raise ValueError("One weight is required per support point")
    cost = cdist(x, y)**p
    rows = kron(eye(n), csr_matrix(np.ones((1, m))))
    cols = kron(csr_matrix(np.ones((1, n))), eye(m))
    result = linprog(cost.ravel(), A_eq=vstack([rows, cols]).tocsr(),
                     b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    plan = result.x.reshape(n, m)
    if max(abs(plan.sum(1)-a).max(), abs(plan.sum(0)-b).max()) > 1e-7:
        raise RuntimeError("Transport marginals failed verification")
    return float(max(result.fun, 0)**(1/p)), plan, cost


def sinkhorn_log(a, b, cost, epsilon=0.1, tol=1e-10, max_iter=20000):
    """Log-domain Sinkhorn for positive masses. Returns plan and diagnostics.

    Objective: <P,C> + epsilon * sum P_ij*(log(P_ij)-1).
    Transport cost alone is NOT this regularized objective or a metric.
    """
    a, b = probability_weights(a), probability_weights(b)
    cost = np.asarray(cost, float)
    if np.any(a == 0) or np.any(b == 0) or epsilon <= 0:
        raise ValueError("Remove zero-mass atoms and use epsilon > 0")
    if cost.shape != (len(a), len(b)) or not np.isfinite(cost).all():
        raise ValueError("Invalid cost matrix")
    logk, logv = -cost / epsilon, np.zeros(len(b))
    for iteration in range(max_iter):
        logu = np.log(a) - logsumexp(logk + logv[None, :], axis=1)
        logv = np.log(b) - logsumexp(logk + logu[:, None], axis=0)
        if iteration % 20 == 0 or iteration == max_iter - 1:
            logplan = logu[:, None] + logk + logv[None, :]
            plan = np.exp(logplan)
            error = max(abs(plan.sum(1)-a).max(), abs(plan.sum(0)-b).max())
            if error < tol:
                break
    if error >= tol:
        raise RuntimeError(f"Sinkhorn did not converge: marginal error {error:g}")
    transport = float(np.sum(plan * cost))
    objective = transport + epsilon * float(np.sum(plan * (logplan - 1)))
    return plan, dict(transport_cost=transport, objective=objective,
                      marginal_error=float(error), iterations=iteration+1)


def mmd2_biased(x, y, bandwidth=1.0):
    """Nonnegative (up to roundoff) empirical squared Gaussian-kernel MMD."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if bandwidth <= 0 or min(len(x), len(y)) == 0:
        raise ValueError("Use a positive bandwidth and nonempty samples")
    def kernel(u, v):
        return np.exp(-cdist(u, v, 'sqeuclidean') / (2 * bandwidth**2))
    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2*kernel(x, y).mean())


def image_measure(image):
    """A nonnegative image becomes mass on (column,row) pixel coordinates."""
    image = np.asarray(image, float)
    if image.ndim != 2:
        raise ValueError("Use a grayscale 2D array")
    weights = probability_weights(image.ravel())
    row, col = np.indices(image.shape)
    coords = np.c_[col.ravel(), row.ravel()]
    keep = weights > 0
    return coords[keep], weights[keep]


def signature_level2(path):
    """Levels 1 and 2 of a piecewise-linear path, via Chen's identity.

    The constant level 0 is omitted. For time series pass (time, value).
    This is an ordinary truncated signature, not a Volterra signature.
    """
    path = np.asarray(path, float)
    d = path.shape[1]
    level1, level2 = np.zeros(d), np.zeros((d, d))
    for increment in np.diff(path, axis=0):
        level2 += np.outer(level1, increment) + 0.5*np.outer(increment, increment)
        level1 += increment
    return np.r_[level1, level2.ravel()]


def signature_level3(path):
    """Ordinary piecewise-linear signature through level 3, excluding level 0."""
    path = np.asarray(path, float)
    d = path.shape[1]
    s1, s2, s3 = np.zeros(d), np.zeros((d, d)), np.zeros((d, d, d))
    for dx in np.diff(path, axis=0):
        dx2 = np.outer(dx, dx)
        s3 += (np.einsum('ij,k->ijk', s2, dx)
               + 0.5*np.einsum('i,jk->ijk', s1, dx2)
               + np.einsum('i,j,k->ijk', dx, dx, dx)/6)
        s2 += np.outer(s1, dx) + dx2/2
        s1 += dx
    return np.r_[s1, s2.ravel(), s3.ravel()]
