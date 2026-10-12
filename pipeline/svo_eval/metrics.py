import numpy as np
from scipy.stats import binomtest


def first_gold_rank(ranked, gold):
    for i, c in enumerate(ranked):
        if c in gold:
            return i + 1
    return None


def success_vector(ranks, k):
    return np.array([(r is not None) and (r <= k) for r in ranks], dtype=bool)


def rr_vector(ranks, k):
    return np.array([1.0 / r if (r is not None and r <= k) else 0.0 for r in ranks], dtype=float)


def precision_vector(label_lists, k):
    return np.array([float(np.mean([bool(x) for x in labels[:k]])) if len(labels[:k]) else 0.0
                     for labels in label_lists], dtype=float)


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap(a, b, n_boot=2000, seed=0, alpha=0.05):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, n_boot=n_boot, seed=seed, alpha=alpha)


def mcnemar_p(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)


def randomisation_p(diff, n_perm=100000, seed=0):
    """Two-sided paired randomisation (sign-flip) test on the mean of the per-query differences."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.sum())
    hits = 0
    for _ in range(n_perm // 10000):
        signs = rng.choice([-1.0, 1.0], size=(10000, len(d)))
        hits += int((np.abs(signs @ d) >= obs - 1e-12).sum())
    return (hits + 1) / (n_perm + 1)


def holm(pvals):
    """Holm step-down adjusted p-values for a dict {key: p}."""
    keys = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * pvals[k]))
        adj[k] = running
    return adj


def fleiss_kappa(counts):
    m = np.asarray(counts, dtype=float)
    n_items, _ = m.shape
    n_raters = m.sum(axis=1)[0]
    p_i = ((m ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_j = m.sum(axis=0) / (n_items * n_raters)
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def summary(values, n_boot=2000, seed=0):
    mean, lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    return {"mean": 100 * mean, "lo": 100 * lo, "hi": 100 * hi, "n": int(len(values))}
