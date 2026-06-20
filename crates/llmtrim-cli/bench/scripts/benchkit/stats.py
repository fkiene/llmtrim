"""Bootstrap estimators shared by the sweep (CI of reduction) and the live leg (paired diff)."""
import random


def bootstrap_weighted_reduction(pairs, n=10000, seed=0):
    """95% CI of the TOKEN-WEIGHTED reduction = 1 - sum(after)/sum(before), resampling
    (before, after) case pairs with replacement. Must match the point statistic we report,
    or the CI lands off the estimate (an earlier bug: per-case-mean CI under a token-weighted
    headline)."""
    if len(pairs) < 2:
        return (None, None)
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        sb = sa = 0
        for _ in range(len(pairs)):
            b, a = pairs[rng.randrange(len(pairs))]
            sb += b
            sa += a
        samples.append(100.0 * (1 - sa / sb) if sb else 0.0)
    samples.sort()
    return (samples[int(0.025 * n)], samples[int(0.975 * n)])


def percentile(values, q):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q / 100.0 * (len(vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(vals):
        return vals[lo] + frac * (vals[lo + 1] - vals[lo])
    return vals[lo]


def paired_bootstrap_diff(diffs, n=10000, seed=0):
    """Mean and 95% CI of per-sample (llmtrim - competitor) score differences. CI excluding 0
    means a significant quality difference - handles continuous scorers, unlike McNemar."""
    diffs = [d for d in diffs if d is not None]
    if len(diffs) < 2:
        return (None, None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(sum(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))) / len(diffs))
    means.sort()
    return (sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n)])
