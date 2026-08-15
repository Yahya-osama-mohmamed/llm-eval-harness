"""Bootstrap confidence intervals.

Every number this harness reports carries an interval, because the whole claim of
the project is that judge scores are measurements. A measurement without an
interval is a number with an opinion attached.

BCa is the default, on theory rather than on evidence from this repository. The
argument for it is that percentile intervals are fine for a mean but skewed for a
bounded statistic like kappa near its ceiling, and BCa corrects bias and skew at
the cost of a jackknife pass.

That argument is not yet supported here. Measured coverage for kappa at n=150 is
identical for both methods to three decimals (0.926 against a nominal 0.95 —
RESULTS.md section 2.1), so BCa currently buys nothing measurable and both
methods under-cover. Treat small-n kappa intervals as approximate. The default
stands because BCa is not worse, and its advantage may appear at other n; that is
an open question, not a settled one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

__all__ = ["CI", "bootstrap_ci", "paired_bootstrap_diff"]

Method = Literal["percentile", "bca"]


@dataclass(frozen=True)
class CI:
    """A point estimate and its interval."""

    estimate: float
    low: float
    high: float
    confidence: float
    method: str
    n_resamples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        """True when ``value`` lies outside the interval — the usual way to ask
        'is this difference distinguishable from zero?'."""
        return value < self.low or value > self.high

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _resample_indices(n: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_ci(
    data: Sequence | tuple[Sequence, ...],
    statistic: Callable[..., float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    method: Method = "bca",
    seed: int | None = 0,
) -> CI:
    """Bootstrap a statistic over items.

    ``data`` is either one sequence or a tuple of equal-length sequences, which
    are resampled *together* — so a (human_label, judge_label) pair stays paired
    and the interval reflects item sampling rather than pretending the two
    columns are independent.

    ``seed`` defaults to 0 rather than None: a confidence interval that moves
    when you rerun the report is not reproducible evidence.
    """
    cols = tuple(np.asarray(c) for c in (data if isinstance(data, tuple) else (data,)))
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("all columns must have the same length")
    if n < 2:
        raise ValueError("need at least 2 items to bootstrap")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(*cols))

    idx = _resample_indices(n, n_resamples, rng)
    boot = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        take = idx[b]
        boot[b] = statistic(*(c[take] for c in cols))

    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return CI(theta_hat, float("nan"), float("nan"), confidence, method, n_resamples)

    alpha = 1.0 - confidence
    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return CI(theta_hat, float(lo), float(hi), confidence, "percentile", n_resamples)

    if method != "bca":
        raise ValueError(f"unknown method {method!r}")

    # bias correction
    prop_below = float(np.mean(boot < theta_hat))
    if prop_below in (0.0, 1.0):
        # BCa is undefined at the edge; fall back rather than emit inf
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return CI(theta_hat, float(lo), float(hi), confidence, "percentile (bca degenerate)", n_resamples)
    z0 = stats.norm.ppf(prop_below)

    # acceleration via jackknife
    jack = np.empty(n, dtype=float)
    all_idx = np.arange(n)
    for i in range(n):
        keep = np.delete(all_idx, i)
        jack[i] = statistic(*(c[keep] for c in cols))
    jack = jack[np.isfinite(jack)]
    jack_mean = jack.mean()
    num = float(np.sum((jack_mean - jack) ** 3))
    den = 6.0 * float(np.sum((jack_mean - jack) ** 2)) ** 1.5
    a = 0.0 if den == 0 else num / den

    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    lo, hi = np.quantile(boot, [np.clip(a1, 0, 1), np.clip(a2, 0, 1)])
    return CI(theta_hat, float(lo), float(hi), confidence, "bca", n_resamples)


def paired_bootstrap_diff(
    data: tuple[Sequence, ...],
    statistic_a: Callable[..., float],
    statistic_b: Callable[..., float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    method: Method = "bca",
    seed: int | None = 0,
) -> CI:
    """Interval for ``statistic_a - statistic_b`` on the same resampled items.

    Use this to compare two judges on one eval set. Resampling both on the same
    item draw removes the shared item-difficulty variance, which is why a paired
    interval is usually much tighter than differencing two independent ones —
    and why comparing two separately-computed CIs by eye is the wrong test.
    """
    return bootstrap_ci(
        data,
        lambda *cols: statistic_a(*cols) - statistic_b(*cols),
        n_resamples=n_resamples,
        confidence=confidence,
        method=method,
        seed=seed,
    )
