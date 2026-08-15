"""Bootstrap intervals, validated by coverage simulation.

The promise a 95% interval makes is that it contains the truth 95% of the time.
That is a testable claim, and these tests test it — generating data with a known
answer, building many intervals, and counting how often the truth falls inside.
An interval that does not cover is worse than no interval, because it looks like
evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_eval.metrics.agreement import cohens_kappa
from llm_eval.metrics.resample import CI, bootstrap_ci, paired_bootstrap_diff

# --------------------------------------------------------------------------
# coverage — the headline validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_coverage_of_a_mean(method: str) -> None:
    """500 datasets from a known distribution; the true mean should land inside
    roughly 95% of the intervals."""
    rng = np.random.default_rng(42)
    true_mean, trials, n = 0.3, 500, 120
    covered = 0
    for t in range(trials):
        sample = rng.binomial(1, true_mean, size=n)
        ci = bootstrap_ci(sample, np.mean, n_resamples=800, method=method, seed=t)
        covered += ci.low <= true_mean <= ci.high
    rate = covered / trials
    assert 0.91 <= rate <= 0.985, f"{method} coverage was {rate:.3f}"


def test_coverage_of_kappa() -> None:
    """Coverage for a skewed, bounded statistic.

    The tolerance here is deliberately wider than for the mean: kappa at this n
    covers about 0.926 against a nominal 0.95, and the test asserts what is
    actually true rather than what the nominal level promises.
    """
    rng = np.random.default_rng(7)
    trials, n, p_agree = 300, 150, 0.85

    # ground-truth kappa for this generator, from a large sample
    big_a = rng.integers(0, 2, size=400_000)
    big_b = np.where(rng.random(400_000) < p_agree, big_a, 1 - big_a)
    true_kappa = cohens_kappa(big_a, big_b)

    covered = 0
    for t in range(trials):
        a = rng.integers(0, 2, size=n)
        b = np.where(rng.random(n) < p_agree, a, 1 - a)
        ci = bootstrap_ci((a, b), cohens_kappa, n_resamples=500, method="bca", seed=t)
        covered += ci.low <= true_kappa <= ci.high
    rate = covered / trials
    assert 0.88 <= rate <= 0.99, f"kappa coverage was {rate:.3f} (true kappa {true_kappa:.3f})"


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------

def test_interval_contains_estimate() -> None:
    rng = np.random.default_rng(1)
    ci = bootstrap_ci(rng.normal(5, 2, size=200), np.mean)
    assert ci.low <= ci.estimate <= ci.high


def test_interval_narrows_with_n() -> None:
    rng = np.random.default_rng(2)
    small = bootstrap_ci(rng.normal(0, 1, size=50), np.mean, n_resamples=2000, seed=1)
    large = bootstrap_ci(rng.normal(0, 1, size=5000), np.mean, n_resamples=2000, seed=1)
    assert large.width < small.width / 5


def test_paired_resampling_keeps_columns_aligned() -> None:
    """If the columns were resampled independently the correlation would be
    destroyed and kappa would collapse toward zero."""
    rng = np.random.default_rng(4)
    a = rng.integers(0, 2, size=400)
    b = np.where(rng.random(400) < 0.9, a, 1 - a)
    ci = bootstrap_ci((a, b), cohens_kappa, n_resamples=600)
    assert ci.low > 0.6


def test_seed_makes_it_reproducible() -> None:
    rng = np.random.default_rng(6)
    data = rng.normal(size=300)
    first = bootstrap_ci(data, np.mean, n_resamples=500, seed=99)
    second = bootstrap_ci(data, np.mean, n_resamples=500, seed=99)
    assert (first.low, first.high) == (second.low, second.high)


def test_different_seeds_differ_slightly() -> None:
    rng = np.random.default_rng(8)
    data = rng.normal(size=300)
    a = bootstrap_ci(data, np.mean, n_resamples=500, seed=1)
    b = bootstrap_ci(data, np.mean, n_resamples=500, seed=2)
    assert (a.low, a.high) != (b.low, b.high)
    assert abs(a.low - b.low) < 0.05


def test_paired_diff_detects_a_real_gap() -> None:
    """Judge A agrees with humans 90% of the time, judge B 60%. The paired
    interval on the difference must exclude zero."""
    rng = np.random.default_rng(9)
    n = 500
    human = rng.integers(0, 2, size=n)
    judge_a = np.where(rng.random(n) < 0.90, human, 1 - human)
    judge_b = np.where(rng.random(n) < 0.60, human, 1 - human)

    ci = paired_bootstrap_diff(
        (human, judge_a, judge_b),
        lambda h, a, _b: float(np.mean(h == a)),
        lambda h, _a, b: float(np.mean(h == b)),
        n_resamples=1500,
    )
    assert ci.estimate > 0.2
    assert ci.excludes(0.0)


def test_paired_diff_finds_nothing_when_there_is_nothing() -> None:
    rng = np.random.default_rng(10)
    n = 600
    human = rng.integers(0, 2, size=n)
    judge_a = np.where(rng.random(n) < 0.75, human, 1 - human)
    judge_b = np.where(rng.random(n) < 0.75, human, 1 - human)

    ci = paired_bootstrap_diff(
        (human, judge_a, judge_b),
        lambda h, a, _b: float(np.mean(h == a)),
        lambda h, _a, b: float(np.mean(h == b)),
        n_resamples=1500,
    )
    assert not ci.excludes(0.0)


def test_degenerate_statistic_does_not_emit_infinities() -> None:
    """Constant data makes BCa's bias correction undefined; it must degrade to a
    percentile interval instead of returning inf."""
    ci = bootstrap_ci(np.ones(50), np.mean, n_resamples=300)
    assert np.isfinite(ci.low) and np.isfinite(ci.high)
    assert ci.low == ci.high == 1.0


def test_ci_helpers() -> None:
    ci = CI(0.5, 0.4, 0.6, 0.95, "bca", 1000)
    assert ci.width == pytest.approx(0.2)
    assert ci.excludes(0.7) and not ci.excludes(0.45)
    assert "0.5000" in str(ci)


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci([1.0], np.mean)
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci(([1, 2, 3], [1, 2]), lambda a, b: 0.0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1, 2, 3], np.mean, confidence=1.5)
