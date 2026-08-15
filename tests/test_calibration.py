"""Calibration metrics, cross-checked against scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from llm_eval.metrics.calibration import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    reliability_curve,
)


@pytest.mark.parametrize("seed", range(15))
def test_brier_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = rng.random(300)
    y = rng.binomial(1, p)
    assert brier_score(y, p) == pytest.approx(brier_score_loss(y, p), abs=1e-12)


def test_brier_hand_computed() -> None:
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.8, 0.4]
    # (.01 + .01 + .04 + .16) / 4
    assert brier_score(y, p) == pytest.approx(0.055)


def test_brier_of_always_half_is_quarter() -> None:
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, size=1000)
    assert brier_score(y, np.full(1000, 0.5)) == pytest.approx(0.25)


def test_perfect_prediction_scores_zero() -> None:
    y = [1, 0, 1, 1, 0]
    assert brier_score(y, y) == 0.0


def test_reliability_curve_matches_sklearn() -> None:
    rng = np.random.default_rng(3)
    p = rng.random(2000)
    y = rng.binomial(1, p)

    ours = reliability_curve(y, p, n_bins=10, strategy="uniform")
    sk_true, sk_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")

    assert np.allclose([b["observed_rate"] for b in ours["bins"]], sk_true, atol=1e-12)
    assert np.allclose([b["mean_predicted"] for b in ours["bins"]], sk_pred, atol=1e-12)


def test_well_calibrated_probabilities_have_low_ece() -> None:
    """y drawn with probability exactly p, so the judge is calibrated by
    construction and ECE should be small."""
    rng = np.random.default_rng(4)
    p = rng.random(20_000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_overconfident_probabilities_have_high_ece() -> None:
    """A judge that always says 0.99 but is right 60% of the time."""
    rng = np.random.default_rng(5)
    n = 2000
    y = rng.binomial(1, 0.6, size=n)
    p = np.full(n, 0.99)
    assert expected_calibration_error(y, p) > 0.3
    assert brier_score(y, p) > 0.25  # worse than always saying 0.5


def test_empty_bins_are_dropped_not_zero_filled() -> None:
    """Judge confidence piles up at the extremes, leaving the middle empty. Those
    bins are missing evidence and must not be counted as perfectly calibrated."""
    y = [1] * 50 + [0] * 50
    p = [0.95] * 50 + [0.05] * 50
    curve = reliability_curve(y, p, n_bins=10, strategy="uniform")
    assert len(curve["bins"]) == 2
    assert all(b["count"] > 0 for b in curve["bins"])


def test_quantile_binning_spreads_the_mass() -> None:
    rng = np.random.default_rng(6)
    p = np.concatenate([rng.uniform(0, 0.05, 500), rng.uniform(0.95, 1.0, 500)])
    y = rng.binomial(1, p)

    uniform = reliability_curve(y, p, n_bins=10, strategy="uniform")
    quantile = reliability_curve(y, p, n_bins=10, strategy="quantile")
    assert len(quantile["bins"]) > len(uniform["bins"])


def test_report_carries_both_binnings() -> None:
    rng = np.random.default_rng(8)
    p = rng.random(500)
    y = rng.binomial(1, p)
    rep = calibration_report(y, p)
    assert rep["n"] == 500
    assert rep["brier"] < rep["brier_baseline_always_half"]
    assert "ece_uniform" in rep and "ece_quantile" in rep


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="binary"):
        brier_score([0, 1, 2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([0, 1], [0.5, 1.4])
    with pytest.raises(ValueError, match="shape mismatch"):
        brier_score([0, 1], [0.5])
    with pytest.raises(ValueError, match="zero items"):
        brier_score([], [])
