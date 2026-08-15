"""Power and MDE, cross-checked against statsmodels."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from llm_eval.metrics.power import (
    mde_two_proportions,
    n_for_power,
    power_report,
    power_two_proportions,
)


@pytest.mark.parametrize(
    "baseline,effect,n",
    [
        (0.50, 0.10, 400), (0.50, 0.05, 1000), (0.80, -0.10, 200),
        (0.30, 0.08, 500), (0.90, -0.05, 800), (0.65, 0.12, 150),
    ],
)
def test_power_close_to_statsmodels(baseline: float, effect: float, n: int) -> None:
    """statsmodels parameterises by Cohen's h rather than raw proportions, so the
    two agree closely but not identically. A 0.05 tolerance confirms the same
    calculation without pretending the parameterisations are the same."""
    expected = NormalIndPower().power(
        effect_size=proportion_effectsize(baseline + effect, baseline),
        nobs1=n, alpha=0.05, ratio=1.0, alternative="two-sided",
    )
    assert power_two_proportions(n, baseline, effect) == pytest.approx(expected, abs=0.05)


def test_power_rises_with_n() -> None:
    powers = [power_two_proportions(n, 0.5, 0.05) for n in (100, 400, 1600, 6400)]
    assert powers == sorted(powers)
    assert powers[0] < 0.3 < powers[-1]


def test_power_rises_with_effect() -> None:
    powers = [power_two_proportions(500, 0.5, e) for e in (0.01, 0.03, 0.05, 0.10)]
    assert powers == sorted(powers)


def test_power_at_zero_effect_is_alpha() -> None:
    """With no true effect the rejection rate is exactly the false-positive
    rate — the sanity check that the whole calculation is anchored correctly."""
    assert power_two_proportions(1000, 0.5, 0.0, alpha=0.05) == pytest.approx(0.05, abs=0.01)


def test_n_for_power_round_trips() -> None:
    for baseline, effect in [(0.5, -0.05), (0.8, -0.10), (0.3, 0.07)]:
        n = n_for_power(baseline, effect, power=0.8)
        assert power_two_proportions(n, baseline, effect) >= 0.8
        assert power_two_proportions(n - 1, baseline, effect) < 0.8


def test_mde_round_trips() -> None:
    for n, baseline in [(500, 0.5), (2000, 0.8), (300, 0.35)]:
        mde = mde_two_proportions(n, baseline, power=0.8)
        assert mde < 0  # default direction is "decrease"
        assert power_two_proportions(n, baseline, mde) == pytest.approx(0.8, abs=0.01)


def test_mde_shrinks_as_the_eval_set_grows() -> None:
    mdes = [abs(mde_two_proportions(n, 0.7)) for n in (100, 500, 2500, 12500)]
    assert mdes == sorted(mdes, reverse=True)


def test_the_question_this_module_exists_for() -> None:
    """"Our eval score dropped 2 points — is that a regression?"

    On a 200-item set at an 80% baseline the smallest detectable drop is around
    11 points, so a 2-point move is invisible and the honest answer is "this eval
    set cannot tell you". Resolving 2 points needs a set in the thousands — a
    number worth knowing *before* committing to an annotation budget.
    """
    small_mde = abs(mde_two_proportions(200, 0.80))
    large_mde = abs(mde_two_proportions(10_000, 0.80))
    assert small_mde > 0.05, "a 200-item set should not resolve even 5pp"
    assert large_mde < 0.02, "a 10k-item set should resolve 2pp"

    needed = n_for_power(0.80, -0.02, power=0.8)
    assert 4_000 < needed < 9_000, f"2pp at 80% baseline needs ~6k items, got {needed}"


def test_mde_returns_nan_when_hopeless() -> None:
    """A tiny eval set cannot reach 80% power at any effect size; that must
    surface as NaN rather than a fabricated number."""
    assert np.isnan(mde_two_proportions(3, 0.5, power=0.99))


def test_report_shape() -> None:
    rep = power_report(400, 0.75)
    assert rep["n_per_group"] == 400
    assert rep["mde_at_80_power"] < 0
    assert rep["n_needed_for_1pp"] > rep["n_needed_for_5pp"]


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="strictly in"):
        power_two_proportions(100, 0.0, 0.1)
    with pytest.raises(ValueError, match="strictly in"):
        power_two_proportions(100, 0.95, 0.10)  # pushes the rate to 1.05
    with pytest.raises(ValueError, match="at least 2"):
        power_two_proportions(1, 0.5, 0.1)
    with pytest.raises(ValueError, match="zero effect"):
        n_for_power(0.5, 0.0)
