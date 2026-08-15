"""Multiple-comparison correction, cross-checked against statsmodels."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from llm_eval.metrics.correction import benjamini_hochberg, bonferroni, correct, holm


@pytest.mark.parametrize("seed", range(20))
def test_holm_matches_statsmodels(seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = rng.random(rng.integers(2, 30))
    _, expected, _, _ = multipletests(p, alpha=0.05, method="holm")
    assert np.allclose(holm(p).p_adjusted, expected, atol=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_bh_matches_statsmodels(seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = rng.random(rng.integers(2, 30))
    _, expected, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    assert np.allclose(benjamini_hochberg(p).p_adjusted, expected, atol=1e-12)


def test_holm_hand_computed() -> None:
    """m=5, p = .01 .02 .03 .04 .05
    step-down multipliers 5,4,3,2,1 give .05 .08 .09 .08 .05, and the
    monotonicity constraint drags the last two up to .09."""
    res = holm([0.01, 0.02, 0.03, 0.04, 0.05])
    assert res.p_adjusted == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])
    assert res.rejected == [True, False, False, False, False]


def test_bh_hand_computed() -> None:
    """Same p-values: m*p/rank is .05 for every one of them."""
    res = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert res.p_adjusted == pytest.approx([0.05] * 5)
    assert all(res.rejected)


def test_adjusted_p_is_monotone() -> None:
    """Adjusted p must not decrease as raw p increases — the property that makes
    the correction interpretable at all."""
    rng = np.random.default_rng(7)
    for method in ("holm", "bh", "bonferroni"):
        p = np.sort(rng.random(40))
        adj = np.asarray(correct(p, method=method).p_adjusted)
        assert np.all(np.diff(adj) >= -1e-12), method


def test_holm_dominates_bonferroni() -> None:
    """Holm rejects everything Bonferroni does, and sometimes more."""
    rng = np.random.default_rng(5)
    p = rng.random(50) ** 3
    h, b = holm(p), bonferroni(p)
    assert np.all(np.asarray(h.p_adjusted) <= np.asarray(b.p_adjusted) + 1e-12)
    assert h.n_rejected >= b.n_rejected


def test_correction_kills_the_free_finding() -> None:
    """The scenario this module exists to prevent.

    Twenty probes against a judge that has no real bias. One comes back at
    p=0.03 purely by chance. Uncorrected it is a "finding"; after Holm it is
    correctly nothing.
    """
    p = [0.03] + [0.2 + 0.03 * i for i in range(19)]
    labels = [f"probe_{i}" for i in range(20)]
    assert sum(x <= 0.05 for x in p) == 1
    res = holm(p, labels=labels)
    assert res.n_rejected == 0
    assert res.significant() == []


def test_fwer_is_actually_controlled() -> None:
    """Simulation under a complete null: the family-wise error rate must sit at
    or below alpha. This is the guarantee, so it gets checked rather than
    assumed."""
    rng = np.random.default_rng(19)
    trials, m, alpha = 2000, 10, 0.05
    family_errors = 0
    for _ in range(trials):
        p = rng.random(m)  # all null -> uniform
        if holm(p, alpha=alpha).n_rejected > 0:
            family_errors += 1
    assert family_errors / trials <= alpha + 0.015


def test_bh_is_more_powerful_than_holm() -> None:
    rng = np.random.default_rng(23)
    p = np.concatenate([rng.random(30) * 0.02, rng.random(30)])
    assert benjamini_hochberg(p).n_rejected >= holm(p).n_rejected


def test_labels_and_table() -> None:
    res = correct([0.001, 0.5], method="holm", labels=["position_bias", "verbosity_bias"])
    assert res.significant() == ["position_bias"]
    assert "position_bias" in res.table()
    assert "holm" in res.table()


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm([0.5, 1.5])
    with pytest.raises(ValueError, match="no p-values"):
        holm([])
    with pytest.raises(ValueError, match="same length"):
        holm([0.1, 0.2], labels=["only_one"])
    with pytest.raises(ValueError, match="unknown method"):
        correct([0.1], method="fdr_by")
