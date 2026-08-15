"""Agreement coefficients checked against independent implementations.

Every coefficient here is verified either against scikit-learn (authoritative for
kappa) or against a deliberately naive reference implementation written a
different way in this file. A statistics library that only tests itself against
itself proves nothing, and a wrong kappa would silently invalidate every
downstream claim the harness makes.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from llm_eval.metrics.agreement import (
    agreement_report,
    cohens_kappa,
    confusion_matrix,
    gwets_ac1,
    krippendorff_alpha,
    raw_agreement,
)

# --------------------------------------------------------------------------
# Cohen's kappa vs scikit-learn
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(15))
def test_kappa_matches_sklearn_binary(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=200)
    b = np.where(rng.random(200) < 0.75, a, 1 - a)  # correlated but not identical
    assert cohens_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-12)


@pytest.mark.parametrize("seed", range(15))
def test_kappa_matches_sklearn_multiclass(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4, size=300)
    b = np.where(rng.random(300) < 0.6, a, rng.integers(0, 4, size=300))
    assert cohens_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-12)


@pytest.mark.parametrize("weighting", ["linear", "quadratic"])
@pytest.mark.parametrize("seed", range(8))
def test_weighted_kappa_matches_sklearn(weighting: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 5, size=250)
    b = np.clip(a + rng.integers(-1, 2, size=250), 0, 4)
    cats = [0, 1, 2, 3, 4]
    expected = cohen_kappa_score(a, b, weights=weighting, labels=cats)
    assert cohens_kappa(a, b, weighting=weighting, categories=cats) == pytest.approx(expected, abs=1e-12)


def test_kappa_hand_computed() -> None:
    """A 2x2 case small enough to check on paper.

    20 items: 10 both-yes, 5 both-no, 3 A-yes/B-no, 2 A-no/B-yes.
    po = 15/20 = 0.75
    A yes = 13/20, B yes = 12/20  ->  pe = .65*.60 + .35*.40 = .39 + .14 = .53
    kappa = (.75 - .53) / (1 - .53) = .22/.47
    """
    a = [1] * 13 + [0] * 7
    b = [1] * 10 + [0] * 3 + [1] * 2 + [0] * 5
    assert cohens_kappa(a, b) == pytest.approx(0.22 / 0.47, abs=1e-12)


def test_perfect_agreement_is_one() -> None:
    a = [0, 1, 2, 1, 0, 2, 1]
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    assert raw_agreement(a, a) == 1.0


def test_kappa_undefined_for_single_category() -> None:
    """Both raters say "fine" to everything. Raw agreement is 1.0 and kappa is
    genuinely undefined — it must not be reported as perfect."""
    a = b = [1] * 50
    assert np.isnan(cohens_kappa(a, b))
    assert raw_agreement(a, b) == 1.0


def test_systematic_disagreement_is_negative() -> None:
    a = [0, 0, 0, 1, 1, 1]
    b = [1, 1, 1, 0, 0, 0]
    assert cohens_kappa(a, b) < 0


# --------------------------------------------------------------------------
# The kappa paradox — the reason gwets_ac1 exists
# --------------------------------------------------------------------------

def test_kappa_paradox_high_agreement_low_kappa() -> None:
    """96% raw agreement on a skewed set, yet kappa lands near zero.

    This is the regime nearly every LLM eval set sits in: most outputs are fine,
    so one category dominates. Reporting kappa alone here would read as "the
    judge is useless" when the raters in fact almost always concur.
    """
    # 94 both-yes, 3 A-yes/B-no, 2 A-no/B-yes, 1 both-no
    a = [1] * 97 + [0] * 3
    b = [1] * 94 + [0] * 3 + [1] * 2 + [0] * 1

    assert raw_agreement(a, b) >= 0.94
    assert cohens_kappa(a, b) < 0.35
    assert gwets_ac1(a, b) > cohens_kappa(a, b)
    assert gwets_ac1(a, b) > 0.9


def test_ac1_equals_kappa_at_balanced_marginals() -> None:
    """With a 50/50 split the two chance corrections should be close, which is
    what makes the divergence above attributable to skew rather than to the
    coefficients simply measuring different things."""
    rng = np.random.default_rng(3)
    a = np.array([0, 1] * 150)
    b = np.where(rng.random(300) < 0.8, a, 1 - a)
    assert gwets_ac1(a, b) == pytest.approx(cohens_kappa(a, b), abs=0.05)


# --------------------------------------------------------------------------
# Krippendorff's alpha vs a brute-force reference
# --------------------------------------------------------------------------

def _alpha_reference(matrix, cats) -> float:
    """Naive O(units x raters^2) implementation, written directly from the
    definition with explicit loops and no matrix algebra. Slow and obviously
    correct — the point is that it shares no code path with the real one."""
    units = []
    for u in range(len(matrix[0])):
        vals = [row[u] for row in matrix if row[u] is not None]
        if len(vals) >= 2:
            units.append(vals)

    pairs: list[tuple] = []
    for vals in units:
        m = len(vals)
        for i in range(m):
            for j in range(m):
                if i != j:
                    pairs.append((vals[i], vals[j], 1.0 / (m - 1)))

    n_total = sum(w for _, _, w in pairs)
    d_obs = sum(w for c, k, w in pairs if c != k) / n_total

    counts = {c: 0.0 for c in cats}
    for c, _, w in pairs:
        counts[c] += w

    d_exp = 0.0
    for c in cats:
        for k in cats:
            if c != k:
                d_exp += counts[c] * counts[k]
    d_exp /= n_total * (n_total - 1)

    return 1.0 - d_obs / d_exp


def test_alpha_matches_reference_two_raters() -> None:
    matrix = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, 1, 3, 2],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 1, 3, 3],
    ]
    cats = [1, 2, 3, 4]
    assert krippendorff_alpha(matrix) == pytest.approx(_alpha_reference(matrix, cats), abs=1e-12)


def test_alpha_matches_reference_with_missing_data() -> None:
    """Three annotators where the third only labelled a subset — exactly the
    shape of a real annotation budget."""
    matrix = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    ]
    cats = [1, 2, 3, 4, 5]
    assert krippendorff_alpha(matrix) == pytest.approx(_alpha_reference(matrix, cats), abs=1e-12)


def test_alpha_perfect_agreement() -> None:
    matrix = [[1, 2, 3, 1, 2], [1, 2, 3, 1, 2]]
    assert krippendorff_alpha(matrix) == pytest.approx(1.0)


def test_alpha_near_zero_for_independent_raters() -> None:
    rng = np.random.default_rng(11)
    n = 4000
    matrix = [rng.integers(0, 3, size=n).tolist(), rng.integers(0, 3, size=n).tolist()]
    assert abs(krippendorff_alpha(matrix)) < 0.05


def test_alpha_drops_units_rated_once() -> None:
    """A unit only one rater touched carries no pairing information."""
    paired = [[1, 2, 3], [1, 2, 3]]
    padded = [[1, 2, 3, 1, 2], [1, 2, 3, None, None]]
    assert krippendorff_alpha(padded) == pytest.approx(krippendorff_alpha(paired))


def test_alpha_rejects_all_singleton_units() -> None:
    with pytest.raises(ValueError, match="at least two raters"):
        krippendorff_alpha([[1, 2, None], [None, None, 3]])


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def test_confusion_matrix_counts() -> None:
    m = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])
    assert m.tolist() == [[1.0, 1.0], [0.0, 2.0]]
    assert m.sum() == 4


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([1, 2, 3], [1, 2])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="zero items"):
        raw_agreement([], [])


def test_report_flags_skew() -> None:
    rep = agreement_report([1] * 95 + [0] * 5, [1] * 93 + [0] * 2 + [1] * 2 + [0] * 3)
    assert rep["n"] == 100
    assert rep["majority_class_share"] == pytest.approx(0.95)
    assert rep["gwets_ac1"] > rep["cohens_kappa"]
