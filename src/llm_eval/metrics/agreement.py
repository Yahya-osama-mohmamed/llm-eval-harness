"""Inter-rater agreement.

The harness exists to ask whether a judge agrees with humans. That question only
has an answer once you can also say how much two *humans* agree, so everything
here is written to work identically for a (judge, human) pair and a
(human, human) pair.

Three coefficients, because no single one is safe alone:

* ``cohens_kappa``   — the default everyone reports.
* ``gwets_ac1``      — reported alongside it because kappa collapses toward zero
  when one category dominates, even at 95% raw agreement. Most LLM eval sets are
  heavily imbalanced (most outputs are fine), so this is not a hypothetical.
* ``krippendorff_alpha`` — handles >2 raters, missing labels and ordinal scales,
  none of which kappa does.

Disagreeing coefficients are a finding about the label distribution, not a bug.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np

__all__ = [
    "cohens_kappa",
    "confusion_matrix",
    "gwets_ac1",
    "krippendorff_alpha",
    "raw_agreement",
]

Weighting = Literal["unweighted", "linear", "quadratic"]


def _as_pair(a: Sequence, b: Sequence) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape:
        raise ValueError(f"rater arrays must be the same length, got {x.shape} and {y.shape}")
    if x.size == 0:
        raise ValueError("cannot compute agreement on zero items")
    return x, y


def _categories(*arrays: np.ndarray) -> list:
    seen: set = set()
    for arr in arrays:
        seen.update(arr.tolist())
    try:
        return sorted(seen)
    except TypeError:  # mixed/unorderable label types
        return sorted(seen, key=repr)


def _weight_matrix(k: int, weighting: Weighting) -> np.ndarray:
    """Disagreement weights: 0 on the diagonal, 1 at maximal disagreement."""
    if weighting == "unweighted":
        return 1.0 - np.eye(k)
    idx = np.arange(k)
    diff = np.abs(idx[:, None] - idx[None, :]).astype(float)
    if k == 1:
        return np.zeros((1, 1))
    if weighting == "linear":
        return diff / (k - 1)
    if weighting == "quadratic":
        return (diff / (k - 1)) ** 2
    raise ValueError(f"unknown weighting {weighting!r}")


def confusion_matrix(a: Sequence, b: Sequence, categories: list | None = None) -> np.ndarray:
    """Counts of (rater A label, rater B label) pairs."""
    x, y = _as_pair(a, b)
    cats = categories if categories is not None else _categories(x, y)
    pos = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)), dtype=float)
    for u, v in zip(x.tolist(), y.tolist(), strict=True):
        m[pos[u], pos[v]] += 1
    return m


def raw_agreement(a: Sequence, b: Sequence) -> float:
    """Fraction of items the two raters labelled identically."""
    x, y = _as_pair(a, b)
    return float(np.mean(x == y))


def cohens_kappa(
    a: Sequence,
    b: Sequence,
    weighting: Weighting = "unweighted",
    categories: list | None = None,
) -> float:
    """Cohen's kappa for two raters.

    ``weighting`` treats labels as ordinal, penalising a 1-vs-5 disagreement more
    than 4-vs-5. Requires ``categories`` to be in a meaningful order — which is
    why the parameter is exposed rather than always inferred.

    Returns NaN when expected agreement is exactly 1 (every rating identical and
    only one category present); kappa is genuinely undefined there rather than
    perfect, and returning NaN keeps that from being silently reported as 1.0.
    """
    x, y = _as_pair(a, b)
    cats = categories if categories is not None else _categories(x, y)
    m = confusion_matrix(x, y, cats)
    n = m.sum()
    w = _weight_matrix(len(cats), weighting)

    observed = m / n
    row, col = observed.sum(axis=1), observed.sum(axis=0)
    expected = np.outer(row, col)

    d_obs = float((w * observed).sum())
    d_exp = float((w * expected).sum())
    if d_exp == 0:
        return float("nan")
    return 1.0 - d_obs / d_exp


def gwets_ac1(a: Sequence, b: Sequence, categories: list | None = None) -> float:
    """Gwet's AC1 — kappa's chance-correction, done differently.

    Kappa estimates chance agreement from the marginals, which inflates the
    correction exactly when one category dominates and drives kappa toward zero
    despite high raw agreement (the "kappa paradox"). AC1 uses a prevalence
    estimate that does not have that failure mode. Report both; a large gap
    between them means the label distribution is skewed enough that kappa alone
    would mislead.
    """
    x, y = _as_pair(a, b)
    cats = categories if categories is not None else _categories(x, y)
    q = len(cats)
    if q == 1:
        return float("nan")

    m = confusion_matrix(x, y, cats)
    n = m.sum()
    p_obs = float(np.trace(m) / n)

    # mean prevalence of each category across the two raters
    pi = (m.sum(axis=1) + m.sum(axis=0)) / (2.0 * n)
    p_exp = float((pi * (1.0 - pi)).sum() / (q - 1))
    if p_exp == 1:
        return float("nan")
    return (p_obs - p_exp) / (1.0 - p_exp)


def krippendorff_alpha(
    reliability_matrix: Sequence[Sequence],
    level: Literal["nominal", "ordinal", "interval"] = "nominal",
    categories: list | None = None,
) -> float:
    """Krippendorff's alpha.

    ``reliability_matrix`` is raters x units; use ``None`` (or ``np.nan``) for a
    unit a rater did not label. Unlike kappa this handles any number of raters
    and missing data, which is what you need once a third annotator only labels
    a subset.

    Units rated by fewer than two raters carry no pairing information and are
    dropped, per the standard definition.
    """
    rows = [list(r) for r in reliability_matrix]
    if not rows:
        raise ValueError("empty reliability matrix")
    n_units = len(rows[0])
    if any(len(r) != n_units for r in rows):
        raise ValueError("every rater must have one entry per unit (use None for missing)")

    def missing(v) -> bool:
        if v is None:
            return True
        return isinstance(v, float) and np.isnan(v)

    units = [[r[u] for r in rows if not missing(r[u])] for u in range(n_units)]
    units = [vals for vals in units if len(vals) >= 2]
    if not units:
        raise ValueError("no unit was rated by at least two raters")

    cats = categories if categories is not None else _categories(
        np.asarray([v for vals in units for v in vals])
    )
    pos = {c: i for i, c in enumerate(cats)}
    q = len(cats)

    # coincidence matrix
    o = np.zeros((q, q), dtype=float)
    for vals in units:
        mu = len(vals)
        for i, ci in enumerate(vals):
            for j, cj in enumerate(vals):
                if i != j:
                    o[pos[ci], pos[cj]] += 1.0 / (mu - 1)

    n_total = o.sum()
    if n_total == 0:
        return float("nan")

    delta = _difference_matrix(cats, level)
    n_c = o.sum(axis=1)

    d_obs = float((delta * o).sum())
    expected = np.outer(n_c, n_c) - np.diag(n_c)
    d_exp = float((delta * expected).sum()) / (n_total - 1)

    if d_exp == 0:
        return float("nan")
    return 1.0 - d_obs / d_exp


def _difference_matrix(cats: list, level: str) -> np.ndarray:
    q = len(cats)
    if level == "nominal":
        return 1.0 - np.eye(q)
    if level == "interval":
        vals = np.asarray([float(c) for c in cats])
        return (vals[:, None] - vals[None, :]) ** 2
    if level == "ordinal":
        # ordinal distance depends on the marginals, which the caller does not
        # have here; the rank-based form is the standard stand-in.
        ranks = np.arange(q, dtype=float)
        return (ranks[:, None] - ranks[None, :]) ** 2
    raise ValueError(f"unknown level {level!r}")


def agreement_report(a: Sequence, b: Sequence, categories: list | None = None) -> dict:
    """All three coefficients plus raw agreement, for one rater pair."""
    counts = Counter(np.asarray(a).tolist())
    total = sum(counts.values())
    return {
        "n": total,
        "raw_agreement": raw_agreement(a, b),
        "cohens_kappa": cohens_kappa(a, b, categories=categories),
        "gwets_ac1": gwets_ac1(a, b, categories=categories),
        "majority_class_share": max(counts.values()) / total,
    }
