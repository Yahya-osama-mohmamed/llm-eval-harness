"""Calibration of a judge's confidence.

A judge that says "90% confident" should be right about 90% of the time. Most
judge pipelines never check, which is how "the judge was very confident" ends up
in a decision document meaning nothing.

ECE is reported both ways deliberately. Equal-width bins are the convention;
they also produce near-empty bins in the middle when judge confidence piles up
near 0 and 1, which is exactly what LLM judges do. Equal-mass bins avoid that.
When the two disagree, the binning is doing the talking, not the model — so the
report shows both rather than picking a winner.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

__all__ = ["brier_score", "expected_calibration_error", "reliability_curve"]

Strategy = Literal["uniform", "quantile"]


def _check(y_true: Sequence, y_prob: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("cannot calibrate on zero items")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_true must be binary 0/1")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("y_prob must lie in [0, 1]")
    return y, p


def brier_score(y_true: Sequence, y_prob: Sequence) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 is the
    score of always saying 0.5, which is the bar a judge must clear to be worth
    anything at all."""
    y, p = _check(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def _bin_edges(p: np.ndarray, n_bins: int, strategy: Strategy) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    raise ValueError(f"unknown strategy {strategy!r}")


def reliability_curve(
    y_true: Sequence,
    y_prob: Sequence,
    n_bins: int = 10,
    strategy: Strategy = "uniform",
) -> dict:
    """Observed frequency against predicted probability, per bin.

    Empty bins are dropped rather than reported as zero — a bin nobody landed in
    is missing evidence, not evidence of miscalibration.
    """
    y, p = _check(y_true, y_prob)
    edges = _bin_edges(p, n_bins, strategy)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": count,
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return {"strategy": strategy, "n_bins_requested": n_bins, "bins": rows}


def expected_calibration_error(
    y_true: Sequence,
    y_prob: Sequence,
    n_bins: int = 10,
    strategy: Strategy = "uniform",
) -> float:
    """Count-weighted mean gap between predicted probability and observed rate."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    total = sum(b["count"] for b in curve["bins"])
    if total == 0:
        return float("nan")
    return float(
        sum(b["count"] * abs(b["mean_predicted"] - b["observed_rate"]) for b in curve["bins"])
        / total
    )


def calibration_report(y_true: Sequence, y_prob: Sequence, n_bins: int = 10) -> dict:
    """Brier plus ECE under both binning strategies."""
    return {
        "n": int(np.asarray(y_true).size),
        "brier": brier_score(y_true, y_prob),
        "brier_baseline_always_half": 0.25,
        "ece_uniform": expected_calibration_error(y_true, y_prob, n_bins, "uniform"),
        "ece_quantile": expected_calibration_error(y_true, y_prob, n_bins, "quantile"),
    }
