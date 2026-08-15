"""Multiple-comparison correction.

An eval report compares several judges over several slices and runs several bias
probes. Twenty comparisons at alpha=0.05 buy you one significant result for
free. Reporting those uncorrected is the most common way an eval dashboard
manufactures findings.

Holm controls the family-wise error rate — use it for the small, decision-bearing
family (does this judge beat that one, did this probe fire). Benjamini-Hochberg
controls the false discovery rate — use it for wide exploratory sweeps like
per-slice screening, where insisting on FWER leaves you unable to detect
anything.

Both are uniformly more powerful than Bonferroni, which is here only as the
conservative reference people ask for by name.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["CorrectionResult", "benjamini_hochberg", "bonferroni", "correct", "holm"]

Method = Literal["holm", "bh", "bonferroni"]


@dataclass(frozen=True)
class CorrectionResult:
    labels: list[str]
    p_raw: list[float]
    p_adjusted: list[float]
    rejected: list[bool]
    alpha: float
    method: str

    @property
    def n_rejected(self) -> int:
        return sum(self.rejected)

    def significant(self) -> list[str]:
        return [lab for lab, rej in zip(self.labels, self.rejected, strict=True) if rej]

    def table(self) -> str:
        head = f"{'comparison':<40} {'p_raw':>10} {'p_adj':>10}  significant"
        lines = [head, "-" * len(head)]
        order = np.argsort(self.p_raw)
        for i in order:
            mark = "yes" if self.rejected[i] else "no"
            lines.append(
                f"{self.labels[i]:<40} {self.p_raw[i]:>10.4g} {self.p_adjusted[i]:>10.4g}  {mark}"
            )
        lines.append(f"\n{self.method}, alpha={self.alpha}: {self.n_rejected}/{len(self.labels)} significant")
        return "\n".join(lines)


def _prep(p_values: Sequence[float], labels: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        raise ValueError("no p-values given")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must lie in [0, 1]")
    if labels is None:
        labs = [f"comparison_{i}" for i in range(p.size)]
    else:
        labs = list(labels)
        if len(labs) != p.size:
            raise ValueError("labels and p_values must be the same length")
    return p, labs


def holm(p_values: Sequence[float], alpha: float = 0.05, labels: Sequence[str] | None = None) -> CorrectionResult:
    """Holm-Bonferroni step-down. Controls FWER with no independence assumption."""
    p, labs = _prep(p_values, labels)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)

    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)  # enforce monotonicity
        adj[i] = min(running, 1.0)

    return CorrectionResult(labs, p.tolist(), adj.tolist(), (adj <= alpha).tolist(), alpha, "holm")


def benjamini_hochberg(
    p_values: Sequence[float], alpha: float = 0.05, labels: Sequence[str] | None = None
) -> CorrectionResult:
    """Benjamini-Hochberg step-up. Controls FDR under independence or positive
    dependence — which covers per-slice screening on a shared eval set."""
    p, labs = _prep(p_values, labels)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)

    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = m * p[i] / (rank + 1)
        running = min(running, val)  # enforce monotonicity
        adj[i] = min(running, 1.0)

    return CorrectionResult(labs, p.tolist(), adj.tolist(), (adj <= alpha).tolist(), alpha, "benjamini-hochberg")


def bonferroni(
    p_values: Sequence[float], alpha: float = 0.05, labels: Sequence[str] | None = None
) -> CorrectionResult:
    """Single-step Bonferroni. Always dominated by Holm; kept for reference."""
    p, labs = _prep(p_values, labels)
    adj = np.minimum(p * p.size, 1.0)
    return CorrectionResult(labs, p.tolist(), adj.tolist(), (adj <= alpha).tolist(), alpha, "bonferroni")


def correct(
    p_values: Sequence[float],
    method: Method = "holm",
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> CorrectionResult:
    fns = {"holm": holm, "bh": benjamini_hochberg, "bonferroni": bonferroni}
    if method not in fns:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(fns)}")
    return fns[method](p_values, alpha=alpha, labels=labels)
