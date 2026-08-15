"""Power and minimum detectable effect.

The question this module exists to answer: *your eval score moved from 8.2 to
8.0 — is that a regression?* Without an MDE the honest answer is "this eval set
cannot tell you", and most teams never compute it, so they ship on noise in both
directions.

Use ``mde_two_proportions`` before building an eval set to find out how big it
has to be, and after a null result to state what the run could and could not
have detected. A null result without an MDE is not evidence of no change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import brentq

__all__ = ["PowerResult", "mde_two_proportions", "n_for_power", "power_two_proportions"]


@dataclass(frozen=True)
class PowerResult:
    n_per_group: int
    baseline_rate: float
    effect: float
    power: float
    alpha: float

    def __str__(self) -> str:
        return (
            f"n={self.n_per_group}/group, baseline={self.baseline_rate:.3f}, "
            f"effect={self.effect:+.4f}, power={self.power:.3f} (alpha={self.alpha})"
        )


def _check_rate(p: float, name: str) -> None:
    if not 0 < p < 1:
        raise ValueError(f"{name} must lie strictly in (0, 1), got {p}")


def power_two_proportions(
    n_per_group: int,
    baseline_rate: float,
    effect: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power of a two-proportion z-test to detect an absolute change ``effect``.

    Normal approximation with unpooled variance under the alternative. Accurate
    for the eval-set sizes this harness targets (hundreds to thousands); at
    n < 30 per group with a rate near 0 or 1, prefer an exact test.
    """
    _check_rate(baseline_rate, "baseline_rate")
    treated = baseline_rate + effect
    _check_rate(treated, "baseline_rate + effect")
    if n_per_group < 2:
        raise ValueError("n_per_group must be at least 2")

    z_crit = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)

    pooled = (baseline_rate + treated) / 2
    se_null = np.sqrt(2 * pooled * (1 - pooled) / n_per_group)
    se_alt = np.sqrt(
        baseline_rate * (1 - baseline_rate) / n_per_group
        + treated * (1 - treated) / n_per_group
    )
    if se_alt == 0:
        return 1.0

    shift = abs(effect) / se_alt
    crit = z_crit * se_null / se_alt
    power = float(stats.norm.sf(crit - shift))
    if two_sided:
        power += float(stats.norm.cdf(-crit - shift))
    return min(max(power, 0.0), 1.0)


def n_for_power(
    baseline_rate: float,
    effect: float,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
    max_n: int = 10_000_000,
) -> int:
    """Smallest per-group n reaching ``power`` for the given effect."""
    if not 0 < power < 1:
        raise ValueError("power must lie in (0, 1)")
    if effect == 0:
        raise ValueError("cannot power a zero effect")

    lo, hi = 2, 1024
    while hi < max_n and power_two_proportions(hi, baseline_rate, effect, alpha, two_sided) < power:
        lo, hi = hi, hi * 2
    if hi >= max_n:
        raise ValueError(f"required n exceeds max_n={max_n}; effect may be too small")

    while lo < hi:
        mid = (lo + hi) // 2
        if power_two_proportions(mid, baseline_rate, effect, alpha, two_sided) < power:
            lo = mid + 1
        else:
            hi = mid
    return lo


def mde_two_proportions(
    n_per_group: int,
    baseline_rate: float,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
    direction: str = "decrease",
) -> float:
    """Smallest absolute change this eval set can detect at the given power.

    ``direction="decrease"`` is the default because the question that matters in
    CI is *would we have caught a regression*, and the two directions are not
    symmetric when the baseline is near a boundary.

    Returns a signed effect.
    """
    _check_rate(baseline_rate, "baseline_rate")
    sign = -1.0 if direction == "decrease" else 1.0
    room = baseline_rate if sign < 0 else 1.0 - baseline_rate
    if room <= 0:
        raise ValueError(f"no room to {direction} from baseline {baseline_rate}")

    def deficit(mag: float) -> float:
        return power_two_proportions(n_per_group, baseline_rate, sign * mag, alpha, two_sided) - power

    lo, hi = 1e-9, room * (1 - 1e-9)
    if deficit(hi) < 0:
        return float("nan")  # even the largest possible effect is underpowered
    return float(sign * brentq(deficit, lo, hi, xtol=1e-7))


def power_report(n_per_group: int, baseline_rate: float, alpha: float = 0.05) -> dict:
    """What an eval set of this size can and cannot see."""
    mde = mde_two_proportions(n_per_group, baseline_rate, alpha=alpha)
    return {
        "n_per_group": n_per_group,
        "baseline_rate": baseline_rate,
        "alpha": alpha,
        "mde_at_80_power": mde,
        "mde_pct_points": None if np.isnan(mde) else round(mde * 100, 2),
        "n_needed_for_1pp": n_for_power(baseline_rate, -0.01, alpha=alpha),
        "n_needed_for_5pp": n_for_power(baseline_rate, -0.05, alpha=alpha),
    }
