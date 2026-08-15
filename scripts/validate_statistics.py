"""Produce the numbers quoted in RESULTS.md section 2.

The harness claims its intervals are trustworthy. This script is the evidence:
it generates data whose true answer is known, runs the machinery many times, and
counts how often it was right. Everything reported here is measured by running
this file — nothing is asserted from theory.

    python scripts/validate_statistics.py

Runtime is a couple of minutes; it is deliberately not part of the CI suite,
which runs smaller versions of the same checks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from llm_eval.metrics.agreement import cohens_kappa
from llm_eval.metrics.correction import benjamini_hochberg, holm
from llm_eval.metrics.power import mde_two_proportions, n_for_power
from llm_eval.metrics.resample import bootstrap_ci

OUT = Path(__file__).resolve().parents[1] / "reports" / "statistical_validation.json"


def coverage_of_mean(method: str, trials: int = 1000, n: int = 120, truth: float = 0.3) -> float:
    rng = np.random.default_rng(42)
    covered = 0
    for t in range(trials):
        sample = rng.binomial(1, truth, size=n)
        ci = bootstrap_ci(sample, np.mean, n_resamples=1000, method=method, seed=t)
        covered += ci.low <= truth <= ci.high
    return covered / trials


def coverage_of_kappa(method: str, trials: int = 500, n: int = 150, p_agree: float = 0.85) -> tuple[float, float]:
    rng = np.random.default_rng(7)
    big_a = rng.integers(0, 2, size=500_000)
    big_b = np.where(rng.random(500_000) < p_agree, big_a, 1 - big_a)
    truth = cohens_kappa(big_a, big_b)

    covered = 0
    for t in range(trials):
        a = rng.integers(0, 2, size=n)
        b = np.where(rng.random(n) < p_agree, a, 1 - a)
        ci = bootstrap_ci((a, b), cohens_kappa, n_resamples=600, method=method, seed=t)
        covered += ci.low <= truth <= ci.high
    return covered / trials, truth


def error_rates_under_the_null(trials: int = 5000, m: int = 20, alpha: float = 0.05) -> dict:
    """All m hypotheses are null. Count how often each policy reports at least
    one 'finding'. Uncorrected should fail about 64% of the time at m=20 —
    which is the entire argument for correcting."""
    rng = np.random.default_rng(19)
    uncorrected = holm_errors = bh_errors = 0
    for _ in range(trials):
        p = rng.random(m)
        uncorrected += int(np.any(p <= alpha))
        holm_errors += int(holm(p, alpha=alpha).n_rejected > 0)
        bh_errors += int(benjamini_hochberg(p, alpha=alpha).n_rejected > 0)
    return {
        "m_hypotheses": m,
        "alpha": alpha,
        "trials": trials,
        "familywise_error_uncorrected": uncorrected / trials,
        "familywise_error_holm": holm_errors / trials,
        "familywise_error_bh": bh_errors / trials,
        "theoretical_uncorrected": 1 - (1 - alpha) ** m,
    }


def eval_set_sizing() -> dict:
    """What an eval set of each size can actually resolve at 80% baseline."""
    rows = {}
    for n in (100, 200, 500, 1000, 2000, 5000, 10000):
        mde = mde_two_proportions(n, 0.80, power=0.8)
        rows[str(n)] = None if np.isnan(mde) else round(abs(mde) * 100, 2)
    return {
        "baseline_rate": 0.80,
        "power": 0.8,
        "alpha": 0.05,
        "mde_percentage_points_by_n": rows,
        "n_needed_for_1pp": n_for_power(0.80, -0.01),
        "n_needed_for_2pp": n_for_power(0.80, -0.02),
        "n_needed_for_5pp": n_for_power(0.80, -0.05),
    }


def main() -> None:
    t0 = time.time()
    results: dict = {"generated_by": "scripts/validate_statistics.py"}

    print("bootstrap coverage (nominal 95%)...")
    results["coverage_mean"] = {
        m: coverage_of_mean(m) for m in ("percentile", "bca")
    }
    for m, v in results["coverage_mean"].items():
        print(f"  mean, {m:<10} {v:.3f}")

    for m in ("percentile", "bca"):
        rate, truth = coverage_of_kappa(m)
        results.setdefault("coverage_kappa", {})[m] = rate
        results["true_kappa"] = truth
        print(f"  kappa, {m:<9} {rate:.3f}   (true kappa {truth:.4f})")

    print("multiple-comparison error rates under a complete null...")
    results["null_error_rates"] = error_rates_under_the_null()
    r = results["null_error_rates"]
    print(f"  uncorrected {r['familywise_error_uncorrected']:.3f}  "
          f"holm {r['familywise_error_holm']:.3f}  bh {r['familywise_error_bh']:.3f}")

    print("eval-set sizing...")
    results["sizing"] = eval_set_sizing()
    print(f"  MDE by n (pp): {results['sizing']['mde_percentage_points_by_n']}")

    results["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}  ({results['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
