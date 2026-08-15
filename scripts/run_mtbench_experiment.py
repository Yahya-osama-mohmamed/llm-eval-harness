"""Milestone 3: how well does the GPT-4 judge agree with humans, against the
ceiling of how well humans agree with each other?

    python scripts/run_mtbench_experiment.py

No API calls — both the human votes and the judge verdicts are in the released
dataset. Results land in reports/mtbench_agreement.json and are quoted in
RESULTS.md section 3.

Design decisions that determine whether the answer means anything:

* **Equal footing.** Judge and ceiling are both measured on the 961 items that
  have >=2 human votes *and* a judge verdict. Measuring the judge on all 2,400
  items and the ceiling on the 961 would compare two different item sets, and
  items get a second annotator for a reason.
* **One comparison per item.** Two distinct annotators are drawn per item; the
  ceiling is (h1 vs h2) and the judge is (judge vs h1). Same items, same
  one-human treatment, so the difference is attributable to the judge.
* **Paired difference.** The ceiling-minus-judge gap is bootstrapped on the same
  resampled items, not differenced from two independent intervals.
* **Ties reported both ways.** Ties are about a quarter of human votes. Dropping
  them silently is the standard way to make a judge look better than it is.
* **Annotator-draw sensitivity.** Which two annotators get drawn is itself
  random, so the whole thing is repeated across seeds and the spread reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from llm_eval.datasets import drop_ties, load_mtbench, sample_human_pair
from llm_eval.metrics import (
    bootstrap_ci,
    cohens_kappa,
    gwets_ac1,
    paired_bootstrap_diff,
    raw_agreement,
)

OUT = Path(__file__).resolve().parents[1] / "reports" / "mtbench_agreement.json"
CATEGORIES = ["hi", "lo", "tie"]
N_BOOT = 4000


def _kappa(a, b):
    return cohens_kappa(list(a), list(b), categories=CATEGORIES)


def _ac1(a, b):
    return gwets_ac1(list(a), list(b), categories=CATEGORIES)


def _raw(a, b):
    return raw_agreement(list(a), list(b))


def build_columns(items, seed: int):
    """Return (h1, h2, judge) aligned per item for one annotator draw."""
    rng = np.random.default_rng(seed)
    h1, h2, jd = [], [], []
    for it in items:
        a, b = sample_human_pair(it, rng)
        h1.append(a)
        h2.append(b)
        jd.append(it.judge_verdict)
    return np.array(h1), np.array(h2), np.array(jd)


def measure(h1, h2, jd, label: str) -> dict:
    """Ceiling, judge, and the gap between them — each with a BCa interval."""
    out = {"label": label, "n_items": len(h1)}

    for name, fn in (("raw_agreement", _raw), ("cohens_kappa", _kappa), ("gwets_ac1", _ac1)):
        ceiling = bootstrap_ci((h1, h2), fn, n_resamples=N_BOOT)
        judge = bootstrap_ci((jd, h1), fn, n_resamples=N_BOOT)
        gap = paired_bootstrap_diff(
            (h1, h2, jd),
            lambda a, b, _c, _fn=fn: _fn(a, b),      # human-human
            lambda a, _b, c, _fn=fn: _fn(c, a),      # judge-human
            n_resamples=N_BOOT,
        )
        out[name] = {
            "human_ceiling": [ceiling.estimate, ceiling.low, ceiling.high],
            "judge_vs_human": [judge.estimate, judge.low, judge.high],
            "ceiling_minus_judge": [gap.estimate, gap.low, gap.high],
            "gap_excludes_zero": gap.excludes(0.0),
        }
    return out


def main() -> None:
    data = load_mtbench()
    summary = data.summary()
    items = data.with_ceiling()
    print(f"dataset: {summary}")
    print(f"items on equal footing (>=2 humans AND judged): {len(items)}\n")

    results: dict = {
        "generated_by": "scripts/run_mtbench_experiment.py",
        "dataset": "lmsys/mt_bench_human_judgments",
        "dataset_summary": summary,
        "n_bootstrap": N_BOOT,
    }

    h1, h2, jd = build_columns(items, seed=0)
    results["with_ties"] = measure(h1, h2, jd, "ties kept")

    h1n, h2n, jdn = drop_ties(list(h1), list(h2), list(jd))
    results["without_ties"] = measure(
        np.array(h1n), np.array(h2n), np.array(jdn), "ties dropped"
    )

    for block in ("with_ties", "without_ties"):
        r = results[block]
        print(f"--- {r['label']}  (n={r['n_items']}) ---")
        for m in ("raw_agreement", "cohens_kappa", "gwets_ac1"):
            c, j, g = r[m]["human_ceiling"], r[m]["judge_vs_human"], r[m]["ceiling_minus_judge"]
            print(f"  {m:<15} ceiling {c[0]:.4f} [{c[1]:.4f},{c[2]:.4f}]   "
                  f"judge {j[0]:.4f} [{j[1]:.4f},{j[2]:.4f}]   "
                  f"gap {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]"
                  f"{'  *' if r[m]['gap_excludes_zero'] else ''}")
        print()

    # position bias: the judge contradicted itself under A/B order swap
    judged = data.with_judge()
    flags = np.array([float(i.judge_inconsistent) for i in judged])
    pos = bootstrap_ci(flags, np.mean, n_resamples=N_BOOT)
    results["position_bias"] = {
        "n_judged_items": len(judged),
        "inconsistent_rate": [pos.estimate, pos.low, pos.high],
    }
    print(f"position bias (order-swap inconsistency): {pos}  over {len(judged)} items\n")

    # how much does the annotator draw matter?
    sens = []
    for s in range(10):
        a, b, c = build_columns(items, seed=s)
        sens.append({"seed": s, "ceiling_kappa": _kappa(a, b), "judge_kappa": _kappa(c, a)})
    results["seed_sensitivity"] = {
        "seeds": len(sens),
        "ceiling_kappa_range": [
            min(x["ceiling_kappa"] for x in sens), max(x["ceiling_kappa"] for x in sens)
        ],
        "judge_kappa_range": [
            min(x["judge_kappa"] for x in sens), max(x["judge_kappa"] for x in sens)
        ],
        "per_seed": sens,
    }
    ck = results["seed_sensitivity"]["ceiling_kappa_range"]
    jk = results["seed_sensitivity"]["judge_kappa_range"]
    print("annotator-draw sensitivity over 10 seeds:")
    print(f"  ceiling kappa {ck[0]:.4f} - {ck[1]:.4f}")
    print(f"  judge   kappa {jk[0]:.4f} - {jk[1]:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
