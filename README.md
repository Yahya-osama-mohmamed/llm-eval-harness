# LLM Evaluation Harness

**An LLM judge is a measurement instrument. An instrument you have not validated
is not a measurement — it is a vibe with a number attached.**

Most LLM eval stacks report "GPT-4 rates our system 8.2/10" with no agreement
statistic, no confidence interval, and no way to tell whether 8.2 → 8.0 is a
regression or noise. This project sits one layer beneath those stacks and asks
the question they skip: *is the scorer any good, and how would you know?*

> **Status: milestone 2 of 7 — statistical core only.**
> No judges, no eval set, no LLM calls yet. See [RESULTS.md](RESULTS.md) for
> exactly what is and is not established. This README describes what exists;
> anything not in RESULTS.md has not been measured.

---

## Why this instead of RAGAS / DeepEval / promptfoo

Those libraries compute scores. They largely do not tell you whether the scorer
is trustworthy, and they mostly do not report uncertainty. Both are necessary
before a score can drive a decision:

- **Is the judge right?** Requires human labels and an agreement statistic — and
  requires knowing how much two *humans* agree, because a judge cannot
  meaningfully beat the human ceiling. Without that ceiling a kappa of 0.55 is
  uninterpretable.
- **Is the difference real?** Requires an interval and a minimum detectable
  effect. A 200-item eval set cannot resolve anything smaller than a 12-point
  drop (measured — [RESULTS.md §2.3](RESULTS.md)), so most "small regressions"
  reported at that scale are noise.

Planned: run one comparable metric through RAGAS on the same data and report
where the numbers differ and why.

## What is built

```
src/llm_eval/metrics/          no LLM dependency — testable without an API key
├── agreement.py    Cohen's kappa (weighted), Gwet's AC1, Krippendorff's alpha
├── calibration.py  Brier, ECE (equal-width + equal-mass), reliability curves
├── resample.py     bootstrap CIs (percentile + BCa), paired difference intervals
├── power.py        power, required n, minimum detectable effect
└── correction.py   Holm, Benjamini-Hochberg, Bonferroni
```

Keeping the statistics free of any LLM dependency is the point, not an
accident: it is what lets a reader verify the harness's claims without an
account, and 165 tests do exactly that.

## Three design decisions worth reading

**Three agreement coefficients, not one.** Kappa collapses toward zero when one
category dominates, even at 95% raw agreement — the "kappa paradox". LLM eval
sets are almost always skewed that way, because most outputs are fine. Gwet's
AC1 does not have that failure mode, so both are reported; a large gap between
them is a finding about the label distribution. Krippendorff's alpha handles the
third annotator who only labelled a subset.

**Every metric returns an interval.** `bootstrap_ci` resamples paired columns
together, so a `(human, judge)` pair stays paired and the interval reflects item
sampling rather than pretending the columns are independent. Comparing two
separately-computed intervals by eye is the wrong test; `paired_bootstrap_diff`
is the right one.

**Corrections are not optional.** Twenty bias probes at α=0.05 report a fake
finding in about two runs of three — 0.636 measured, against 0.056 under Holm
([RESULTS.md §2.2](RESULTS.md)). Any probe suite without correction is
manufacturing results.

## Quick start

```bash
python -m venv venv && venv/Scripts/activate   # Linux/macOS: source venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

Reproduce every number in RESULTS.md (~40s):

```bash
python scripts/validate_statistics.py
```

## Example

```python
from llm_eval.metrics import agreement_report, bootstrap_ci, cohens_kappa, holm

human  = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
judge  = [1, 0, 1, 0, 0, 1, 1, 1, 1, 1]

agreement_report(human, judge)
# {'n': 10, 'raw_agreement': 0.8, 'cohens_kappa': 0.5238,
#  'gwets_ac1': 0.6552, 'majority_class_share': 0.7}

ci = bootstrap_ci((human, judge), cohens_kappa)
print(ci)          # 0.5238 [-0.2500, 1.0000]  <- 10 items establish nothing

holm([0.03, 0.2, 0.4], labels=["position", "verbosity", "self_pref"]).significant()
# []   <- the p=0.03 "finding" does not survive three comparisons
```

## Roadmap

| # | Milestone | State |
|---|---|---|
| 1 | Eval set + human labels (≥2 annotators on a subset, for the ceiling) | not started |
| 2 | Judge adapters + metrics core | **core done** |
| 3 | Judge–human agreement vs the human–human ceiling | not started |
| 4 | Bias probes: position, verbosity, self-preference, formatting | not started |
| 5 | Slicing + power analysis | not started |
| 6 | Regression gate wired into CI | not started |
| 7 | Docker, architecture diagram, limitations | not started |

Milestones 1–3 alone constitute a defensible project; they ship before 4 begins.

## Limitations

Listed in full in [RESULTS.md §5](RESULTS.md). The two that matter most today:
kappa confidence intervals under-cover at small n (92.6% measured against a
nominal 95%), and no judge has been evaluated, so this repository currently
supports **no claim about any LLM's agreement with humans.**

## License

MIT
