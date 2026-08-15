# LLM Evaluation Harness

**An LLM judge is a measurement instrument. An instrument you have not validated
is not a measurement — it is a vibe with a number attached.**

Most LLM eval stacks report "GPT-4 rates our system 8.2/10" with no agreement
statistic, no confidence interval, and no way to tell whether 8.2 → 8.0 is a
regression or noise. This project sits one layer beneath those stacks and asks
the question they skip: *is the scorer any good, and how would you know?*

> **Status: milestones 1–3 of 7.** Statistical core validated, an eval set with a
> real human ceiling wired in, and one judge measured. No live judge adapters, no
> CI gate yet. [RESULTS.md](RESULTS.md) is the source of truth — anything not in
> it has not been measured.

---

## The first result

GPT-4 judging MT-Bench, measured against the ceiling of how well 65 human
annotators agree with each other, on the 961 items that carry both (full detail
in [RESULTS.md §3](RESULTS.md)):

| | Human ceiling | GPT-4 vs human | Gap |
|---|---|---|---|
| Cohen's κ, ties kept | 0.499 [0.454, 0.543] | 0.474 [0.428, 0.518] | +0.026 [−0.024, +0.072] |
| Cohen's κ, ties dropped | 0.728 [0.667, 0.780] | 0.750 [0.689, 0.802] | −0.022 [−0.081, +0.038] |

Three things fall out, and the second and third are the ones you will not find in
the write-ups this replicates:

1. **The judge is indistinguishable from the human ceiling** — no gap interval
   excludes zero. That reproduces the MT-Bench paper's headline, with the
   intervals it did not report.
2. **The ceiling is low.** Human–human κ ≈ 0.50. "As good as a human" here partly
   means humans do not agree with each other much, and no judge can be shown to
   beat a measurement that noisy.
3. **Tie handling moves the number more than the judge does.** Dropping ties
   removes 43% of items and moves κ from 0.50 to 0.73 — an order of magnitude
   more than the judge–human gap. The biggest driver of a reported agreement
   score on this dataset is a preprocessing choice.

Position bias, measured over all 2,400 judged items: **15.8% [14.4%, 17.3%]** of
GPT-4's verdicts flip when the two responses are swapped.

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
| 1 | Eval set with a human ceiling (MT-Bench, 65 annotators) | **done** |
| 2 | Metrics core, validated by simulation | **done** |
| 3 | Judge–human agreement vs the human–human ceiling | **done** |
| 4 | Bias probes: position ✅, verbosity, self-preference, formatting | position done |
| 5 | Slicing + power analysis | not started |
| 6 | Regression gate wired into CI | not started |
| 7 | Live judge adapters, Docker, architecture diagram | not started |

Milestone 1 was originally scoped as hand-annotating ~400 items. MT-Bench
replaced that with 3,355 votes from 65 annotators — and removed the
annotator-fatigue and self-anchoring failure modes that self-labelling would have
introduced, since the person building the harness would also have been the person
generating its ground truth.

## Limitations

Listed in full in [RESULTS.md §6](RESULTS.md). The three that matter most:

- **One judge, one dataset, one task format.** GPT-4 on pairwise preference.
  Nothing here generalises to other judges or rubrics without re-measuring.
- The judge verdicts are as released in 2023, so this is a fixed historical
  artefact rather than a live benchmark of any current model.
- Kappa confidence intervals under-cover at small n (92.6% measured against a
  nominal 95%), so small-set kappa intervals here are approximate.

## License

MIT
