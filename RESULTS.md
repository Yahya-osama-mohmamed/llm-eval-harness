# RESULTS — LLM Evaluation Harness

Source of truth. Every number below was produced by running code in this
repository; the command that produces it is given. Nothing here is projected,
estimated, or copied from a paper — including the numbers that happen to agree
with one.

**Status: milestones 1–3 of 7 complete.** The statistical core is built and
validated, an eval set with a genuine human ceiling is wired in, and the first
judge has been measured. No judge adapters (no live LLM calls), no slicing, no
CI regression gate yet.

---

## 1. What is implemented

| Module | Contents | State |
|---|---|---|
| `metrics/agreement.py` | Cohen's kappa (unweighted/linear/quadratic), Gwet's AC1, Krippendorff's alpha | done, validated |
| `metrics/calibration.py` | Brier, ECE (equal-width + equal-mass), reliability curves | done, validated |
| `metrics/resample.py` | Bootstrap CIs (percentile + BCa), paired difference intervals | done, validated |
| `metrics/power.py` | Two-proportion power, required n, minimum detectable effect | done, validated |
| `metrics/correction.py` | Holm, Benjamini-Hochberg, Bonferroni | done, validated |
| `datasets/mtbench.py` | MT-Bench human judgments: pair normalisation, ceiling extraction, sampling | done, validated |
| `judges/`, `probes/`, `gates/`, `report/` | — | **not started** |

```bash
pytest tests/ -q                          # 178 tests
ruff check src tests scripts
python scripts/validate_statistics.py     # section 2, ~40s
python scripts/run_mtbench_experiment.py  # section 3, ~3min
```

The statistical core has no LLM dependency and no dataset dependency. It is
fully testable without an API key, which is what makes the harness's own claims
checkable by a reader who has no account.

## 2. Validating the instrument

Raw output: [`reports/statistical_validation.json`](reports/statistical_validation.json).

### 2.1 Bootstrap interval coverage

A 95% interval should contain the true value 95% of the time. Measured over
independent simulated datasets with a known answer:

| Statistic | Method | Trials | n | Measured coverage |
|---|---|---|---|---|
| Mean of Bernoulli(0.3) | percentile | 1,000 | 120 | **0.959** |
| Mean of Bernoulli(0.3) | BCa | 1,000 | 120 | **0.962** |
| Cohen's kappa (true κ = 0.7010) | percentile | 500 | 150 | **0.926** |
| Cohen's kappa (true κ = 0.7010) | BCa | 500 | 150 | **0.926** |

The mean is fine — both methods land slightly conservative, the safe direction.

Kappa is not. Both cover 92.6% against a nominal 95%, so a "95% interval" on
kappa at n=150 is really about a 93% interval. **Small-n kappa intervals in this
harness are mildly optimistic and should be read as approximate.**

**BCa did not beat percentile — identical to three decimals.** The stated reason
for making BCa the default was that it corrects skew for bounded statistics like
kappa. On this evidence that reason is unsupported. It remains the default only
because it is not worse; the docstring in `resample.py` has been rewritten to say
so rather than to keep asserting the theory.

### 2.2 Multiple-comparison correction

20 hypotheses, all null, α=0.05, 5,000 trials. "Family-wise error" = the run
reported at least one significant result when nothing was real.

| Policy | Family-wise error rate |
|---|---|
| Uncorrected | **0.636** |
| Holm | **0.056** |
| Benjamini-Hochberg | **0.057** |

The uncorrected rate matches theory (1 − 0.95²⁰ = 0.642) within simulation noise,
which validates the harness. Practical reading: **a 20-check probe suite with no
correction reports a fake finding in about two runs out of three.**

### 2.3 What an eval set of a given size can resolve

Minimum detectable effect at 80% power, α=0.05, two-sided, 80% baseline:

| Items per arm | 100 | 200 | 500 | 1,000 | 2,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|---|---|
| Smallest detectable drop | 17.9 pp | 12.3 pp | 7.5 pp | 5.2 pp | 3.7 pp | 2.3 pp | 1.6 pp |

Detecting a 1-point regression needs **25,583** items per arm; 2 points needs
**6,510**; 5 points needs **1,094**.

The common practice of eyeballing a regression on a 200-item eval set cannot
detect anything smaller than a 12-point drop, so most "small regressions"
reported at that scale are noise in both directions.

## 3. First judge measured: GPT-4 on MT-Bench

Raw output: [`reports/mtbench_agreement.json`](reports/mtbench_agreement.json).
Dataset: `lmsys/mt_bench_human_judgments` (Zheng et al., 2023) — 2,400 judged
items, 3,355 human votes, **65 distinct annotators**. No API calls; the human
votes and the GPT-4 verdicts both ship with the dataset.

**961 items** carry ≥2 human votes *and* a judge verdict. Everything below is
measured on those 961, so the judge and the ceiling stand on the same items —
measuring the judge on all 2,400 and the ceiling on the 961 would compare
different item sets, and items get a second annotator for a reason.

Per item, two *distinct* annotators are drawn: the ceiling is (h1 vs h2), the
judge is (judge vs h1). Intervals are BCa over 4,000 item resamples; the gap is
a paired bootstrap on the same resampled items.

### 3.1 Ties kept (n = 961)

| Metric | Human ceiling | GPT-4 vs human | Ceiling − judge |
|---|---|---|---|
| Raw agreement | 0.675 [0.643, 0.702] | 0.657 [0.624, 0.685] | +0.019 [−0.015, +0.049] |
| Cohen's κ | 0.499 [0.454, 0.543] | 0.474 [0.428, 0.518] | +0.026 [−0.024, +0.072] |
| Gwet's AC1 | 0.520 [0.474, 0.562] | 0.491 [0.443, 0.535] | +0.029 [−0.020, +0.075] |

### 3.2 Ties dropped (n = 546)

| Metric | Human ceiling | GPT-4 vs human | Ceiling − judge |
|---|---|---|---|
| Raw agreement | 0.865 [0.832, 0.888] | 0.876 [0.844, 0.899] | −0.011 [−0.042, +0.018] |
| Cohen's κ | 0.728 [0.667, 0.780] | 0.750 [0.689, 0.802] | −0.022 [−0.081, +0.038] |
| Gwet's AC1 | 0.820 [0.778, 0.854] | 0.834 [0.795, 0.869] | −0.015 [−0.056, +0.024] |

### 3.3 Findings

**1. The judge is statistically indistinguishable from the human ceiling.** Not
one of the six gap intervals excludes zero. This reproduces the MT-Bench paper's
headline claim — but with the confidence intervals it did not report, which is
the difference between "GPT-4 agrees with humans as much as humans do" as a
slogan and as a measurement. With ties dropped the judge is nominally *above*
the ceiling (κ 0.750 vs 0.728), still not distinguishable.

**2. The ceiling is low, and that is the more important half of the finding.**
Human–human κ is 0.499 with ties kept — moderate agreement at best. "GPT-4
matches human agreement" is true and partly means *humans do not agree with each
other much on this task.* Any claim of the form "the judge is as good as a human"
inherits that ceiling and should be quoted with it. A judge cannot be shown to be
better than a measurement this noisy.

**3. Tie handling moves the numbers more than the judge does.** Dropping ties
removes 415 of 961 items (43%) and moves raw agreement from 0.675 to 0.865, κ
from 0.50 to 0.73. The judge–ceiling gap, meanwhile, moves by about 0.02 and
never becomes significant. **The single largest driver of a reported agreement
number on this dataset is a preprocessing choice, not the judge.** A paper
reporting only the tie-excluded figure is reporting a different quantity, and the
two are not comparable.

**4. Position bias is real and large: 15.8% [14.4%, 17.3%].** GPT-4 contradicts
itself on about one item in six when the two responses are swapped — measured
over all 2,400 judged items, not inferred. Against §2.3, an eval set would need
roughly 1,000 items merely to detect a 5-point change in that rate.

**5. The annotator draw does not drive the result.** Over 10 seeds, ceiling κ
ranges 0.481–0.507 and judge κ 0.469–0.512 — within the width of the confidence
intervals, so the conclusion is not an artefact of which annotators were drawn.

## 4. What failed or surprised

1. **BCa did not earn its keep** (§2.1); the default is justified by reputation,
   not by measurement in this repo.
2. **Kappa intervals under-cover at n=150** (92.6% vs nominal 95%).
3. **Ties turned out to dominate** (§3.3.3). The plan treated tie policy as a
   reporting detail to mention; it is the largest single lever on the headline
   number and now leads the findings.
4. **Two test expectations were wrong on first run** and the implementations
   right: a "kappa paradox" case not skewed enough (κ came out 0.479, not <0.35),
   and an assumption that 5,000 items resolve 2 points when the real figure is
   6,510 — an intuition off by 30%, in the optimistic direction. A small
   demonstration of the project's own thesis.

## 5. Open items

- Justify or drop the BCa default with a coverage sweep across n and κ.
- Variance-stabilised kappa intervals; re-measure coverage.
- Milestones 4–7: live judge adapters, the remaining bias probes (verbosity,
  self-preference, formatting), slicing, CI regression gate, report renderer.
- Second dataset (SummEval) to test whether these findings hold on an ordinal
  task with a different annotator pool.

## 6. Limitations

- **One judge, one dataset, one task format.** GPT-4 on pairwise preference.
  Nothing here generalises to other judges, rubrics, or scoring formats without
  being re-measured.
- The judge verdicts are as released in 2023; the model behind them has since
  changed, so this is a fixed historical artefact, not a live benchmark.
- Position bias is measured from the dataset's own inconsistency flag. The other
  three probes in the brief (verbosity, self-preference, formatting) need live
  judge calls and are not done.
- Power calculations use a normal approximation; below ~30 items per arm, or at
  rates near 0 or 1, prefer an exact test.
- Krippendorff's ordinal distance uses the rank-based form.
- Coverage was measured on binary labels only.
