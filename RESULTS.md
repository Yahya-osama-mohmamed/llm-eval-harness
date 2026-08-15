# RESULTS — LLM Evaluation Harness

Source of truth. Every number below was produced by running code in this
repository; the command that produces it is given. Nothing here is projected,
estimated, or copied from a paper.

**Status: milestone 2 of 7.** The statistical core is built and validated. There
are no judges, no eval set, no human labels and no LLM API calls yet, so there
are no agreement results about any actual judge. Do not put judge findings on a
CV from this file — there are none.

---

## 1. What is implemented

| Module | Contents | State |
|---|---|---|
| `metrics/agreement.py` | Cohen's kappa (unweighted/linear/quadratic), Gwet's AC1, Krippendorff's alpha (n raters, missing data, nominal/ordinal/interval) | done, validated |
| `metrics/calibration.py` | Brier, ECE (equal-width + equal-mass), reliability curves | done, validated |
| `metrics/resample.py` | Bootstrap CIs (percentile + BCa), paired difference intervals | done, validated |
| `metrics/power.py` | Two-proportion power, required n, minimum detectable effect | done, validated |
| `metrics/correction.py` | Holm, Benjamini-Hochberg, Bonferroni | done, validated |
| `judges/`, `probes/`, `gates/`, `report/` | — | **not started** |

Deliberate constraint: the statistical core has no LLM dependency. It is fully
testable without an API key, which is what makes the harness's own claims
checkable by a reader who has no account.

```bash
pytest tests/ -q          # 165 tests
ruff check src tests scripts
```

Both pass as of this commit. 165 tests over ~1,000 lines of implementation.

## 2. Validation results

Produced by `python scripts/validate_statistics.py`; raw output in
[`reports/statistical_validation.json`](reports/statistical_validation.json).

### 2.1 Bootstrap interval coverage

A 95% interval should contain the true value 95% of the time. Measured over
independent simulated datasets with a known answer:

| Statistic | Method | Trials | n per trial | Measured coverage |
|---|---|---|---|---|
| Mean of a Bernoulli(0.3) | percentile | 1,000 | 120 | **0.959** |
| Mean of a Bernoulli(0.3) | BCa | 1,000 | 120 | **0.962** |
| Cohen's kappa (true κ = 0.7010) | percentile | 500 | 150 | **0.926** |
| Cohen's kappa (true κ = 0.7010) | BCa | 500 | 150 | **0.926** |

**Two honest readings of this table.**

The mean is fine — both methods land slightly conservative, which is the
expected and safe direction.

Kappa is not. Both methods cover 92.6% against a nominal 95%, so at n=150 a
"95% interval" on kappa is really about a 93% interval. That is a real
limitation and it is stated here rather than smoothed over: **kappa intervals
from small eval sets in this harness are mildly optimistic and should be read as
approximate.**

Worse for my own design, **BCa did not beat percentile here — the two agree to
three decimals.** The stated rationale for making BCa the default was that it
corrects skew for bounded statistics like kappa. On this evidence that rationale
is unsupported. BCa remains the default for now because it is not *worse* and
its advantage may appear at other n and κ, but the docstring claim in
`resample.py` is currently stronger than the measurement, and that gap is
tracked as an open item in §4.

### 2.2 Multiple-comparison correction

20 hypotheses, all null, α=0.05, 5,000 trials. "Family-wise error" means the run
reported at least one significant result when nothing was real.

| Policy | Family-wise error rate |
|---|---|
| Uncorrected | **0.636** |
| Holm | **0.056** |
| Benjamini-Hochberg | **0.057** |

The uncorrected rate matches theory (1 − 0.95²⁰ = 0.642) to within simulation
noise, which validates the harness. The practical reading: **a bias-probe suite
of 20 checks with no correction reports a fake finding about two runs in three.**
Holm and BH both bring that to the nominal 5%.

### 2.3 What an eval set of a given size can resolve

Minimum detectable effect at 80% power, α=0.05, two-sided, against an 80%
baseline pass rate:

| Eval items per arm | Smallest detectable drop |
|---|---|
| 100 | 17.9 pp |
| 200 | 12.3 pp |
| 500 | 7.5 pp |
| 1,000 | 5.2 pp |
| 2,000 | 3.7 pp |
| 5,000 | 2.3 pp |
| 10,000 | 1.6 pp |

Detecting a 1-point regression at this baseline needs **25,583 items per arm**;
2 points needs **6,510**; 5 points needs **1,094**.

This table is the most immediately useful output in the repository. The common
practice of eyeballing a regression on a 200-item eval set cannot detect
anything smaller than a 12-point drop — so most reported "small regressions" at
that scale are noise, in both directions.

## 3. What failed or surprised

1. **BCa did not earn its keep** (§2.1). The default is currently justified by
   reputation rather than by measurement in this repo.
2. **Kappa intervals under-cover at n=150.** Documented rather than hidden;
   likely improved by a variance-stabilising transform, which is untested here.
3. **Two test expectations were wrong on first run** and the implementations
   were right: a constructed "kappa paradox" case that was not skewed enough
   (kappa came out 0.479, not <0.35), and an assumption that 5,000 items resolve
   2 points when the real requirement is 6,510. Both were corrected against the
   measurement. The second is a small illustration of the project's own thesis —
   an intuition about eval-set size was off by 30%, in the optimistic direction.

## 4. Open items

- Justify or drop the BCa default with a coverage sweep across n and κ.
- Variance-stabilised kappa intervals; re-measure coverage.
- Milestones 1 and 3–7 from the brief: eval set with human labels, judge
  adapters, bias probes, slicing, CI regression gate, report renderer.

## 5. Limitations

- No judge has been evaluated. No claim about any LLM's agreement with humans is
  supported by this repository today.
- Power calculations use a normal approximation; below ~30 items per arm, or at
  rates very near 0 or 1, prefer an exact test.
- Krippendorff's ordinal distance uses the rank-based form, not the
  marginal-dependent one.
- Coverage was measured on binary labels only; ordinal and multi-class coverage
  is untested.
