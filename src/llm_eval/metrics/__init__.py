"""Statistical core: agreement, calibration, resampling, power, correction.

Deliberately free of any LLM dependency. Every function here is testable without
an API key, which is what makes the harness's claims about judges verifiable at
all — if the statistics only ran behind a paid API, nobody could check them.
"""

from .agreement import (
    agreement_report,
    cohens_kappa,
    confusion_matrix,
    gwets_ac1,
    krippendorff_alpha,
    raw_agreement,
)
from .calibration import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    reliability_curve,
)
from .correction import CorrectionResult, benjamini_hochberg, bonferroni, correct, holm
from .power import mde_two_proportions, n_for_power, power_report, power_two_proportions
from .resample import CI, bootstrap_ci, paired_bootstrap_diff

__all__ = [
    "CI",
    "CorrectionResult",
    "agreement_report",
    "benjamini_hochberg",
    "bonferroni",
    "bootstrap_ci",
    "brier_score",
    "calibration_report",
    "cohens_kappa",
    "confusion_matrix",
    "correct",
    "expected_calibration_error",
    "gwets_ac1",
    "holm",
    "krippendorff_alpha",
    "mde_two_proportions",
    "n_for_power",
    "paired_bootstrap_diff",
    "power_report",
    "power_two_proportions",
    "raw_agreement",
    "reliability_curve",
]
