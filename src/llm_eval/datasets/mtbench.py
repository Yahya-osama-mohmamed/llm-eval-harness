"""MT-Bench human judgments — the eval set with a built-in human ceiling.

Source: ``lmsys/mt_bench_human_judgments`` (Zheng et al., *Judging LLM-as-a-Judge
with MT-Bench and Chatbot Arena*, 2023). It is the right dataset for this harness
for three reasons:

1. It carries **human** pairwise preferences and **GPT-4 judge** verdicts over the
   same items, so judge-vs-human agreement needs no API calls to measure.
2. Many items were rated by more than one human, which gives the human-human
   ceiling. Without that ceiling a judge kappa is uninterpretable, and most
   eval sets cannot supply one at all.
3. Its GPT-4 split records order-swap inconsistency explicitly, so position bias
   is measurable from the released data rather than inferred.

Two normalisation problems have to be solved before any statistic is computed,
and both are easy to get silently wrong:

**Pair ordering.** The same comparison appears as (A, B) in one row and (B, A) in
another. Keyed naively, one item becomes two and the ceiling collapses. Every
item here is keyed on the *sorted* model pair, and verdicts are re-expressed as
``lo``/``hi`` relative to that sorted pair.

**Ties.** Roughly a quarter of human votes are ties, and the judge emits both a
genuine ``tie`` and a ``tie (inconsistent)`` produced when it contradicted itself
under order swap. Collapsing those two into one label would hide the position
bias, so they are kept distinct and the tie policy is left to the caller.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["lo", "hi", "tie"]

__all__ = [
    "MTBench",
    "PairwiseItem",
    "build_mtbench",
    "drop_ties",
    "load_mtbench",
    "sample_human_pair",
    "sample_one_human",
]

_HF_DATASET = "lmsys/mt_bench_human_judgments"


@dataclass(frozen=True)
class PairwiseItem:
    """One comparison: a question, two models, and everyone's verdict on it."""

    question_id: int
    turn: int
    model_lo: str
    model_hi: str
    human_votes: tuple[tuple[str, Verdict], ...] = ()
    judge_verdict: Verdict | None = None
    judge_inconsistent: bool = False

    @property
    def key(self) -> tuple:
        return (self.question_id, self.turn, self.model_lo, self.model_hi)

    @property
    def n_human_votes(self) -> int:
        return len(self.human_votes)

    @property
    def has_ceiling(self) -> bool:
        """True when at least two humans rated it, so a human-human pair exists."""
        return self.n_human_votes >= 2


@dataclass
class MTBench:
    items: list[PairwiseItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def with_judge(self) -> list[PairwiseItem]:
        return [i for i in self.items if i.judge_verdict is not None]

    def with_ceiling(self) -> list[PairwiseItem]:
        """Items rated by >=2 humans *and* judged — the only items on which the
        judge and the ceiling can be compared on equal footing.

        Comparing a judge measured on all items against a ceiling measured on the
        subset with repeat annotation is the obvious trap here: the subsets differ
        systematically, because items get a second annotator for a reason.
        """
        return [i for i in self.items if i.has_ceiling and i.judge_verdict is not None]

    def summary(self) -> dict:
        judged = self.with_judge()
        return {
            "n_items": len(self.items),
            "n_with_judge": len(judged),
            "n_with_ceiling": len(self.with_ceiling()),
            "n_human_votes": sum(i.n_human_votes for i in self.items),
            "n_distinct_annotators": len({j for i in self.items for j, _ in i.human_votes}),
            "judge_inconsistent_rate": (
                sum(i.judge_inconsistent for i in judged) / len(judged) if judged else float("nan")
            ),
        }


def _sorted_pair(model_a: str, model_b: str) -> tuple[str, str]:
    lo, hi = sorted([model_a, model_b])
    return lo, hi


def _normalise_verdict(winner: str, model_a: str, model_b: str) -> Verdict:
    """Re-express a winner as ``lo``/``hi`` against the sorted pair."""
    if winner.startswith("tie"):
        return "tie"
    lo, _hi = _sorted_pair(model_a, model_b)
    won = model_a if winner == "model_a" else model_b
    return "lo" if won == lo else "hi"


def load_mtbench(cache_dir: str | None = None) -> MTBench:
    """Load and normalise the dataset. Requires ``datasets`` (dev extra).

    Downloads ~5,700 rows on first call and caches them; later calls are local.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "load_mtbench needs the 'datasets' package: pip install -e '.[data]'"
        ) from exc

    ds = load_dataset(_HF_DATASET, cache_dir=cache_dir)
    return build_mtbench(list(ds["human"]), list(ds["gpt4_pair"]))


def build_mtbench(human_rows: list[dict], judge_rows: list[dict]) -> MTBench:
    """Pure assembly step, split out from the download so it can be unit-tested
    on hand-written rows without touching the network."""
    votes: dict[tuple, list[tuple[str, Verdict]]] = defaultdict(list)
    meta: dict[tuple, tuple[int, int, str, str]] = {}

    def register(row: dict) -> tuple:
        lo, hi = _sorted_pair(row["model_a"], row["model_b"])
        k = (row["question_id"], row["turn"], lo, hi)
        meta[k] = (row["question_id"], row["turn"], lo, hi)
        return k

    for row in human_rows:
        k = register(row)
        votes[k].append(
            (row["judge"], _normalise_verdict(row["winner"], row["model_a"], row["model_b"]))
        )

    judged: dict[tuple, tuple[Verdict, bool]] = {}
    for row in judge_rows:
        k = register(row)
        raw = row["winner"]
        judged[k] = (
            _normalise_verdict(raw, row["model_a"], row["model_b"]),
            raw == "tie (inconsistent)",
        )

    items = []
    for k, m in sorted(meta.items()):
        verdict, inconsistent = judged.get(k, (None, False))
        items.append(
            PairwiseItem(
                question_id=m[0],
                turn=m[1],
                model_lo=m[2],
                model_hi=m[3],
                human_votes=tuple(votes.get(k, ())),
                judge_verdict=verdict,
                judge_inconsistent=inconsistent,
            )
        )
    return MTBench(items=items)


# ---------------------------------------------------------------------------
# sampling
#
# An item rated by seven humans must not contribute seven times the weight of an
# item rated by two: the bootstrap resamples *items*, so each item has to supply
# exactly one comparison. These helpers enforce that, deterministically.
# ---------------------------------------------------------------------------


def sample_one_human(item: PairwiseItem, rng) -> Verdict:
    """One human verdict for this item, chosen uniformly at random."""
    if not item.human_votes:
        raise ValueError(f"item {item.key} has no human votes")
    return item.human_votes[int(rng.integers(len(item.human_votes)))][1]


def sample_human_pair(item: PairwiseItem, rng) -> tuple[Verdict, Verdict]:
    """Two verdicts from two *distinct* annotators on this item.

    Sampling with replacement would let one annotator be compared against
    themselves, inflating the ceiling toward 1.0 — the single most important
    detail in this file, because the ceiling is what every judge number is
    measured against.
    """
    if not item.has_ceiling:
        raise ValueError(f"item {item.key} has fewer than two human votes")
    i, j = rng.choice(len(item.human_votes), size=2, replace=False)
    return item.human_votes[int(i)][1], item.human_votes[int(j)][1]


def drop_ties(*label_columns: list[Verdict]) -> tuple[list[Verdict], ...]:
    """Keep only positions where no column is a tie.

    Reported alongside the tie-inclusive figures rather than instead of them.
    Ties are a quarter of the human votes, and dropping them silently is a
    common way to make judge agreement look far better than it is.
    """
    keep = [i for i in range(len(label_columns[0])) if all(c[i] != "tie" for c in label_columns)]
    return tuple([c[i] for i in keep] for c in label_columns)
