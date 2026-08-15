"""MT-Bench loader tests.

All offline: ``build_mtbench`` is separated from the download precisely so the
normalisation logic can be tested on hand-written rows. The two behaviours that
would silently corrupt every downstream number are pair-ordering and
without-replacement annotator sampling, so both get direct tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_eval.datasets import (
    build_mtbench,
    drop_ties,
    sample_human_pair,
    sample_one_human,
)


def row(qid, a, b, winner, judge="expert_1", turn=1):
    return {
        "question_id": qid, "model_a": a, "model_b": b,
        "winner": winner, "judge": judge, "turn": turn,
    }


# --------------------------------------------------------------------------
# pair ordering — the normalisation that makes the ceiling exist at all
# --------------------------------------------------------------------------

def test_swapped_pairs_collapse_to_one_item() -> None:
    """(A,B) and (B,A) are the same comparison. Keyed naively they become two
    items, each with a single vote, and the human-human ceiling vanishes."""
    data = build_mtbench(
        [
            row(1, "alpaca", "vicuna", "model_a", "expert_1"),
            row(1, "vicuna", "alpaca", "model_b", "expert_2"),  # same verdict, swapped
        ],
        [],
    )
    assert len(data) == 1
    item = data.items[0]
    assert item.n_human_votes == 2
    assert item.has_ceiling
    # both experts picked alpaca, which sorts first -> both "lo"
    assert [v for _, v in item.human_votes] == ["lo", "lo"]


def test_swapped_disagreement_stays_disagreement() -> None:
    data = build_mtbench(
        [
            row(1, "alpaca", "vicuna", "model_a", "expert_1"),  # alpaca
            row(1, "vicuna", "alpaca", "model_a", "expert_2"),  # vicuna
        ],
        [],
    )
    assert [v for _, v in data.items[0].human_votes] == ["lo", "hi"]


def test_turns_are_distinct_items() -> None:
    data = build_mtbench(
        [row(1, "a", "b", "model_a", "e1", turn=1), row(1, "a", "b", "model_a", "e2", turn=2)],
        [],
    )
    assert len(data) == 2


def test_ties_normalise_regardless_of_flavour() -> None:
    data = build_mtbench(
        [row(1, "a", "b", "tie", "e1")],
        [row(1, "a", "b", "tie (inconsistent)")],
    )
    item = data.items[0]
    assert item.human_votes[0][1] == "tie"
    assert item.judge_verdict == "tie"
    assert item.judge_inconsistent is True


def test_consistent_tie_is_not_flagged() -> None:
    data = build_mtbench([], [row(1, "a", "b", "tie")])
    assert data.items[0].judge_verdict == "tie"
    assert data.items[0].judge_inconsistent is False


# --------------------------------------------------------------------------
# equal footing
# --------------------------------------------------------------------------

def test_with_ceiling_requires_both_a_judge_and_two_humans() -> None:
    data = build_mtbench(
        [
            row(1, "a", "b", "model_a", "e1"), row(1, "a", "b", "model_b", "e2"),  # 2 humans, judged
            row(2, "a", "b", "model_a", "e1"), row(2, "a", "b", "model_a", "e2"),  # 2 humans, unjudged
            row(3, "a", "b", "model_a", "e1"),                                     # 1 human, judged
        ],
        [row(1, "a", "b", "model_a"), row(3, "a", "b", "model_a")],
    )
    assert len(data.with_judge()) == 2
    assert {i.question_id for i in data.with_ceiling()} == {1}


def test_summary_counts() -> None:
    data = build_mtbench(
        [row(1, "a", "b", "model_a", "e1"), row(1, "a", "b", "tie", "e2")],
        [row(1, "a", "b", "tie (inconsistent)")],
    )
    s = data.summary()
    assert s == {
        "n_items": 1, "n_with_judge": 1, "n_with_ceiling": 1,
        "n_human_votes": 2, "n_distinct_annotators": 2,
        "judge_inconsistent_rate": 1.0,
    }


# --------------------------------------------------------------------------
# sampling — the ceiling is only honest if annotators are distinct
# --------------------------------------------------------------------------

def test_human_pair_never_compares_an_annotator_with_themselves() -> None:
    """With replacement, an annotator could be paired with themselves and agree
    trivially, inflating the ceiling every judge number is measured against."""
    data = build_mtbench(
        [
            row(1, "a", "b", "model_a", "e1"),
            row(1, "a", "b", "model_b", "e2"),
            row(1, "a", "b", "tie", "e3"),
        ],
        [],
    )
    item = data.items[0]
    rng = np.random.default_rng(0)
    # the three annotators gave three different verdicts, so a self-pair would
    # be the only way to draw two equal labels
    for _ in range(300):
        a, b = sample_human_pair(item, rng)
        assert a != b


def test_sampling_is_deterministic_under_a_seed() -> None:
    data = build_mtbench(
        [row(1, "a", "b", "model_a", f"e{i}") for i in range(4)]
        + [row(1, "a", "b", "model_b", "e9")],
        [],
    )
    item = data.items[0]
    first = [sample_human_pair(item, np.random.default_rng(7)) for _ in range(5)]
    second = [sample_human_pair(item, np.random.default_rng(7)) for _ in range(5)]
    assert first == second


def test_sample_one_human_covers_every_annotator() -> None:
    data = build_mtbench(
        [row(1, "a", "b", "model_a", "e1"), row(1, "a", "b", "model_b", "e2")], []
    )
    rng = np.random.default_rng(1)
    seen = {sample_one_human(data.items[0], rng) for _ in range(100)}
    assert seen == {"lo", "hi"}


def test_sampling_rejects_items_without_enough_votes() -> None:
    data = build_mtbench([row(1, "a", "b", "model_a", "e1")], [])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="fewer than two"):
        sample_human_pair(data.items[0], rng)

    empty = build_mtbench([], [row(1, "a", "b", "model_a")])
    with pytest.raises(ValueError, match="no human votes"):
        sample_one_human(empty.items[0], rng)


# --------------------------------------------------------------------------
# ties
# --------------------------------------------------------------------------

def test_drop_ties_removes_the_position_from_every_column() -> None:
    a = ["lo", "tie", "hi", "lo"]
    b = ["lo", "hi", "hi", "tie"]
    c = ["hi", "lo", "hi", "lo"]
    ka, kb, kc = drop_ties(a, b, c)
    assert ka == ["lo", "hi"]
    assert kb == ["lo", "hi"]
    assert kc == ["hi", "hi"]


def test_drop_ties_keeps_columns_aligned() -> None:
    a = ["tie"] * 3 + ["lo"] * 3
    b = ["lo"] * 6
    ka, kb = drop_ties(a, b)
    assert len(ka) == len(kb) == 3
