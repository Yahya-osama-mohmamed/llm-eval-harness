"""Eval sets. Each loader normalises a public dataset into the shape the metrics
expect, and each documents the normalisation traps it had to solve."""

from .mtbench import (
    MTBench,
    PairwiseItem,
    build_mtbench,
    drop_ties,
    load_mtbench,
    sample_human_pair,
    sample_one_human,
)

__all__ = [
    "MTBench",
    "PairwiseItem",
    "build_mtbench",
    "drop_ties",
    "load_mtbench",
    "sample_human_pair",
    "sample_one_human",
]
