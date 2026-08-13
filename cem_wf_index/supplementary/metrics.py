"""Small, deterministic metrics used by the supplementary evaluators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def continuous_ndcg_at_k(scores: object, relevance: object, *, k: int = 10) -> float:
    """Compute nDCG with non-negative continuous relevance values."""

    score = np.asarray(scores, dtype=np.float64)
    gain = np.asarray(relevance, dtype=np.float64)
    if score.ndim != 1 or gain.ndim != 1 or score.shape != gain.shape or score.size == 0:
        raise ValueError("scores and relevance must be non-empty, aligned vectors")
    if not np.isfinite(score).all() or not np.isfinite(gain).all() or np.any(gain < 0.0):
        raise ValueError("scores must be finite and relevance must be finite and non-negative")
    if k < 1:
        raise ValueError("k must be positive")
    width = min(k, score.size)
    # Stable sorting keeps ties deterministic with respect to the input contract.
    order = np.argsort(-score, kind="stable")[:width]
    ideal_order = np.argsort(-gain, kind="stable")[:width]
    discount = 1.0 / np.log2(np.arange(2, width + 2, dtype=np.float64))
    # Continuous relevance follows the standard graded-gain form used by the
    # development study, rather than treating IoU as a binary label.
    ranked_gain = np.power(2.0, gain[order]) - 1.0
    ideal_gain = np.power(2.0, gain[ideal_order]) - 1.0
    dcg = float(np.sum(ranked_gain * discount))
    ideal = float(np.sum(ideal_gain * discount))
    return dcg / ideal if ideal > 0.0 else 0.0


def ranked_binary_metrics(
    candidates: Sequence[tuple[str, float]],
    relevant: set[str],
    *,
    ndcg_k: int = 10,
    count_k: int = 20,
) -> tuple[float, int]:
    """Return binary nDCG and the number of distinct relevant top-K events."""

    if not relevant:
        raise ValueError("the relevance set must not be empty")
    ordered = sorted(candidates, key=lambda item: (-float(item[1]), str(item[0])))
    identifiers = [str(candidate_id) for candidate_id, _ in ordered]
    gains = np.asarray([1.0 if item in relevant else 0.0 for item in identifiers[:ndcg_k]])
    ideal_len = min(ndcg_k, len(relevant))
    discount = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    ideal = float(np.sum(1.0 / np.log2(np.arange(2, ideal_len + 2, dtype=np.float64))))
    ndcg = float(np.sum(gains * discount) / ideal) if ideal > 0.0 else 0.0
    relevant_count = len(set(identifiers[:count_k]).intersection(relevant))
    return ndcg, relevant_count
