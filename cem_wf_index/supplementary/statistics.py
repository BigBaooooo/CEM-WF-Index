"""Query-level paired-bootstrap uncertainty estimates.

The unit of resampling is the query.  The helper is reusable for any paired
comparison whose per-query values have already been computed under the same
evaluation contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PairedBootstrapResult:
    """A percentile interval for the mean paired improvement."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    query_count: int
    resamples: int
    seed: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _paired_values(baseline: object, proposed: object) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(proposed, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("paired values must be one-dimensional")
    if left.size == 0 or left.shape != right.shape:
        raise ValueError("paired values must be non-empty and have equal length")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired values must be finite")
    return left, right


def paired_bootstrap_interval(
    baseline: object,
    proposed: object,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20262202,
) -> PairedBootstrapResult:
    """Estimate ``mean(proposed - baseline)`` with a paired query bootstrap.

    Both vectors must contain one value per query in the same order.  Keeping
    the pair intact during resampling preserves within-query dependence.
    """

    left, right = _paired_values(baseline, proposed)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")

    differences = right - left
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    # Chunking bounds temporary memory even for larger query sets.
    cursor = 0
    while cursor < resamples:
        width = min(1_000, resamples - cursor)
        indices = rng.integers(0, differences.size, size=(width, differences.size))
        bootstrap_means[cursor : cursor + width] = differences[indices].mean(axis=1)
        cursor += width

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return PairedBootstrapResult(
        estimate=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        query_count=int(differences.size),
        resamples=int(resamples),
        seed=int(seed),
    )
