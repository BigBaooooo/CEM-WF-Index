"""Focused evaluators for reviewer-requested development studies.

The package contains evaluation logic only.  It deliberately does not bundle
the external annotations, candidate tables, or frozen score files.
"""

from .climatenet_portability import evaluate_climatenet_portability
from .matched_scoring import evaluate_matched_scoring
from .physical_metric_ablation import evaluate_physical_metric_ablation
from .statistics import PairedBootstrapResult, paired_bootstrap_interval

__all__ = [
    "PairedBootstrapResult",
    "evaluate_climatenet_portability",
    "evaluate_matched_scoring",
    "evaluate_physical_metric_ablation",
    "paired_bootstrap_interval",
]
