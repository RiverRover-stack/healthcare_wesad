"""
Evaluation Package: Metrics and Splitting

Modules:
    - metrics: Classification metrics computation
    - splitting: LOSO and fixed subject-level splits
"""

from .metrics import compute_metrics, aggregate_fold_metrics, print_aggregated_summary
from .splitting import loso_split, fixed_subject_split, get_split_info
