"""
Utilities: Shared Helper Functions

Responsibility:
    Provide common utility functions used across multiple modules.

Inputs:
    Various (function-specific)

Outputs:
    Helper functions for:
    - Random seed setting
    - Dictionary validation
    - Safe mathematical operations
    - Formatting and printing

Assumptions:
    None

Failure Modes:
    - Division by zero: safe_divide returns fill_value
    - Missing keys: validate_dict_structure raises ValueError
"""

import numpy as np
import random
from typing import Any, Dict


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Purpose: Ensure deterministic results across runs.
    Inputs: seed (int)
    Outputs: None (side effect: sets global seeds)
    """
    random.seed(seed)
    np.random.seed(seed)


def validate_dict_structure(data: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate dictionary contains required keys.

    Purpose: Early validation of data structures.
    Inputs: data dict, list of required keys
    Outputs: True if valid, raises ValueError otherwise
    """
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    return True


def safe_divide(numerator: np.ndarray, denominator: np.ndarray,
                fill_value: float = 0.0) -> np.ndarray:
    """
    Safe division handling divide-by-zero.

    Purpose: Avoid NaN/Inf in division operations.
    Inputs: numerator, denominator arrays, fill_value for zeros
    Outputs: Result array with zeros replaced by fill_value
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = fill_value
    return result


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def print_section_header(title: str, char: str = "=", width: int = 60) -> None:
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def get_class_distribution(labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    Calculate class distribution statistics.

    Purpose: Summarize label counts and percentages.
    Inputs: labels array
    Outputs: Dict with count and percentage per class
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    distribution = {}
    for label, count in zip(unique, counts):
        distribution[int(label)] = {
            'count': int(count),
            'percentage': count / total,
            'percentage_str': format_percentage(count / total)
        }

    return distribution
