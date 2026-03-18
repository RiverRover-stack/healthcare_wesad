"""
Window Data: Data Containers for Windowed Data

Responsibility:
    Define dataclasses for window metadata and windowed data storage.

Inputs:
    Window parameters during construction

Outputs:
    WindowInfo and WindowedData dataclass objects

Assumptions:
    - Windows are fixed-duration segments
    - Each window has a binary label

Failure Modes:
    - Empty windows: num_windows will be 0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WindowInfo:
    """Metadata for a single window."""

    subject_id: str
    window_idx: int
    start_sample: int
    end_sample: int
    start_time_sec: float
    end_time_sec: float
    label_binary: int
    label_purity: float
    is_valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class WindowedData:
    """Container for all windowed data from a subject."""

    subject_id: str
    windows: Dict[str, np.ndarray] = field(default_factory=dict)
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    window_info: List[WindowInfo] = field(default_factory=list)
    num_windows: int = 0
    num_baseline: int = 0
    num_stress: int = 0
