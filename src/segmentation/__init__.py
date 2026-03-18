"""
Segmentation Package: Windowing and Synchronization

Modules:
    - window_data: WindowInfo and WindowedData containers
    - windowing: Sliding window creation with label assignment
"""

from .window_data import WindowInfo, WindowedData
from .windowing import create_all_windows, create_windows_for_subject
