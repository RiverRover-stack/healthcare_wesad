"""
Data Package: Data Loading and Container Classes

Modules:
    - subject_data: SubjectData dataclass
    - loader: WESAD pickle loading functions
"""

from .subject_data import SubjectData
from .loader import load_all_subjects, load_single_subject
