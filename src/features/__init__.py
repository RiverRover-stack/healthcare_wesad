"""
Features Package: Feature Extraction

Modules:
    - statistical: Basic statistical features
    - temporal: Time-domain features
    - frequency: Spectral features
    - eda: EDA-specific features
    - extractor: Main extraction pipeline
"""

from .extractor import extract_all_features, extract_features_from_window
