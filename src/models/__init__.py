"""
Models Package: ML Models and Baselines

Modules:
    - classifiers: LogReg and RandomForest models
    - baselines: Heuristic baseline detectors
"""

from .classifiers import create_logistic_regression, create_random_forest, train_and_predict
from .baselines import run_all_baselines
