"""
Classifiers: ML Model Training and Prediction

Responsibility:
    Train and evaluate Logistic Regression and Random Forest models.

Inputs:
    Training features/labels, test features

Outputs:
    Trained models, predictions, probabilities

Assumptions:
    - Features are extracted and normalized
    - Class imbalance handled via class_weight

Failure Modes:
    - Convergence issues: LogReg max_iter increased
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED


def create_logistic_regression() -> LogisticRegression:
    """Create Logistic Regression model."""
    return LogisticRegression(
        penalty='l2', C=1.0, class_weight='balanced',
        max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs'
    )


def create_random_forest() -> RandomForestClassifier:
    """Create Random Forest model."""
    return RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        random_state=RANDOM_SEED, n_jobs=-1
    )


def train_and_predict(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, scale: bool = True) -> Tuple:
    """Train model and get predictions."""
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return predictions, probabilities, model, scaler


def get_feature_importance(model: Any, feature_names: list) -> dict:
    """Extract feature importance from trained model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        return {}
    return dict(zip(feature_names, importances))
