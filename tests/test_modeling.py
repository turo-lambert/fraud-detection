"""Tests for the model training and evaluation process."""
from pathlib import Path
from typing import Any, Tuple

import numpy as np

# Define fixtures for setup
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lib.load_config import HPTuningConfig
from lib.model import FraudDetectionModel


@pytest.fixture(scope="module")
def setup_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HPTuningConfig, dict[str, Any]]:
    """Fixture to generate a synthetic dataset and return necessary data for model testing.

    Returns:
        Tuple: Training and test data (features and labels), hyperparameter tuning configuration, and model parameters.
    """
    # Generate a synthetic dataset for fraud detection (binary classification)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.9, 0.1],  # Imbalanced dataset
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dummy hyperparameter config
    hp_config = HPTuningConfig(
        float_hp={"learning_rate": (0.01, 0.3)},
        int_hp={"max_depth": (3, 10)},
        log_hp={"min_child_weight": (1, 5)},
        n_trials=5,  # Reduced for testing speed
    )

    model_params = {"n_estimators": 50, "objective": "binary:logistic"}

    return X_train, X_test, y_train, y_test, hp_config, model_params


def test_train_and_evaluate(
    setup_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HPTuningConfig, dict[str, Any]]
) -> None:
    """Tests the training and evaluation process of the model.

    Args:
        setup_data (tuple): The data and configurations for model training and evaluation.
    """
    X_train, X_test, y_train, y_test, hp_config, model_params = setup_data
    model = FraudDetectionModel(scale_pos_weight=10.0, model_params=model_params, hp_config=hp_config)

    # Training the model
    model.train(X_train, y_train)
    assert model.is_trained, "Model should be marked as trained after training"

    # Evaluating the model
    eval_metrics = model.evaluate(X_test, y_test)
    assert "classification_report" in eval_metrics, "Evaluation metrics should include classification report"
    assert "auc_roc" in eval_metrics, "Evaluation metrics should include AUC-ROC score"
    assert eval_metrics["auc_roc"] > 0.5, "AUC-ROC should be greater than 0.5 for a non-random model"


def test_save_and_load_model(
    setup_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HPTuningConfig, dict[str, Any]]
) -> None:
    """Tests the save and load functionality of the model.

    Args:
        setup_data (tuple): The data and configurations for model training and persistence.
    """
    X_train, _, y_train, _, hp_config, model_params = setup_data
    model = FraudDetectionModel(scale_pos_weight=10.0, model_params=model_params, hp_config=hp_config)
    model.train(X_train, y_train)

    # Saving the model
    filepath = "test_model.pkl"
    model.save_model(filepath)
    assert Path(filepath).exists(), "Model file should exist after saving"

    # Loading the model
    loaded_model = FraudDetectionModel(scale_pos_weight=10.0, model_params=model_params, hp_config=hp_config)
    loaded_model.load_model(filepath)
    assert loaded_model.is_trained, "Model should be marked as trained after loading"

    # Clean up saved model file
    Path.unlink(filepath)


def test_threshold_setting(
    setup_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, "HPTuningConfig", dict[str, Any]]
) -> None:
    """Tests the threshold setting functionality of the model.

    Args:
        setup_data (tuple): The data and configurations for model training and threshold adjustment.
    """
    X_train, _, y_train, _, hp_config, model_params = setup_data
    model = FraudDetectionModel(scale_pos_weight=10.0, model_params=model_params, hp_config=hp_config)
    model.train(X_train, y_train)

    # Check that the optimal threshold is set
    assert 0.0 <= model.threshold <= 1.0, "Threshold should be between 0 and 1 after optimization"
