"""Modelling code for the fraud detection model."""
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from lib.load_config import HPTuningConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """A class to represent the fraud detection model."""

    def __init__(self, scale_pos_weight: float, model_params: dict[str, str], hp_config: HPTuningConfig) -> None:
        """Initializes the FraudDetectionModel with parameters.

        Args:
            scale_pos_weight (float): The weight to handle imbalanced data.
                This is used to give more importance to the minority class (fraud).
            model_params (dict[str, str]): Dictionary of additional XGBoost parameters.
            hp_config (HPTuningConfig): Configuration for hyperparameter tuning.
        """
        self.scale_pos_weight = scale_pos_weight
        self.model_params = model_params
        self.hp_config = hp_config
        self.model_params["scale_pos_weight"] = scale_pos_weight
        self.model = xgb.XGBClassifier(**self.model_params)
        self.is_trained = False
        self.threshold = 0.5

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the XGBoost model using the provided training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self._optimize_hyperparameters(X_train, y_train)
        self.model.fit(X_train, y_train)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        self._find_optimal_threshold(y_val, y_pred_proba)
        self.is_trained = True
        logger.info("Model training completed.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        """Evaluates the model on the test set and returns the evaluation metrics.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            Dict[str, Any]: A dictionary containing the classification report and AUC-ROC score.

        Raises:
            Exception: If the model is not trained before evaluation.
        """
        if not self.is_trained:
            raise Exception("Model has not been trained yet.")

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        report = classification_report(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        logger.info("Classification Report:\n%s", report)
        logger.info(f"AUC-ROC: {auc_roc}")

        return {"classification_report": report, "auc_roc": auc_roc}

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Optimizes hyperparameters using Optuna.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        # Split the data into a training and validation set for tuning
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Define the Optuna objective function
        def objective(trial: optuna.Trial) -> float:
            float_hp = {
                hp_name: trial.suggest_float(hp_name, low=bounds[0], high=bounds[1])
                for hp_name, bounds in dict(self.hp_config.float_hp).items()
            }
            int_hp = {
                hp_name: trial.suggest_int(hp_name, low=bounds[0], high=bounds[1])
                for hp_name, bounds in dict(self.hp_config.int_hp).items()
            }
            categorical_hp = {
                hp_name: trial.suggest_float(hp_name, low=bounds[0], high=bounds[1], log=True)
                for hp_name, bounds in dict(self.hp_config.log_hp).items()
            }

            params = {
                **self.model_params,
                **float_hp,
                **int_hp,
                **categorical_hp,
            }

            model = xgb.XGBClassifier(**params)
            y_pred_proba = cross_val_predict(model, X_train, y_train, cv=skf, method="predict_proba")[:, 1]

            return roc_auc_score(y_train, y_pred_proba)

        # Run the optimization
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        study.optimize(objective, n_trials=self.hp_config.n_trials)

        # Update model parameters and reinitialize the model with the best params
        self.model_params.update(study.best_params)
        self.model = xgb.XGBClassifier(**self.model_params)
        logger.info("Best parameters found by Optuna: %s", study.best_params)

    def _find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Finds the optimal classification threshold for F1 score."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"Optimal threshold for F1 score: {optimal_threshold}")
        self.threshold = optimal_threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions on new data.

        Args:
            X (np.ndarray): Features of the new data.

        Returns:
            np.ndarray: Predictions (0 -> Non-fraudulent, 1 -> Fraudulent).

        Raises:
            Exception: If the model has not been trained before making predictions.
        """
        if not self.is_trained:
            raise Exception("Model has not been trained yet.")

        y_pred_proba = self.model.predict_proba(X)[:, 1]

        return (y_pred_proba >= self.threshold).astype(int)

    def save_model(self, filepath: str) -> None:
        """Saves the entire FraudDetectionModel instance to a file using pickle.

        Args:
            filepath (str): Path where the model should be saved.
        """
        with Path(filepath).open("wb") as f:
            pickle.dump(self, f)  # Save the entire instance
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "FraudDetectionModel":
        """Loads a previously saved FraudDetectionModel instance from a file.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            FraudDetectionModel: The loaded FraudDetectionModel instance.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"No model found at {filepath}")

        with Path(filepath).open("rb") as f:
            model_instance = pickle.load(f)  # Load the entire instance
        logger.info(f"Model loaded from {filepath}")
        return model_instance


def load_latest_model(directory: str) -> FraudDetectionModel:
    """Loads the most recent model from a given directory.

    Args:
        directory (str): Path to the directory containing model files.

    Returns:
        FraudDetectionModel: The loaded model.
    """
    model_dir = Path(directory)
    model_files = sorted(model_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not model_files:
        raise FileNotFoundError("No model files found in the specified directory.")

    latest_model_path = model_files[0]
    return FraudDetectionModel.load_model(latest_model_path)
