"""Modelling code for the fraud detection model."""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """A class to represent the fraud detection model."""

    def __init__(self, scale_pos_weight: Optional[float] = None, model_params: Optional[Dict[str, Any]] = None) -> None:
        """Initializes the FraudDetectionModel with optional parameters.

        Args:
            scale_pos_weight (Optional[float]): The weight to handle imbalanced data.
                This is used to give more importance to the minority class (fraud). Defaults to None.
            model_params (Optional[Dict[str, Any]]): Optional dictionary of additional XGBoost parameters.
                If None, a default parameter set will be used. Defaults to None.
        """
        self.scale_pos_weight = scale_pos_weight
        self.model_params = model_params or {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "use_label_encoder": False,
        }
        if self.scale_pos_weight:
            self.model_params["scale_pos_weight"] = scale_pos_weight

        self.model = xgb.XGBClassifier(**self.model_params)
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the XGBoost model using the provided training data.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Model training completed.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
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

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        logger.info("Classification Report:\n%s", report)
        logger.info(f"AUC-ROC: {auc_roc}")

        return {"classification_report": report, "auc_roc": auc_roc}

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50) -> None:
        """Optimizes hyperparameters using Optuna.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            n_trials (int): Number of trials for Optuna optimization. Defaults to 50.

        """
        # Split the data into a training and validation set for tuning
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Define the Optuna objective function
        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "use_label_encoder": False,
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train_split, y_train_split)
            y_pred_val = model.predict_proba(X_val_split)[:, 1]

            return roc_auc_score(y_val_split, y_pred_val)

        # Run the optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Update model parameters and reinitialize the model with the best params
        self.model_params.update(study.best_params)
        self.model = xgb.XGBClassifier(**self.model_params)
        logger.info("Best parameters found by Optuna: %s", study.best_params)

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

        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """Saves the trained model to a file using pickle.

        Args:
            filepath (str): Path where the model should be saved.
        """
        with Path.open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Loads a previously trained model from a file.

        Args:
            filepath (str): Path to the saved model file.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
        """
        if not Path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")

        with Path.open(filepath, "rb") as f:
            self.model = pickle.load(f)
            self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
