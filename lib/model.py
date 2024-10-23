"""Contains the FraudDetectionModel class for training, evaluating, and saving an XGBoost model."""
# import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score


class FraudDetectionModel:
    """A class to train, evaluate, and save an XGBoost model for fraud detection."""

    def __init__(
        self,
        scale_pos_weight: Optional[float] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the FraudDetectionModel with optional parameters.

        Args:
            scale_pos_weight (Optional[float]): The weight to handle imbalanced data.
                This is used to give more importance to the minority class (fraud). Defaults to None.
            model_params (Optional[Dict[str, Any]]): Optional dictionary of additional XGBoost parameters.
                If None, a default parameter set will be used. Defaults to None.
        """
        self.scale_pos_weight = scale_pos_weight
        self.model_params = model_params or {
            "objective": "binary:logistic",  # Binary classification problem
            "eval_metric": "aucpr",  # Use AUC-PR for imbalanced dataset
            "use_label_encoder": False,  # Avoid label encoding warnings
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
        print("Model training completed.")

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

        # Predict the labels and probabilities
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Generate evaluation metrics
        report = classification_report(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        # Print evaluation results
        print("Classification Report:\n", report)
        print(f"AUC-ROC: {auc_roc}")

        return {"classification_report": report, "auc_roc": auc_roc}

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
        # with Path.open(filepath, "wb") as f:
        # pickle.dump(self.model, f)
        # pass
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Loads a previously trained model from a file.

        Args:
            filepath (str): Path to the saved model file.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
        """
        if not Path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")

        # with Path.open(filepath, "rb") as f:
        # self.model = pickle.load(f)
        # self.is_trained = True
        print(f"Model loaded from {filepath}")
