"""Module for dynamic configuration variables of the project."""
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
MODEL_PATH = "models/fraud_detection/fraud_detection_model.pkl"
FE_PATH = "models/feature_engineering/feature_engineer.pkl"
