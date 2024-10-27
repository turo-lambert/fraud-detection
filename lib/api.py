"""API."""
import numpy as np
import pandas as pd
from flask import Flask, Request, jsonify, request

from lib.data_processing import FeatureEngineer
from lib.model import FraudDetectionModel

MODEL_PATH = "models/fraud_detection/fraud_detection_model.pkl"
FE_PATH = "models/feature_engineering/feature_engineer.pkl"


app = Flask(__name__)

try:
    model = FraudDetectionModel.load_model(MODEL_PATH)
except FileNotFoundError:
    model = None
    app.logger.error("No model found. Make sure a model file exists in the specified directory.")

try:
    fe = FeatureEngineer.load_model(FE_PATH)
except FileNotFoundError:
    fe = None
    app.logger.error("No Feature Engineering instance found. Make sure a model file exists in the specified directory.")


def preprocess_observation(fe: FeatureEngineer, observation: list[float]) -> np.ndarray:
    """Preprocesses a single observation before inference.

    Args:
        fe (FeatureEngineer): Feature engineering instance to process the observation.
        observation (list[float]): Raw observation data as a list of features.

    Returns:
        np.ndarray: Processed observation ready for prediction.
    """
    # Placeholder for actual preprocessing or feature engineering steps
    processed_observation = pd.DataFrame.from_dict(observation)
    processed_observation = fe.transform(processed_observation)
    processed_observation = np.array(processed_observation).reshape(1, -1)
    return processed_observation


@app.route("/score", methods=["POST"])
def score() -> Request:
    """Scores a single observation using the fraud detection model.

    Returns:
        Response: JSON response with prediction and probability.
    """
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    # Get JSON data from request
    data = request.get_json()
    # TODO: pydantic schema to validate data input

    # Preprocess the input observation
    observation = data
    processed_obs = preprocess_observation(fe, observation)
    print(processed_obs)

    # Make prediction
    try:
        prediction = model.predict(processed_obs)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    # Return the prediction and probability
    return jsonify({"prediction": str(prediction[0])})


# For local testing and debugging, use:
if __name__ == "__main__":
    app.run()
