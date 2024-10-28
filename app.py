"""Fast API taking care of serving the model."""
import pandas as pd
import shap
from flask import Flask, Request, jsonify, request

from config.config import FE_PATH, MODEL_PATH
from lib.data_processing import FeatureEngineer
from lib.model import FraudDetectionModel

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


def preprocess_observation(fe: FeatureEngineer, observation: list[float]) -> pd.DataFrame:
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

    # Preprocess the input observation
    observation = data
    processed_obs = preprocess_observation(fe, observation)

    # Make prediction
    try:
        prediction = model.predict(processed_obs)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    try:
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(processed_obs)
    except Exception as e:
        return jsonify({"error": f"SHAP error: {str(e)}"}), 500

    # Return the prediction and probability
    return jsonify(
        {
            "prediction": str(prediction[0]),
            "processed_obs": processed_obs.to_dict(orient="list"),
            "shap_values": shap_values.tolist(),
        }
    )


# For local testing and debugging, use:
if __name__ == "__main__":
    app.run(port=5001)
