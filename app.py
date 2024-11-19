import pickle
import pandas as pd
from flask import Flask, request, Response
from collections import OrderedDict
import json

# Load the trained and calibrated model
with open('calibrated_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define label mapping
label_mapping = {
    0: 'XGBRegressor',
    1: 'HUBERREGRESSOR',
    2: 'LinearSVR',
    3: 'LASSO',
    4: 'QUANTILEREGRESSOR',
    5: 'ELASTICNETCV'
}

# Extract feature names expected by the model
if hasattr(model, 'feature_names_in_'):
    expected_features = list(model.feature_names_in_)
else:
    raise ValueError("Feature names not found in the model. Check your training process.")

# Function to correct input feature names
def correct_feature_names(input_features):
    corrected_features = {}
    for feature in expected_features:
        corrected_features[feature] = input_features.get(feature, 0)  # Use 0 for missing features
    return corrected_features

# Flask app initialization
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.json
        if not input_data:
            return Response(json.dumps({"error": "No input data provided"}), status=400, mimetype='application/json')

        response = OrderedDict()  # Use OrderedDict to ensure response order
        for instance_id, features in input_data.items():
            # Correct feature names to match the model's expected input
            corrected_features = correct_feature_names(features)

            # Convert corrected features to a DataFrame
            feature_df = pd.DataFrame([corrected_features])

            # Make predictions
            probabilities = model.predict_proba(feature_df)[0]

            # Map class indices to labels and round probabilities
            instance_prediction = {
                label_mapping[class_idx]: round(prob, 6)
                for class_idx, prob in enumerate(probabilities)
            }

            # Sort predictions by probability values (descending order)
            sorted_prediction = OrderedDict(
                sorted(instance_prediction.items(), key=lambda item: item[1], reverse=True)
            )

            # Add sorted predictions to the response
            response[instance_id] = sorted_prediction

        # Serialize response as JSON manually to preserve order
        return Response(json.dumps(response), status=200, mimetype='application/json')

    except Exception as e:
        # Handle unexpected errors
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)