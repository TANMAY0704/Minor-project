from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained Random Forest model from the absolute file path
model = joblib.load('C:/Users/abtsn/OneDrive/Documents/GitHub/minorproject/testcode/random_forest_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()

    # Preprocess the data if necessary (e.g., convert JSON data to a DataFrame)

    # Make predictions using the loaded model
    predictions = model.predict(data)

    # Return the predictions as JSON response
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
