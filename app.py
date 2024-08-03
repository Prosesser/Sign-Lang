from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('katyhacks/random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Ensure 'features' is present in the JSON data
        if 'features' not in data:
            return jsonify({'error': 'No features found in request'}), 400
        
        # Convert features to a numpy array and reshape for prediction
        features = np.array(data['features']).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(features)
        
        # Return the prediction in JSON format
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
