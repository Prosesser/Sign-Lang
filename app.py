from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def post(self):
        # Load the trained model
        model = tf.keras.models.load_model('sign_cnn_model.keras')
        
        # Read the uploaded image file
        file = request.files['file']
        
        if file:
            # Convert the file to a numpy array
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Preprocess the image
            img = cv2.resize(img, (28, 28))
            img = img.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = model.predict(img_batch)
            predicted_class_index = np.argmax(prediction)
            
            # Map the class index to the corresponding letter
            def decode_label(index):
                return chr(ord('A') + index)
            
            predicted_letter = decode_label(predicted_class_index)
            
            return jsonify({'prediction': predicted_letter})
        else:
            return jsonify({'error': 'No file provided.'}), 400

class GetData(Resource):
    def post(self):
        # Accept a file upload
        file = request.files['file']
        
        # Check if the file is not empty
        if file and file.filename:
            try:
                # Assuming the file is an Excel file
                df = pd.read_excel(file)
                
                # Convert DataFrame to JSON
                res = df.to_json(orient='records')
                
                # Return the JSON response
                return jsonify(res)
            except Exception as e:
                return jsonify({"error": "Failed to process the file."}), 400
        else:
            return jsonify({"error": "No file provided."}), 400

api.add_resource(GetData, '/api')
api.add_resource(Prediction, '/prediction')

if __name__ == '__main__':
    app.run(debug=True)
