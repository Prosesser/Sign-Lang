from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('sign_cnn_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    file = request.files['file']
    
    # Read the image file
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
    
    # Preprocess the image
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0
    
    # Expand dimensions to simulate a batch
    img_batch = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch)
    
    # Find the most probable class
    predicted_class_index = np.argmax(prediction)
    
    # Map the class index to the corresponding letter
    def decode_label(index):
        return chr(ord('A') + index)
    
    predicted_letter = decode_label(predicted_class_index)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': predicted_letter})

if __name__ == '__main__':
    app.run(debug=True)