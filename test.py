import requests

from PIL import Image
import numpy as np

def image_to_feature_vector(image_path, size=(28, 28), normalize=True):
    """
    Convert an image to a feature vector.

    Parameters:
    - image_path (str): Path to the image file.
    - size (tuple): Desired size of the image (width, height). Default is (28, 28).
    - normalize (bool): Whether to normalize pixel values to the range [0, 1]. Default is True.

    Returns:
    - numpy.ndarray: Flattened feature vector of the image.
    """
    # Load the image
    with Image.open(image_path) as img:
        # Convert to grayscale if not already
        img = img.convert('L')

        # Resize the image to the desired size
        img = img.resize(size)

        # Convert image to numpy array
        image_array = np.array(img)

        # Flatten the array
        feature_vector = image_array.flatten()

        return feature_vector

# Example usage
image_path = 'katyhacks/test3.png'
feature_vector = image_to_feature_vector(image_path)
print(feature_vector)



# URL of the Flask API
url = 'http://localhost:5000/predict'

feature_vector_list = feature_vector.tolist()

# Example feature vector
data = {
    'features': feature_vector_list  # Replace with actual feature values
}

# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())