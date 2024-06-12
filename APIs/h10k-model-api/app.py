from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import date

# Import necessary libraries

# Define the class labels
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('APIs/h10k-model-api/SCDSNet-H10K_Model-1.keras')


# Define a function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    print(request.json)
    data = request.json
    # Check if an image is sent
    if 'image' not in data:
        return jsonify({'error': 'No image found in request'}), 400
    
    # Get the image file from the request
    image_file = request.files

    # Save the image to a temporary location
    image_path = data["image"]
#    image_file.save(image_path)

    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(processed_image)
    
    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    # Get the current date
    current_date = date.today().strftime("%Y-%m-%d")

    ## Get the predicted class label and probability
    prediction_probability = predictions[0][predicted_class_index]
    prediction_probability_str = str(prediction_probability)
    print(prediction_probability_str)

    # Return the prediction result with current date, class label, and probability
    return jsonify({'success': True, 'lesion_type': predicted_class_label, 'probability': prediction_probability_str, 'report_generate_date': current_date}), 200


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
