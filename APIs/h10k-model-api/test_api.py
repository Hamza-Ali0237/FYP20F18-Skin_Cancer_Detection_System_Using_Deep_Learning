# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "/Users/macbook/Downloads/H10K-Dataset/HAM10000_images_part_2/ISIC_0031271.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if 'prediction' in r:
    # Print the predicted label
    print("Predicted Label:", r["prediction"])

# otherwise, print an error message
else:
    print("Error occurred during prediction")
