from flask import Flask, request, jsonify
import numpy as np
import joblib
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load Text Classification Models
nb_model = joblib.load(r"D:\Project\naive_bayes_text_model.pkl")
text_nn_model = load_model(r"D:\Project\text_nn_model.keras")
vectorizer = joblib.load(r"D:\Project\tfidf_vectorizer.pkl")

# Load Image Classification Model
image_model = load_model(r"D:\Project\cnn_image_model.keras")

# Set Image Input Size
IMG_SIZE = 128

###TEXT CLASSIFICATION API ###
import time  # Add this at the top

@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_path = f"temp_{int(time.time())}.jpg"  # Use unique name to avoid conflicts
    image_file.save(image_path)

    # Debugging: Check if file exists
    if not os.path.exists(image_path):
        return jsonify({"error": "Flask failed to save the image."}), 500

    # Debugging: Check file size
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        return jsonify({"error": "Saved file is empty. Check upload format."}), 400

    # Try reading the image
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "OpenCV failed to load the image. Try PNG or JPG."}), 400

    # Preprocess Image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = image_model.predict(img)[0][0]
    label = "Terrorist" if prediction > 0.5 else "Non-Terrorist"

    # Clean up
    os.remove(image_path)

    return jsonify({
        "Image_Prediction": label,
        "Confidence": float(prediction)
    })


### API HOME ###
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Terrorism Detection API is running!"})

# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
