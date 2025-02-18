import streamlit as st
import time
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from joblib import load
import streamlit.components.v1 as components
import random

# Load models
from joblib import load

# Load models
vectorizer = load("tfidf_vectorizer.pkl")
nb_model = load("naive_bayes_text_model.pkl") 
text_nn_model = tf.keras.models.load_model("text_nn_model.keras")
cnn_model = tf.keras.models.load_model("cnn_image_model.keras")

# Streamlit UI
st.title("🛡️ Terrorism Detection Model")
# Themed Animation
st.markdown(
    """
    <style>
    @keyframes flicker {
        0% { opacity: 1; }
        50% { opacity: 0.1; }
        100% { opacity: 1; }
    }
    .hacker-text {
        font-size: 30px;
        color: #00FF00;
        font-family: 'Courier New', monospace;
        text-shadow: 0px 0px 5px #00FF00;
        animation: flicker 1s infinite;
    }
    .background {
        background-color: black;
        color: #00FF00;
        font-family: 'Courier New', monospace;
    }
    </style>
    <div class="background">
    <h1 class="hacker-text">⚠️  System Activated ⚠️</h1>
    <p class="hacker-text">Tracking Terrorist Activities...</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Simulated Loading Effect
with st.spinner("Initializing ..."):
    time.sleep(3)

st.success("✅ Secure System Ready!")

# ---- TEXT CLASSIFICATION ----
st.header("📜 Text Scan")
user_text = st.text_area("🔍 Enter a message for analysis:")
if st.button("🕵️ Scan Text"):
    if user_text.strip():
        text_vectorized = vectorizer.transform([user_text])
        
        nb_prediction = nb_model.predict(text_vectorized)[0]
        nn_prediction = text_nn_model.predict(text_vectorized.toarray())[0][0]

        st.markdown(
            f"<h3 class='hacker-text'>📡 Report:</h3>",
            unsafe_allow_html=True
        )
        st.write(f"🧠 **Naïve Bayes Analysis:** {'🚨 ALERT: Possible Threat' if nb_prediction == 1 else '✅ Safe Communication'}")
        st.write(f"🤖 **Neural Network Analysis:** {'🚨 ALERT: Possible Threat' if nn_prediction > 0.8 else '✅ Safe Communication'}")

    else:
        st.warning("⚠️ Please enter some text.")

# ---- IMAGE CLASSIFICATION ----
st.header("🖼️ Image Scan")
uploaded_image = st.file_uploader("Upload an image for analysis:", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="🖼️ Uploaded Image", width=250)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = cnn_model.predict(img_array)[0][0]
    result = "🚨 ALERT: Terrorist Image Detected!" if prediction > 0.5 else "✅ No Threat Detected"

    # FBI Terminal-Style Output
    st.markdown(
        f"<h3 class='hacker-text'>📡 Image Report:</h3>",
        unsafe_allow_html=True
    )
    st.write(f"📷 **CNN Analysis:** {result}")

