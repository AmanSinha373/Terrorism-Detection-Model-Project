import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load models
from joblib import load

vectorizer = load("tfidf_vectorizer.pkl")  # ✅ Works correctly


text_nn_model = tf.keras.models.load_model("text_nn_model.keras")
cnn_model = tf.keras.models.load_model("cnn_image_model.keras")


nb_model = load("naive_bayes_text_model.pkl")  # ✅ Naive Bayes Model (should have predict method)

# Function to classify text
def classify_text():
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showerror("Error", "Please enter text")
        return
    
    text_vectorized = vectorizer.transform([text])
    nb_prediction = nb_model.predict(text_vectorized)[0]
    nn_prediction = text_nn_model.predict(text_vectorized.toarray())[0][0]
    
    nb_result = "Terrorist" if nb_prediction == 1 else "Non-Terrorist"
    nn_result = "Terrorist" if nn_prediction > 0.8 else "Non-Terrorist"
    
    nb_label.config(text=f"Naïve Bayes: {nb_result}")
    nn_label.config(text=f"Neural Network: {nn_result}")

# Function to classify image
def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    img = load_img(file_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = cnn_model.predict(img_array)[0][0]
    result = "Terrorist Image" if prediction > 0.5 else "Non-Terrorist Image"
    
    cnn_label.config(text=f"CNN Prediction: {result}")
    
    # Display image
    img = Image.open(file_path)
    img.thumbnail((150, 150))
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img

# GUI Setup
root = tk.Tk()
root.title("Terrorism Detection Model")
root.geometry("500x500")

# Text Input Box
text_frame = tk.LabelFrame(root, text="Text Classification")
text_frame.pack(pady=10, fill="both", expand=True)
text_entry = tk.Text(text_frame, height=5, width=50)
text_entry.pack()
tk.Button(text_frame, text="Classify Text", command=classify_text).pack()
nb_label = tk.Label(text_frame, text="Naïve Bayes: ")
nb_label.pack()
nn_label = tk.Label(text_frame, text="Neural Network: ")
nn_label.pack()

# Image Input Box
image_frame = tk.LabelFrame(root, text="Image Classification")
image_frame.pack(pady=10, fill="both", expand=True)
tk.Button(image_frame, text="Upload Image", command=classify_image).pack()
cnn_label = tk.Label(image_frame, text="CNN Prediction: ")
cnn_label.pack()
img_label = tk.Label(image_frame)
img_label.pack()

root.mainloop()
