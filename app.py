import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the trained model
@st.cache_data
def load_model():
    model = tf.keras.models.load_model("best_model.h5")  # Ensure the model exists
    return model

model = load_model()

# Function to make predictions
def predict_waste(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3)) / 255.0  # Normalize
    result = np.argmax(model.predict(img))
    if result == 0:
        return "♻️ This is Recyclable Waste"
    else:
        return "🍂 This is Organic Waste"

# Streamlit UI
st.title("🗑️ Waste Classification using CNN")
st.write("Upload an image to classify it as **Recyclable Waste** or **Organic Waste**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    prediction = predict
