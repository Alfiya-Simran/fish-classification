# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# App title
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish and let the model predict its category!")

# Load trained model
MODEL_PATH = "cnn_best.h5"  # Path to best saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (replace with actual dataset classes in correct order)
CLASS_NAMES = ['class1', 'class2', 'class3', 'class4', 'class5']  # <-- CHANGE THIS

# Image uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"### Prediction: **{predicted_class}**")
    st.write(f"**Confidence:** {confidence:.2f}")

