import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained CNN model
model = load_model("plant_disease_cnn_model.h5")

# Define class names (update these with your actual folder names from training)
class_names = ['Apple___Black_rot', 'Apple___Healthy', 'Corn___Common_rust', 'Corn___Healthy']  # example, update to 38 if needed

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Leaf Image', use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show result
    health_status = "Healthy" if "healthy" in predicted_class.lower() else "Not Healthy"

    # Medicine/treatment recommendations dictionary
    medicine_advice = {
        'Apple___Black_rot': 'Use fungicides containing chlorothalonil or myclobutanil.',
        'Apple___Healthy': 'No treatment needed. Plant is healthy.',
        'Corn___Common_rust': 'Apply fungicides such as propiconazole and monitor regularly.',
        'Corn___Healthy': 'No treatment needed. Plant is healthy.'
    }

    advice = medicine_advice.get(predicted_class, "No specific treatment advice available.")

    st.success(f"🌱 Predicted Disease: **{predicted_class}**")
    st.info(f"Health Status: {health_status}")
    st.warning(f"Recommended Treatment: {advice}")
