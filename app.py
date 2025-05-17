import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('plant_disease_model.h5')  # Your model file

# Class labels (replace with your actual class names)
class_names = ['Apple___Scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot',
               'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy']  # Add more if needed

# Title
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to identify the disease")

# Upload
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # --- Feature extraction for model expecting 5 features ---
    mean_rgb = img_array.mean(axis=(1, 2))  # shape (1,3)
    std_rgb = img_array.std(axis=(1, 2))    # shape (1,3)
    features = np.concatenate([mean_rgb, std_rgb], axis=1)  # shape (1,6)
    features_5 = features[:, :5]  # Select first 5 features (shape (1,5))

    # Predict using extracted features
    prediction = model.predict(features_5)

    # Get predicted class index and confidence
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    health_status = "Healthy" if "healthy" in class_names[class_index].lower() else "Not Healthy"

    # Medicine/treatment recommendations dictionary
    medicine_advice = {
        'Apple___Scab': 'Use fungicides containing chlorothalonil or myclobutanil.',
        'Apple___Black_rot': 'Apply copper-based fungicides and prune infected branches.',
        'Apple___Cedar_apple_rust': 'Use fungicides and remove nearby cedar trees if possible.',
        'Apple___healthy': 'No treatment needed. Plant is healthy.',
        'Blueberry___healthy': 'No treatment needed. Plant is healthy.',
        'Cherry___Powdery_mildew': 'Apply sulfur-based fungicides and improve air circulation.',
        'Cherry___healthy': 'No treatment needed. Plant is healthy.',
        'Corn___Cercospora_leaf_spot': 'Use fungicides like azoxystrobin and rotate crops.',
        'Corn___Common_rust': 'Apply fungicides such as propiconazole and monitor regularly.',
        'Corn___Northern_Leaf_Blight': 'Use resistant hybrids and fungicides if necessary.',
        'Corn___healthy': 'No treatment needed. Plant is healthy.'
    }

    advice = medicine_advice.get(class_names[class_index], "No specific treatment advice available.")

    st.success(f"Predicted: {class_names[class_index]} ({confidence*100:.2f}%)")
    st.info(f"Health Status: {health_status}")
    st.warning(f"Recommended Treatment: {advice}")
