import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Define model paths and corresponding class names
MODEL_PATHS = {
    "brain_cancer": "./brain_cancer/BrainCancerMobileNet.h5",
    "breast_cancer": "./breast_cancer/BreastCancer-MobileNetV3.h5",
    "cervical_cancer": "./cervical_cancer/CervicalCancerMobileNet.h5",
    "kidney_cancer": "./kidney_cancer/KidneyCancerMobileNet.h5",
    "lung_colon_cancer": "./lung_and_colon_cancer/LungCancerMobileNet.h5",
    "lymphoma": "./lymphoma_cancer/Lymphoma  - MobileNetV3.h5",
    "oral_cancer": "./oral_cancer/OralCancerMobileNet.h5"
}

CLASS_NAMES = {
    "brain_cancer": ["brain_glioma", "brain_menin", "brain_tumor"],
    "breast_cancer": ["breast_benign", "breast_malignant"],
    "cervical_cancer": ["Cervix_dyk", "Cervix_koc", "Cervix_mep", "Cervix_pab", "Cervix_sfi"],
    "kidney_cancer": ["kidney_normal", "kidney_tumor"],
    "lung_colon_cancer": ["colon_aca", "colon_bnt", "lung_aca", "lung_bnt", "lung_scc"],
    "lymphoma": ["lymph_cll", "lymph_fl", "lymph_mcl"],
    "oral_cancer": ["oral_normal", "oral_scc"]
}

# Load trained models and cache them
@st.cache_resource
def load_models():
    models = {}
    for cancer_type, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[cancer_type] = tf.keras.models.load_model(path)
    return models

models = load_models()

def preprocess_input_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    return img_array

# Streamlit UI
st.title("Cancer Classification")
st.write("Upload an image to predict its class")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    img_array = preprocess_input_image(image_data)
    
    best_prediction = (None, None, 0)  # (cancer_type, predicted_class_name, confidence)
    
    for cancer_type, model in models.items():
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        predicted_class_name = CLASS_NAMES[cancer_type][predicted_class_index]
        
        if confidence > best_prediction[2]:
            best_prediction = (cancer_type, predicted_class_name, confidence)
    
    if best_prediction[0]:
        st.write(f"### Predicted Cancer Type: {best_prediction[0].replace('_', ' ').title()}")
        st.write(f"### Predicted Class: {best_prediction[1]}")
        st.write(f"### Confidence Score: {best_prediction[2]:.5f}")
    else:
        st.write("No models detected or unable to classify the image.")
