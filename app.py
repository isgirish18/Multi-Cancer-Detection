import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io


# Load the trained model
@st.cache_resource
def load_trained_model():
    model_path = "./breast_cancer/BreastCancerMobileNet.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# Function to preprocess the uploaded image
def preprocess_input_image(img):
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    return img_array

# Load model
model = load_trained_model()

# Define class names
class_names = ["class_1", "class_2", "class_3", "class_4", "class_5"]  # Update with actual class names

# Streamlit UI
st.title("Cancer Classification")
st.write("Upload an image to predict its class")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_input_image(image_data)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    
    # Display the prediction
    st.write(f"### Predicted Class: {predicted_class_name}")
