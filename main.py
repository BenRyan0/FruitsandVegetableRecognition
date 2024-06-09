import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once at the start with caching
def load_model():
    try:
        model = tf.keras.models.load_model("trained_model.h5", compile=False)
        # Compile the model with the correct optimizer
        model.compile(optimizer=tf.keras.optimizers.Adam(),  # Choose appropriate optimizer
                      loss='categorical_crossentropy',  # Choose appropriate loss
                      metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

# Tensorflow Model Prediction
def model_prediction(test_image, top_k=3):
    if model is None:
        st.error("Model not loaded properly.")
        return -1
    
    image = Image.open(test_image)
    image = image.resize((64, 64))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    top_indices = predictions[0].argsort()[-top_k:][::-1]
    top_confidences = predictions[0][top_indices]
    return list(zip(top_indices, top_confidences))  # Return list of tuples (index, confidence)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if test_image:
        st.image(test_image, width=4, use_column_width=True)
    if st.button("Predict"):
        # st.snow()
        st.write("Our Predictions")
        top_k = st.slider("Number of Top Predictions:", 1, 10, 3)  # Slider to select number of top predictions
        result_indices_confidences = model_prediction(test_image, top_k)
        if result_indices_confidences != -1:
            with open("labels.txt") as f:
                content = f.readlines()
            labels = [i.strip() for i in content]
            for idx, confidence in result_indices_confidences:
                st.success(f"{labels[idx]}: {confidence*100:.2f}% confidence")
