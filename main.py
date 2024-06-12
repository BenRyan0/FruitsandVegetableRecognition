import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# Read the image file for the favicon
file_path = "images/plant_.png"  # Replace with your image file path
with open(file_path, "rb") as f:
    img_bytes = f.read()

# Encode image to base64
encoded_img = base64.b64encode(img_bytes).decode()

# Set the page configuration
st.set_page_config(
    page_title="Snap-n-Sprout",  # Title of the browser tab
    page_icon=f"data:image/png;base64,{encoded_img}",  # Favicon as base64 encoded image
    layout="centered",  # Layout can be "centered" or "wide"
)


# Inject custom CSS
st.markdown("""
    <style>
   @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');

            
    
    *{
        font-family: "Poppins", sans-serif;
    }
     .stButton>button {
        background-color: #02A367;
        border: 2px solid #02A367;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        font-weight: 700;
        margin: 3px 2px;
        cursor: pointer;
        border-radius: 10px;
        transition: .3s ease-in-out; 
    }
    .stButton>button:hover{
        border: 2px solid #02A367;
        background-color: #fff;
        color: #02A367

    }
    .stButton>button:active{
        background-color: #02A367;
        border: 2px solid #02A367;
        color: white;
    
    }
  
    </style>
    """, unsafe_allow_html=True)


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
    st.header("FRUITS & VEGETABLES RECOGNITION")
    image_path = "images/Main_BG.png"
    st.image(image_path)
    st.markdown("""
    Snap-n-Sprout's mission is to empower everyone to become confident explorers in the fresh produce aisle. Through our image recognition technology and engaging resources, we strive to demystify fruits and vegetables, promote healthy eating habits, reduce food waste, and cultivate a love for the vibrant world of fresh food.
                
        ### How It Works
    1. **Upload Image:** Just open the app and upload a picture of your mystery fruit or vegetable.
    2. **Analysis:** Our powerful image recognition technology instantly analyzes the picture and reveals the produce's identity.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Prediction** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)


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
    st.markdown("[Dataset Used In Training](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)")
    # st.markdown("[Code Github Repository](https://github.com/BenRyan0/ITBAN3_Plant_Desease_Detection.git)")

    st.header("Our Team")
    
    team_members = [
        {"name": "SUMINGUIT, ABDU RASHID", "image": "team/abdu.png","section": "ITBAN3 : IT3A"},
        {"name": "CASILA, SHENA MAE",  "image": "team/Sheena.png","section": "ITBAN3 : IT3A"},
    ]

    cols = st.columns(2)  # Adjust the number of columns based on the number of team members

    for idx, member in enumerate(team_members):
        with cols[idx % 2]:  # Loop through the columns
            st.image(member["image"], width=300)
            st.subheader(member["name"])
            st.markdown(member["section"])
          
        

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
