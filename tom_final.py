import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load both models
@st.cache_resource
def load_models():
    tomato_check_model = load_model("vgg16_is_tomato.h5")       # Binary classifier
    tomato_stage_model = load_model("vgg16_tomato_stage.h5")    # 4-class stage classifier
    return tomato_check_model, tomato_stage_model

# Preprocessing function
def prepare_image(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Streamlit UI
st.title("🍅 Tomato Classifier and Analysis")
st.write("Upload an image of a tomato, and get the report,classification of its stage")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    tomato_check_model, tomato_stage_model = load_models()
    processed = prepare_image(image)

    # Stage 1: Check if it's a tomato
    tomato_prob = tomato_check_model.predict(processed)[0][0]
    is_tomato = tomato_prob > 0.5

    if is_tomato:
        st.success(f"Yes, this is a Tomato (Confidence: {tomato_prob*100:.2f}%)")

        # Stage 2: Predict stage
        stage_pred = tomato_stage_model.predict(processed)[0]
        classes = ["Damaged", "Old", "Unripe", "Ripe"]
        stage = classes[np.argmax(stage_pred)]

        st.markdown(f"### Stage Prediction: **{stage}**")
        st.write("**Stage Probabilities:**")
        for i, label in enumerate(classes):
            st.write(f"- {label}: {stage_pred[i]*100:.2f}%")

    else:
        st.error(f"Not a Tomato (Confidence: {(1 - tomato_prob)*100:.2f}%)")
