import os

# Explicitly install the google-generativeai library if not already installed
os.system('pip install google-generativeai')

import streamlit as st
import google.generativeai as genai
from PIL import Image

# Configure the Gemini API key
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Load Gemini 1.5 Flash model
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit interface
st.title("🍅 Tomato Classifier and Detailed Analysis")
st.write("Upload an image of a tomato, and I'll automatically detect if it's a tomato, classify its stage, and provide a detailed analysis.")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# When an image is uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    # Define the prompt to detect tomato and provide detailed analysis
    prompt = """
    Is this a tomato? If yes, classify its stage into one of the following categories: Damaged, Old, Unripe, or Ripe. 
    If it is a tomato, also provide a detailed analysis with the following:
    - The percentage of the tomato that is ripe.
    - The percentage of the tomato that is damaged.
    - The percentage of the tomato that is unripe.
    - The percentage of the tomato that is old.
    - Describe the texture of the tomato's skin: Is it smooth, rough, or wrinkled?
    - What is the color distribution of the tomato? Provide the percentage of red, green, yellow, etc.
    - Describe the size/shape of the tomato: Is it small, fully grown, or misshaped?
    - Does the tomato show any signs of rot, blemishes, or health issues? If yes, describe them.
    Provide the analysis in a clear and easy-to-understand format.
    If it's not a tomato, respond with 'Not a tomato.'
    """

    # Automatically send to Gemini Vision
    with st.spinner("Classifying tomato and generating detailed analysis..."):
        try:
            response = model.generate_content([prompt, image])
            st.success("Prediction and detailed analysis generated!")
            st.markdown("### 🍅 Prediction and Detailed Analysis:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error generating content: {str(e)}")
