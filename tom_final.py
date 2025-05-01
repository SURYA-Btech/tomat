import streamlit as st
import google.generativeai as genai
from PIL import Image

# 🔑 Set your Gemini API key here
genai.configure(api_key="AIzaSyAN6OqsAZZ8SumP1bmqKxgf06cHwSmzVbc")

# Load Gemini 1.5 Flash model
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit interface
st.title("🍅 Tomato Classifier and Analysis")
st.write("Upload an image of a tomato, and I'll automatically detect if it's a tomato, classify its stage, and provide a detailed analysis.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    # Default prompt: Check if the image is a tomato, provide its stage, and give detailed analysis
    prompt = """
    Is this a tomato? If yes, classify its stage into one of the following categories: Damaged, Old, Unripe, or Ripe. 
    If it is a tomato, also provide a detailed analysis with the following:
    - The percentage of the tomato that is ripe.
    - The percentage of the tomato that is damaged.
    - The percentage of the tomato that is unripe.
    - The percentage of the tomato that is old.
    just answer as straight, dont answer in your AI words
    If it's not a tomato, respond with 'Not a tomato.'
    """

    # Automatically send to Gemini Vision
    with st.spinner("Classifying tomato and generating analysis..."):
        response = model.generate_content([prompt, image])
        st.success("Prediction and analysis generated!")

    # Display the prediction and analysis
    st.markdown("### 🍅 Prediction and Analysis:")
    st.write(response.text)
