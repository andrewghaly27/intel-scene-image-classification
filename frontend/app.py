import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title='Intel Scene Classifier', layout='centered')

st.title("Intel Scene Image Classifier")
st.write(
    "Upload a scene image. The app sends it to a FastAPI backend and returns the top predictions."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    
    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(
            API_URL,
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        )
        
        if response.status_code == 200:
            data = response.json()
            preds = data["predictions"]
            
            st.subheader("Predictions")
            st.success(f"Top Predictions: {preds[0]['class_name']}")
            
            st.subheader("Top 3 Predictions")
            for i, pred in enumerate(preds,start=1):
                st.write(f"{i}. **{pred['class_name']}** — {pred['confidence'] * 100:.2f}%")
        else:
            st.error("Prediction failed. Check whether the FastAPI server is running.")