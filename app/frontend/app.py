import streamlit as st
import requests
from PIL import Image
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="Galaxy Classifier XAI",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Galaxy Classification & Explainable AI")
st.markdown("""
This application predicts the morphological probabilities of a galaxy image using a fine-tuned ResNet-50 model. 
It also provides **Grad-CAM XAI** visualization to understand which parts of the galaxy the neural network focused on to make its decision.
""")

st.sidebar.header("Upload Galaxy Image")
st.sidebar.markdown("Supported formats: PNG, JPG, JPEG")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# FastAPI backend URL (run locally via uvicorn or via Docker network)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict/")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with st.spinner("Analyzing galaxy morphology..."):
        # Send to FastAPI
        try:
            # Re-read file pointer to start
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                with col2:
                    st.subheader("Prediction Results")
                    top_pred = result["top_prediction"]
                    
                    st.success(f"**Primary Morphology:** {top_pred}")
                    
                    st.markdown("### Top 5 Class Probabilities")
                    # Display as a clean horizontal bar chart or metrics
                    probs = result["prediction_probabilities"]
                    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:5]
                    for cls_name, prob in sorted_probs:
                        st.progress(min(prob, 1.0), text=f"{cls_name} ({prob:.2f})")
                        
                st.markdown("---")
                st.subheader("Explainable AI (Grad-CAM)")
                st.info("The Grad-CAM module highlights the regions of the image that were most important in determining the top predicted class.")
                
                # In a real app, the backend would generate and return the Grad-CAM image.
                # For this portfolio skeleton, we note that it would be displayed here.
                st.warning("⚠️ Note: Grad-CAM generation API integration is pending in this template.")
                
            else:
                st.error(f"Error from API: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Could not connect to the backend API. Please make sure the FastAPI server is running (`uvicorn app.api.main:app --reload`).")
