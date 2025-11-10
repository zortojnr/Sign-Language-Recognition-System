import streamlit as st
import tensorflow as tf
import numpy as np
import string
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="Sign Language Recognition System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-letter {
        font-size: 5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-bar {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Class names (A-Z excluding J and Z)
CLASS_NAMES = list(string.ascii_lowercase.replace('j', '').replace('z', ''))

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'models/sign_language_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None

def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image).astype(float)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Reshape for model input: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=3)
    
    return img_array

def predict_sign(model, image):
    """Make prediction on preprocessed image"""
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence, predictions[0]

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 Sign Language Recognition System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered American Sign Language Letter Recognition</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This application uses a **Convolutional Neural Network (CNN)** 
        to recognize American Sign Language letters.
        
        **Features:**
        - 🎯 99.71% Accuracy
        - 📸 Image Upload Support
        - 🔍 Real-time Predictions
        - 📊 Confidence Scores
        
        **Supported Letters:**
        A-Z (excluding J and Z)
        """)
        
        st.header("🔧 Instructions")
        st.markdown("""
        1. Upload a sign language image
        2. Or use the camera to capture
        3. View the prediction instantly
        4. Check confidence scores
        """)
        
        st.header("📊 Model Info")
        st.markdown("""
        - **Architecture**: CNN
        - **Parameters**: 442,264
        - **Input Size**: 28×28 grayscale
        - **Classes**: 24 letters
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running `python main.py`")
        st.info("The model will be saved to `models/sign_language_model.h5` after training.")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # Image upload options
        upload_option = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Camera"],
            horizontal=True
        )
        
        uploaded_file = None
        
        if upload_option == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a 28x28 grayscale image of a sign language letter"
            )
        else:
            uploaded_file = st.camera_input("Take a picture")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Preprocess and predict
            with st.spinner("Processing image..."):
                processed_image = preprocess_image(image)
                predicted_class, confidence, all_predictions = predict_sign(model, processed_image)
            
            # Display prediction
            predicted_letter = CLASS_NAMES[predicted_class]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Prediction</h2>
                <div class="prediction-letter">{predicted_letter.upper()}</div>
                <h3>Confidence: {confidence * 100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.header("📊 Prediction Details")
        
        if uploaded_file is not None:
            # Top 5 predictions
            st.subheader("Top 5 Predictions")
            top_5_indices = np.argsort(all_predictions)[-5:][::-1]
            
            for idx in top_5_indices:
                letter = CLASS_NAMES[idx]
                conf = all_predictions[idx]
                bar_color = "#667eea" if idx == predicted_class else "#e0e0e0"
                
                st.markdown(f"**{letter.upper()}**")
                st.progress(conf, text=f"{conf * 100:.2f}%")
            
            # Confidence distribution
            st.subheader("Confidence Distribution")
            chart_data = {CLASS_NAMES[i]: float(all_predictions[i]) for i in range(len(CLASS_NAMES))}
            st.bar_chart(chart_data)
            
            # Model information
            st.subheader("ℹ️ Model Information")
            st.info(f"""
            - **Predicted Letter**: {predicted_letter.upper()}
            - **Confidence**: {confidence * 100:.2f}%
            - **Class Index**: {predicted_class}
            """)
        else:
            st.info("👆 Upload an image or use the camera to see predictions here")
            st.image("https://via.placeholder.com/400x300?text=Upload+Image+to+Start", use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Sign Language Recognition System | Powered by TensorFlow & Streamlit</p>
        <p>Model Accuracy: 99.71% | 24 ASL Letters Supported</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

