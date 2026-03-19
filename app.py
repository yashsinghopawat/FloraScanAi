"""
Streamlit Web App for Plant Disease Detection
Interactive UI for uploading images and getting predictions
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 10px 0;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Treatment recommendations dictionary
TREATMENT_RECOMMENDATIONS = {
    'Apple___Apple_scab': {
        'disease': 'Apple Scab',
        'severity': 'Moderate',
        'treatment': [
            'Remove and destroy infected leaves',
            'Apply fungicide (Captan or Sulfur)',
            'Prune to improve air circulation',
            'Avoid overhead watering'
        ],
        'prevention': 'Plant resistant varieties, ensure proper spacing'
    },
    'Apple___Black_rot': {
        'disease': 'Apple Black Rot',
        'severity': 'Severe',
        'treatment': [
            'Prune infected branches immediately',
            'Apply copper-based fungicide',
            'Remove mummified fruits',
            'Improve drainage'
        ],
        'prevention': 'Regular pruning, remove dead wood'
    },
    'Tomato___Late_blight': {
        'disease': 'Late Blight',
        'severity': 'Critical',
        'treatment': [
            'Apply fungicide immediately (Mancozeb)',
            'Remove and destroy infected plants',
            'Improve drainage',
            'Avoid evening watering'
        ],
        'prevention': 'Plant resistant varieties, monitor weather'
    },
    'Apple___healthy': {
        'disease': 'Healthy Plant',
        'severity': 'None',
        'treatment': ['No treatment needed - plant is healthy!'],
        'prevention': 'Continue regular care and monitoring'
    },
    'Tomato___healthy': {
        'disease': 'Healthy Plant',
        'severity': 'None',
        'treatment': ['No treatment needed - plant is healthy!'],
        'prevention': 'Continue regular care and monitoring'
    },
    'Corn_(maize)___healthy': {
        'disease': 'Healthy Plant',
        'severity': 'None',
        'treatment': ['No treatment needed - plant is healthy!'],
        'prevention': 'Continue regular care and monitoring'
    },
    'default': {
        'disease': 'Unknown Disease',
        'severity': 'Unknown',
        'treatment': [
            'Consult agricultural expert',
            'Send sample to plant pathology lab',
            'Isolate affected plants'
        ],
        'prevention': 'Regular monitoring and good agricultural practices'
    }
}

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    try:
        model = keras.models.load_model('best_model.h5')
        class_names = np.load('class_names.npy', allow_pickle=True).item()
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(img):
    """Preprocess uploaded image for prediction"""
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_severity_color(severity):
    """Return color based on severity level"""
    colors = {
        'None': '#4CAF50',      # Green
        'Moderate': '#FF9800',  # Orange
        'Severe': '#F44336',    # Red
        'Critical': '#D32F2F',  # Dark Red
        'Unknown': '#757575'    # Grey
    }
    return colors.get(severity, '#757575')

def main():
    # Header
    st.markdown('<p class="main-header">🌿 Plant Disease Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a plant leaf image to detect diseases and get treatment recommendations</p>', unsafe_allow_html=True)
    
    # Load model
    model, class_names = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first using train_model.py")
        st.info("👉 Run: python train_model.py")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            This AI system detects plant diseases from leaf images using Deep Learning (CNN).
            
            **Features:**
            - 38+ disease classes
            - 95%+ accuracy
            - Treatment recommendations
            - Real-time predictions
            
            **Supported Plants:**
            - Apple, Tomato, Corn
            - Potato, Grape, Cherry
            - And more...
            """
        )
        
        st.header("📊 Model Info")
        st.success(f"✅ Model Loaded\n\n**Classes:** {len(class_names)}\n\n**Architecture:** EfficientNetB0")
        
        st.header("🎯 How to Use")
        st.markdown("""
        1. Upload a clear leaf image
        2. Wait for prediction
        3. View disease diagnosis
        4. Follow treatment steps
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Predict button
            if st.button('🔍 Detect Disease', type='primary', use_container_width=True):
                with st.spinner('Analyzing image...'):
                    # Preprocess and predict
                    img_array = preprocess_image(image)
                    predictions = model.predict(img_array, verbose=0)[0]
                    
                    # Get top 3 predictions
                    top_3_idx = np.argsort(predictions)[-3:][::-1]
                    
                    # Best prediction
                    best_idx = np.argmax(predictions)
                    predicted_class = class_names[best_idx]
                    confidence = predictions[best_idx] * 100
                    
                    # Get treatment info
                    treatment_info = TREATMENT_RECOMMENDATIONS.get(
                        predicted_class,
                        TREATMENT_RECOMMENDATIONS['default']
                    )
                    
                    # Store in session state
                    st.session_state['prediction_done'] = True
                    st.session_state['treatment_info'] = treatment_info
                    st.session_state['confidence'] = confidence
                    st.session_state['top_3_idx'] = top_3_idx
                    st.session_state['predictions'] = predictions
                    st.session_state['class_names'] = class_names
    
    with col2:
        st.header("📋 Diagnosis Results")
        
        if 'prediction_done' in st.session_state and st.session_state['prediction_done']:
            treatment_info = st.session_state['treatment_info']
            confidence = st.session_state['confidence']
            
            # Disease info
            severity_color = get_severity_color(treatment_info['severity'])
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="margin-top:0;">🔬 Detected Disease</h3>
                <h2 style="color:{severity_color}; margin:10px 0;">{treatment_info['disease']}</h2>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                <p><strong>Severity:</strong> <span style="color:{severity_color}; font-weight:bold;">{treatment_info['severity']}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Treatment recommendations
            st.subheader("💊 Treatment Recommendations")
            for i, step in enumerate(treatment_info['treatment'], 1):
                st.markdown(f"**{i}.** {step}")
            
            # Prevention
            st.subheader("🛡️ Prevention")
            st.info(treatment_info['prevention'])
            
            # Top 3 predictions
            st.subheader("📊 Alternative Possibilities")
            top_3_idx = st.session_state['top_3_idx']
            predictions = st.session_state['predictions']
            class_names = st.session_state['class_names']
            
            for idx in top_3_idx:
                disease_name = class_names[idx].replace('___', ' - ').replace('_', ' ')
                conf = predictions[idx] * 100
                st.progress(conf / 100, text=f"{disease_name}: {conf:.2f}%")
            
        else:
            st.info("👆 Upload an image and click 'Detect Disease' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Built with ❤️ using TensorFlow & Streamlit | Plant Disease Detection System</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
