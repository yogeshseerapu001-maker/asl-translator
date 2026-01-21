# =============================================================================
# streamlit_app.py - Beautiful web UI for ASL Translation
# =============================================================================

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from helpers import HandTracker
import config
import time
from collections import deque

# Page configuration
st.set_page_config(
    page_title="ASL Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        padding: 1rem;
        border-radius: 5px;
        background: #f0f2f6;
        margin: 0.5rem 0;
    }
    .stats-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=10)
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# Load model
@st.cache_resource
def load_model():
    """Load the trained ASL model"""
    try:
        model = tf.keras.models.load_model(config.MODEL_FILE)
        return model
    except:
        return None

# Header
st.markdown('<h1 class="main-header">🤟 ASL Translator</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time American Sign Language Recognition</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence to accept prediction"
    )
    
    show_landmarks = st.checkbox("Show Hand Landmarks", value=True)
    show_fps = st.checkbox("Show FPS", value=True)
    
    st.divider()
    
    st.header("📊 Statistics")
    st.metric("Total Predictions", st.session_state.total_predictions)
    
    if st.button("Clear Translation", type="primary"):
        st.session_state.translated_text = ""
        st.session_state.prediction_history.clear()
        st.session_state.total_predictions = 0
        st.rerun()
    
    st.divider()
    
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. **Allow camera access** when prompted
    2. **Show ASL letter** to the camera
    3. **Hold steady** for 1-2 seconds
    4. **Watch** your translation appear!
    
    **Supported Letters:**
    A-Y (excluding J and Z)
    """)
    
    st.divider()
    
    st.header("📚 ASL Alphabet")
    st.image("https://www.startasl.com/wp-content/uploads/sign-language-alphabet.png", 
             caption="ASL Letter Reference", use_container_width=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Placeholder for video
    video_placeholder = st.empty()
    
    # Start/Stop button
    if st.button("🎥 Start Camera", type="primary"):
        model = load_model()
        
        if model is None:
            st.error("❌ Model not found! Please train the model first.")
        else:
            st.success("✅ Model loaded successfully!")
            
            # Initialize camera and tracker
            cap = cv2.VideoCapture(0)
            tracker = HandTracker()
            
            # FPS calculation
            fps_time = time.time()
            fps = 0
            
            # Prediction smoothing
            recent_predictions = deque(maxlen=5)
            
            stop_button = st.button("🛑 Stop Camera")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect hand
                results = tracker.detect(frame, draw=show_landmarks)
                landmarks = tracker.get_landmarks(results)
                
                # Predict if hand detected
                predicted_letter = None
                confidence = 0.0
                
                if np.any(landmarks):
                    landmarks_reshaped = landmarks.reshape(1, -1)
                    prediction = model.predict(landmarks_reshaped, verbose=0)
                    
                    class_idx = np.argmax(prediction)
                    confidence = prediction[0][class_idx]
                    
                    if confidence > confidence_threshold:
                        predicted_letter = config.LETTERS[class_idx]
                        recent_predictions.append(predicted_letter)
                        
                        # Use most common prediction from recent frames
                        if len(recent_predictions) >= 3:
                            most_common = max(set(recent_predictions), 
                                            key=recent_predictions.count)
                            if recent_predictions.count(most_common) >= 3:
                                if (len(st.session_state.translated_text) == 0 or 
                                    st.session_state.translated_text[-1] != most_common):
                                    st.session_state.translated_text += most_common
                                    st.session_state.prediction_history.append(
                                        (most_common, confidence)
                                    )
                                    st.session_state.total_predictions += 1
                
                # Calculate FPS
                if show_fps:
                    current_time = time.time()
                    fps = 1 / (current_time - fps_time)
                    fps_time = current_time
                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display prediction on frame
                if predicted_letter and confidence > confidence_threshold:
                    cv2.putText(frame, f"{predicted_letter} ({confidence*100:.1f}%)", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Show ASL sign", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display in Streamlit
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Small delay
                time.sleep(0.01)
            
            # Cleanup
            cap.release()
            tracker.cleanup()

with col2:
    st.subheader("💬 Translated Text")
    
    # Display translated text in a nice box
    if st.session_state.translated_text:
        st.markdown(f'<div class="prediction-box">{st.session_state.translated_text}</div>', 
                   unsafe_allow_html=True)
        
        # Add space character button
        if st.button("Add Space", use_container_width=True):
            st.session_state.translated_text += " "
            st.rerun()
        
        # Delete last character button
        if st.button("Delete Last", use_container_width=True):
            st.session_state.translated_text = st.session_state.translated_text[:-1]
            st.rerun()
    else:
        st.info("👆 Start the camera and show ASL signs to begin translation!")
    
    st.divider()
    
    # Recent predictions
    st.subheader("📈 Recent Predictions")
    if st.session_state.prediction_history:
        for letter, conf in list(st.session_state.prediction_history)[-5:]:
            st.markdown(f"""
            <div class="confidence-bar">
                <strong>{letter}</strong>: {conf*100:.1f}% confidence
                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 {conf*100}%); 
                            height: 10px; border-radius: 5px; margin-top: 5px;"></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No predictions yet")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using TensorFlow, MediaPipe, and Streamlit</p>
    <p>ASL Translator © 2026</p>
</div>
""", unsafe_allow_html=True)