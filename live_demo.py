# =============================================================================
# live_demo.py
# =============================================================================

import cv2
import numpy as np
from tensorflow import keras
from helpers import HandTracker
import config

def run_live_prediction():
    """Real-time ASL recognition using webcam"""
    
    # load the trained model
    print("Loading model...")
    try:
        model = keras.models.load_model(config.MODEL_FILE)
        print("Model loaded successfully!\n")
    except:
        print("ERROR: Could not load model!")
        print("Make sure you've trained the model first.")
        return
    
    # setup hand tracker
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    print("Starting live prediction...")
    print("Press Q to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # mirror the frame
        frame = cv2.flip(frame, 1)
        
        # detect hand
        results = tracker.detect(frame, draw=True)
        landmarks = tracker.get_landmarks(results)
        
        # predict if we have a hand
        if np.any(landmarks):
            # reshape for model input
            landmarks_reshaped = landmarks.reshape(1, -1)
            
            # get prediction
            prediction = model.predict(landmarks_reshaped, verbose=0)
            
            # find the most likely letter
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            predicted_letter = config.LETTERS[class_idx]
            
            # only show if confidence is decent
            if confidence > 0.7:
                # show prediction on screen
                text = f"Letter: {predicted_letter}"
                conf_text = f"Confidence: {confidence*100:.1f}%"
                
                cv2.putText(frame, text, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, conf_text, (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            else:
                cv2.putText(frame, "Low confidence", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
        else:
            cv2.putText(frame, "Show your hand", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('ASL Translator - Live Demo', frame)
        
        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()
    print("\nDemo stopped.")
