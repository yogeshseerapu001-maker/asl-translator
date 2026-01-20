
# =============================================================================
# helpers.py
# =============================================================================

import cv2 
import numpy as np 
import mediapipe as mp 
import os
import config

class HandTracker:
    """Wrapper around MediaPipe for easier hand detection"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # initialize the hand detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.DETECTION_CONF,
            min_tracking_confidence=config.TRACKING_CONF
        )
    
    def detect(self, img, draw=True):
        """Find hands in the image"""
        # mediapipe needs RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        # draw landmarks if requested
        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return results
    
    def get_landmarks(self, results):
        """Extract landmark coordinates as flat array"""
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            coords = []
            for landmark in hand.landmark:
                coords.append(landmark.x)
                coords.append(landmark.y)
                coords.append(landmark.z)
            return np.array(coords)
        else:
            # return zeros if no hand detected
            return np.zeros(config.INPUT_SIZE)
    
    def cleanup(self):
        self.hands.close()


def setup_folders():
    """Make sure data folder exists"""
    os.makedirs(config.DATA_FOLDER, exist_ok=True)
    print(f"Data folder ready: {config.DATA_FOLDER}")

