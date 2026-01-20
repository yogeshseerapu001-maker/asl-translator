# =============================================================================
# config.py
# =============================================================================

import os

# Project paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_PATH, 'data', 'keypoints')
MODEL_FILE = os.path.join(BASE_PATH, 'my_asl_model.h5')

# Letters I'm training on (J and Z need motion so skipping those)
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# How many samples to collect per letter
SAMPLES = 100

# MediaPipe confidence thresholds (these seem to work well)
DETECTION_CONF = 0.7
TRACKING_CONF = 0.5

# Model stuff
HAND_LANDMARKS = 21  # mediapipe gives us 21 points
COORDS = 3  # x, y, z for each point
INPUT_SIZE = HAND_LANDMARKS * COORDS  # total features = 63

# Training hyperparameters (tuned these by trial and error)
BATCH = 32
EPOCHS = 50
VAL_SPLIT = 0.2
LR = 0.001
