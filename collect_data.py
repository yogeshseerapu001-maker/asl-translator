# =============================================================================
# collect_data.py
# =============================================================================

import cv2
import numpy as np
import os
from helpers import HandTracker, setup_folders
import config

def collect_samples(letter):
    """Collect training samples for one letter"""
    
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    collected = []
    
    print(f"\n=== Collecting data for letter: {letter} ===")
    print(f"Goal: {config.SAMPLES} samples")
    print("Controls: SPACE = capture, Q = quit early\n")
    
    while len(collected) < config.SAMPLES:
        success, frame = cap.read()
        if not success:
            print("Can't read from webcam!")
            break
        
        # flip so it's like a mirror
        frame = cv2.flip(frame, 1)
        
        # detect hand
        results = tracker.detect(frame, draw=True)
        
        # show info on screen
        text1 = f"Letter: {letter}"
        text2 = f"Captured: {len(collected)}/{config.SAMPLES}"
        cv2.putText(frame, text1, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, text2, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # spacebar pressed - capture this frame
            landmarks = tracker.get_landmarks(results)
            if np.any(landmarks):  # make sure we actually got a hand
                collected.append(landmarks)
                print(f"Captured sample {len(collected)}")
            else:
                print("No hand detected! Try again")
        
        elif key == ord('q'):
            print("Quitting early...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()
    
    # save the data
    if len(collected) > 0:
        save_path = os.path.join(config.DATA_FOLDER, f"{letter}.npy")
        np.save(save_path, np.array(collected))
        print(f"Saved {len(collected)} samples to {save_path}\n")
    
    return len(collected)


def collect_all_letters():
    """Go through each letter and collect samples"""
    
    setup_folders()
    
    print("\n" + "="*60)
    print("ASL DATA COLLECTION")
    print("="*60)
    print("We'll go through each letter one by one")
    print("Show the sign when you're ready and press SPACE to capture\n")
    
    for idx, letter in enumerate(config.LETTERS):
        print(f"\n[{idx+1}/{len(config.LETTERS)}] Ready for letter: {letter}")
        input("Press ENTER to start...")
        collect_samples(letter)
    
    print("\nAll done! Data collection complete.")

