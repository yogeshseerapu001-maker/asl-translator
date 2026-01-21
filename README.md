# ASL Translator 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://asl-translator-yogesh.streamlit.app)

##  Live Demo

**[Try it live on Streamlit Cloud!](https://asl-translator-yogesh.streamlit.app)**

> Note: Camera functionality works when running locally. The cloud version showcases the UI and architecture.

##  Run Locally (with Camera)

\`\`\`bash
git clone https://github.com/yogeshseerapu001-maker/asl-translator.git
cd asl-translator
pip install -r requirements.txt
streamlit run streamlit_app.py
\`\`\`

Real-time American Sign Language letter recognition using MediaPipe and deep learning.

## Features
- Recognizes 24 ASL letters (A-Y, excluding J and Z which require motion)
- Real-time webcam-based recognition
- Deep learning with TensorFlow/Keras
- MediaPipe hand landmark detection

## Project Structure
```
asl_translator/
├── config.py              # Configuration settings
├── helpers.py             # Hand detection utilities
├── collect_data.py        # Data collection script
├── build_model.py         # Model architecture
├── train_model.py         # Training script
├── live_demo.py          # Real-time prediction
├── run.py                # Main entry point
├── requirements.txt      # Dependencies
└── data/
    └── keypoints/        # Training data storage
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yogeshseerapu001-maker/asl-translator.git
cd asl-translator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python run.py
```

Then follow the menu:
1. **Collect training data** - Capture samples for each letter
2. **Train the model** - Train the neural network
3. **Run live prediction** - Test real-time recognition

## Requirements
- Python 3.8-3.11
- Webcam
- TensorFlow 2.10+
- OpenCV
- MediaPipe

## How It Works
1. MediaPipe detects hand landmarks (21 points per hand)
2. Landmarks are extracted as 63 features (x, y, z coordinates)
3. Neural network classifies the hand shape into ASL letters
4. Real-time predictions displayed on webcam feed

## Model Architecture
- Input: 63 features (21 landmarks × 3 coordinates)
- 3 hidden layers (128, 64, 32 units)
- Dropout for regularization
- Output: 24 classes (softmax activation)

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
