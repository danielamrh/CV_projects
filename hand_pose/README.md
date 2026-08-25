# Hand Pose — Gesture Classification

Real-time hand gesture recognition using MediaPipe for keypoint detection and a PyTorch MLP classifier.

## Gestures

| Key | Gesture |
|-----|---------|
| 0 | ✊ Fist |
| 1 | ☝️ One finger |
| 2 | ✌️ Peace |
| 3 | ✋ Open hand |
| 4 | 👍 Thumbs up |

## Setup

```bash
conda create -n cv_env python=3.10
conda activate cv_env
pip install mediapipe==0.10.9 opencv-python numpy torch
```

## Usage

**1. Test landmark detection:**
```bash
python src/detect.py
```

**2. Collect gesture data:**
```bash
python src/collect_data.py
```
Press a number key (0–4) to start collecting that gesture. 200 samples per gesture are saved to `data/gestures.csv`.

**3. Train the classifier:**
```bash
python src/train.py
```
Trains a 3-layer MLP on the collected landmarks. Best model saved to `models/gesture_classifier.pth`.

**4. Run live demo:**
```bash
python src/demo.py
```

## Architecture

Input: 21 hand landmarks × 3 (x, y, z) = 63 values  
Model: MLP — Linear(63→128) → ReLU → Dropout → Linear(128→128) → ReLU → Dropout → Linear(128→5)  
Val Accuracy: 87.3% (2271 samples, 5 classes)