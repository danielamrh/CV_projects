"""
CNN Gesture Classifier — Live Demo.

Loads the trained CNN and classifies hand gestures in real-time.

Usage:
    python src/demo.py
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from pathlib import Path
from train import GestureCNN, GESTURES, IMG_SIZE, MODEL_PATH, DEVICE

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

CONF_THRESH = 0.7   # minimum softmax confidence to show prediction

# Same normalization as training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def get_hand_crop(frame: np.ndarray, hand_landmarks) -> np.ndarray | None:
    """Crop and resize hand region from frame."""
    h, w = frame.shape[:2]

    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x1 = int(min(xs) * w)
    y1 = int(min(ys) * h)
    x2 = int(max(xs) * w)
    y2 = int(max(ys) * h)

    pad_x = int((x2 - x1) * 0.3)
    pad_y = int((y2 - y1) * 0.3)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))

def run():
    model = GestureCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded. Press Q to quit.")

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            label_text = "No hand"
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                crop = get_hand_crop(frame, hand_lm)
                if crop is not None:
                    x = transform(crop).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        probs      = torch.softmax(model(x), dim=1)
                        conf, pred = probs.max(dim=1)
                        confidence = conf.item()
                        if confidence >= CONF_THRESH:
                            label_text = GESTURES[pred.item()]
                        else:
                            label_text = "uncertain"

            color = (0, 255, 0) if confidence >= CONF_THRESH else (0, 165, 255)
            cv2.putText(frame, f"{label_text} ({confidence:.0%})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow("Gesture CNN Demo", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()