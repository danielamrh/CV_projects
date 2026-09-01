"""
Gesture Image Collection for CNN Training.

Captures hand crop images from webcam and saves them
to class folders under data/.

Usage:
    python src/collect_data.py
"""

import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GESTURES = {
    "0": "fist",
    "1": "one_finger",
    "2": "peace",
    "3": "open_hand",
    "4": "thumbs_up",
}
SAMPLES_PER_GESTURE = 200
IMG_SIZE            = 64       # pixels — CNN input size
DATA_DIR            = Path("data")

def get_hand_crop(frame: np.ndarray, hand_landmarks) -> np.ndarray:
    """
    Crop the hand region from the frame based on landmark bounding box.
    Adds padding and resizes to IMG_SIZE x IMG_SIZE.

    Returns None if the crop is out of bounds.
    """
    h, w = frame.shape[:2]

    # Get bounding box from landmarks
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x1 = int(min(xs) * w)
    y1 = int(min(ys) * h)
    x2 = int(max(xs) * w)
    y2 = int(max(ys) * h)

    # Add padding (30% of box size)
    pad_x = int((x2 - x1) * 0.3)
    pad_y = int((y2 - y1) * 0.3)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

def collect():

    cap = cv2.VideoCapture(0)
    current_class = None
    count = 0

    print("Keys: 0=fist, 1=one_finger, 2=open_hand, 3=peace, 4=thumbs_up")
    print("Press s to quit.")

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks and current_class is not None:
                hand_lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                crop = get_hand_crop(frame, hand_lm)
                if crop is not None:
                    save_dir = DATA_DIR / GESTURES[current_class]
                    save_dir.mkdir(parents=True, exist_ok=True)
                    filename = save_dir / f"{count:04d}.jpg"
                    cv2.imwrite(str(filename), crop)
                    count += 1

                if count >= SAMPLES_PER_GESTURE:
                    print(f"\n  {SAMPLES_PER_GESTURE} Pictures for '{GESTURES[current_class]}' saved!")
                    current_class = None
                    count = 0

            label = GESTURES.get(current_class, "--- No gesture active ---")
            cv2.putText(frame, f"{label} ({count}/{SAMPLES_PER_GESTURE})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Data Collection", frame)

            key = chr(cv2.waitKey(1) & 0xFF)
            if key == "s":
                break
            elif key in GESTURES:
                current_class = key
                count = 0
                print(f"\nSammle: {GESTURES[key]} ...")

    cap.release()
    cv2.destroyAllWindows()
    print("Fertig.")

if __name__ == "__main__":
    collect()              