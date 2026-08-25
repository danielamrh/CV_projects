import cv2
import csv
import numpy as np
import mediapipe as mp
from pathlib import Path

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

GESTURES = {
    "0": "fist",
    "1": "one_finger",
    "2": "peace",
    "3": "open_hand",
    "4": "thumbs_up",
}
SAMPLES_PER_GESTURE = 200
OUTPUT_PATH = Path("data/gestures.csv")

def extract_landmarks(hand_landmarks) -> list:
    """Flatten 21 landmarks (x, y, z) into a list of 63 values."""
    result = []
    for lm in hand_landmarks.landmark:
        result.extend([lm.x, lm.y, lm.z])
    return result

def collect():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Open CSV (append mode so we can collect in multiple sessions)
    csv_file = open(OUTPUT_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    cap = cv2.VideoCapture(0)
    current_label = None
    count = 0

    print("0=fist  1=one_finger  2=peace  3=open_hand  4=thumbs_up")
    print("'s' to stop")

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

            if results.multi_hand_landmarks and current_label is not None:
                hand_lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

                landmarks = extract_landmarks(hand_lm)
                writer.writerow([current_label] + landmarks)
                count += 1

            # Overlay
            label_text = GESTURES.get(current_label, "None")
            cv2.putText(frame, f"Geste: {label_text} ({count}/{SAMPLES_PER_GESTURE})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Data Collection", frame)

            key = chr(cv2.waitKey(1) & 0xFF)
            if key == "s":
                break
            elif key in GESTURES:
                current_label = key
                count = 0
                print(f"\nSammle: {GESTURES[key]} ...")

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nDaten gespeichert in: {OUTPUT_PATH}")


if __name__ == "__main__":
    collect()

