import cv2
import torch
import numpy as np
import mediapipe as mp
from pathlib import Path
from train import GestureClassifier, GESTURES, INPUT_SIZE, MODEL_PATH

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


def extract_landmarks(hand_landmarks) -> torch.Tensor:
    result = []
    for lm in hand_landmarks.landmark:
        result.extend([lm.x, lm.y, lm.z])
    return torch.tensor(result, dtype=torch.float32).unsqueeze(0)


def run():
    model = GestureClassifier()
    model.load_state_dict(torch.load(MODEL_PATH))
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

            label_text = "No hand detected"
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

                with torch.no_grad():
                    x      = extract_landmarks(hand_lm)
                    probs  = torch.softmax(model(x), dim=1)
                    conf, pred = probs.max(dim=1)
                    label_text = GESTURES[pred.item()]
                    confidence = conf.item()

            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
            cv2.putText(frame, f"{label_text} ({confidence:.0%})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow("Gesture Demo", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()