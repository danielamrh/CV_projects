"""
Hand Pose Detection — Real-time Keypoint Visualization

Uses MediaPipe Hands to detect 21 hand landmarks per hand
and draws them live on the webcam feed.

Usage:
    python src/detect.py
    python src/detect.py --camera 1
"""

import argparse

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_landmarks(frame, hand_landmarks, handedness):
    mp_drawing.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )
    wrist = hand_landmarks.landmark[0]
    h, w  = frame.shape[:2]
    cv2.putText(frame, handedness,
        (int(wrist.x * w), int(wrist.y * h) - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def run(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    tick = cv2.getTickFrequency()
    t_prev = cv2.getTickCount()

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, 
        min_detection_confidence=0.7, min_tracking_confidence=0.6
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = hand_info.classification[0].label
                    draw_landmarks(frame, hand_lm, label)

            t_now = cv2.getTickCount()
            fps = tick / (t_now - t_prev)
            t_prev = t_now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Hand Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run(args.camera)
