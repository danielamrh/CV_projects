"""
Object 3D Localization — YOLOv8 + MiDaS

Detects objects with YOLOv8 and estimates their depth with MiDaS.
Combines both to output the approximate 3D position of each object.

Usage:
    python src/detect.py
    python src/detect.py --camera 1
"""

import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────
YOLO_MODEL  = "yolov8n.pt"       # nano - fastest, downloads automatically
MIDAS_MODEL = "MiDaS_small"      # small - good balance speed/accuracy
CONF_THRESH = 0.4                 # minimum YOLO confidence
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """Load YOLOv8 and MiDaS models."""
    print(f"Using device: {DEVICE}")

    # YOLOv8 — downloads yolov8n.pt automatically on first run
    yolo = YOLO(YOLO_MODEL)

    # MiDaS — depth estimation
    midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL)
    midas.to(DEVICE)
    midas.eval()

    # MiDaS preprocessing transform
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform  = transforms.small_transform

    return yolo, midas, transform

def estimate_depth(frame: np.ndarray, midas, transform) -> np.ndarray:
    """
    Run MiDaS on a BGR frame and return a normalized depth map (0-1).
    Higher value = closer to camera.
    """
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(DEVICE)

    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_np = depth.cpu().numpy()

    # Normalize to 0-1 (invert so closer = higher value)
    d_min, d_max = depth_np.min(), depth_np.max()
    depth_norm   = 1.0 - (depth_np - d_min) / (d_max - d_min + 1e-6)

    return depth_norm

def get_box_depth(depth_map: np.ndarray, box) -> float:
    """
    Compute the median depth inside a YOLO bounding box.
    Uses median instead of mean to ignore background pixels.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    roi = depth_map[y1:y2, x1:x2]
    return float(np.median(roi)) if roi.size > 0 else 0.0

def draw_detection(frame: np.ndarray, box, label: str, depth: float) -> None:
    """Draw bounding box, label and depth value on the frame."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf            = float(box.conf[0])

    # Color: green = close, red = far
    g     = int(255 * depth)
    r     = int(255 * (1.0 - depth))
    color = (0, g, r)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{label} {conf:.0%} | depth: {depth:.2f}",
        (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
    )

def run(camera_index: int = 0) -> None:
    yolo, midas, transform = load_models()
    cap = cv2.VideoCapture(camera_index)
    print("Running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Depth map for entire frame
        depth_map = estimate_depth(frame, midas, transform)

        # YOLO detections
        results = yolo(frame, conf=CONF_THRESH, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = yolo.names[cls_id]
            depth  = get_box_depth(depth_map, box)
            draw_detection(frame, box, label, depth)

        # Show depth map as overlay (top-right corner)
        h, w      = frame.shape[:2]
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_col = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        small     = cv2.resize(depth_col, (w // 4, h // 4))
        frame[10:10 + small.shape[0], w - small.shape[1] - 10:w - 10] = small

        cv2.imshow("Object 3D Localization", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run(args.camera)