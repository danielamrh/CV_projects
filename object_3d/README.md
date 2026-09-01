# Object 3D Localization — YOLOv8 + MiDaS

Real-time object detection with depth estimation using a standard RGB webcam.
YOLOv8 detects objects and draws bounding boxes; MiDaS estimates per-pixel depth.
Both are combined to approximate the 3D position of each detected object.

```
Webcam Frame
    ├── → MiDaS  → Depth Map (per-pixel relative depth)
    ├── → YOLO   → Bounding Boxes (label + confidence)
    └── Combination → depth value per detected object
```

Bounding box color indicates depth: **green = close**, **red = far**.
Top-right corner shows the full depth map with MAGMA colormap.

> **Note:** MiDaS outputs relative depth, not absolute meters.
> Object A can be compared to Object B, but no real-world distance is available without camera calibration.

## Setup

```bash
conda activate cv_env
pip install ultralytics torch torchvision timm
```

## Usage

```bash
python src/detect.py
python src/detect.py --camera 1   # different camera index
```

YOLOv8n weights (~6 MB) download automatically on first run.

## Models

| Model | Description |
|-------|-------------|
| YOLOv8n | Nano — fastest YOLO variant, 80 COCO classes |
| MiDaS small | Monocular depth estimation, good speed/accuracy balance |