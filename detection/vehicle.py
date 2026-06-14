"""
Uses the same YOLOv8 model already loaded for weapon detection.
No extra model download needed — YOLO knows vehicles out of the box.

YOLO class IDs:
    1  = bicycle
    2  = car
    3  = motorcycle
    5  = bus
    7  = truck

"""

from ultralytics import YOLO
import numpy as np
import os

# Model
_model = None

VEHICLE_CLASSES = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

# Confidence threshold per class
# Trucks and buses are large so, easy to detect, higher threshold
# Bicycles are small and often partially visible, lower threshold
CONFIDENCE_THRESHOLDS = {
    'bicycle'   : 0.30,
    'car'       : 0.35,
    'motorcycle': 0.30,
    'bus'       : 0.40,
    'truck'     : 0.40,
}


def _get_model():
    """Load model once, reuse forever. Saves ~200MB RAM."""
    global _model
    if _model is None:
        weights = os.path.join(
            os.path.dirname(__file__), '..', 'datasets', 'yolov8n.pt'
        )
        _model = YOLO(weights)
    return _model


def detect_vehicles(frame: np.ndarray) -> list[dict]:

    # Run YOLO on a frame and return only vehicle detections.

    model   = _get_model()
    results = model(frame, verbose=False)[0]

    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])

        # Skip anything that is not a vehicle
        if class_id not in VEHICLE_CLASSES:
            continue

        vehicle_type = VEHICLE_CLASSES[class_id]
        conf         = float(box.conf[0])
        threshold    = CONFIDENCE_THRESHOLDS[vehicle_type]

        # Skip low-confidence detections
        if conf < threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w  = x2 - x1
        h  = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2

        detections.append({
            'class'  : vehicle_type,
            'bbox'   : (x1, y1, x2, y2),
            'conf'   : conf,
            'center' : (cx, cy),
            'area'   : w * h,
        })

    return detections
