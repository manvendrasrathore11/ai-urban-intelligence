"""
Traffic congestion model loader.

Unlike waterlogging, this module DOES use a real pretrained
network: Ultralytics YOLOv8n (yolov8n.pt), trained on the public
COCO dataset. Ultralytics auto-downloads this checkpoint from
their GitHub Releases the first time it is requested by name
(no manual download step needed, same as accident/model.py
already does in this repo).

COCO's 80 classes already include car / motorcycle / bus / truck,
so no custom training is required to count and classify vehicles;
we only need to add the counting + congestion-banding logic on
top, which lives in detector.py.
"""

from ultralytics import YOLO

from traffic.config import MODEL_WEIGHTS


def load_traffic_model():
    print("Loading traffic congestion model...")

    model = YOLO(MODEL_WEIGHTS)

    print("Traffic congestion model loaded successfully.")

    return model
