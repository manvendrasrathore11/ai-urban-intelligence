from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_INPUT_DIR = PROJECT_ROOT / "videos" / "input"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"
EVENT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "events"


# ============================================================
# MODEL
# ============================================================

# Same pretrained Ultralytics YOLO checkpoint already vendored
# in src/ (yolov8n.pt) - trained on COCO, which already includes
# vehicle classes. No new download needed; this module simply
# points at the same weights accident/model.py uses so the
# vehicle detector is not loaded into memory twice needlessly
# if you later choose to share one YOLO() instance across
# accident + traffic (optional optimization, not required).
MODEL_WEIGHTS = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.35


# ============================================================
# VEHICLE CLASSES - COCO
# ============================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# CONGESTION THRESHOLDS
# Tune these against your own footage; defaults assume a
# bus-mounted camera looking ahead at a single lane/road width.
# ============================================================

CONGESTION_BANDS = {
    "LOW": 0,
    "MODERATE": 6,
    "HIGH": 12,
}

# A HIGH-congestion reading must persist across this many
# consecutive sampled frames before it is logged as a confirmed
# congestion event (mirrors accident's PROLONGED_COLLISION_FRAMES).
PERSISTENT_CONGESTION_FRAMES = 20

FRAME_SAMPLE_RATE = 3


# ============================================================
# DEMO GPS
# ============================================================

BUS_ID = "BUS_001"

GPS_LOCATION = {
    "latitude": 19.070,
    "longitude": 72.877,
}
