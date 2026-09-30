from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_INPUT_DIR = PROJECT_ROOT / "videos" / "input"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"
EVENT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "events"




ACCIDENT_EVIDENCE_DIR = (
    PROJECT_ROOT / "outputs" / "evidence" / "accident"
)

ACCIDENT_EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)
# ============================================================
# DETECTION
# ============================================================

CONFIDENCE_THRESHOLD = 0.40

IOU_COLLISION_THRESHOLD = 0.20

PROLONGED_COLLISION_FRAMES = 30


# ============================================================
# DISPLAY
# ============================================================

SHOW_WINDOW = True

WINDOW_NAME = "AI Urban Intelligence - Accident Detection"

EXIT_KEYS = (ord("q"), 27)


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
# DEMO GPS
# ============================================================

BUS_ID = "BUS_001"

GPS_LOCATION = {
    "latitude": 19.070,
    "longitude": 72.877,
}