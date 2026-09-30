from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# VEHICLE DETECTION
# ============================================================

VEHICLE_MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"

VEHICLE_CONF_THRESHOLD = 0.40

VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# PLATE DETECTION
# ============================================================

PLATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "anpr"
    / "plate_detector.pt"
)

PLATE_CONF_THRESHOLD = 0.35

PLATE_MODEL_HF_REPO = (
    "Koushim/yolov8-license-plate-detection"
)

PLATE_MODEL_HF_FILENAME = "best.pt"


# ============================================================
# OCR
# ============================================================

OCR_LANGUAGES = ["en"]

OCR_MIN_CONFIDENCE = 0.35

OCR_MIN_CROP_HEIGHT_PX = 64

# Only characters that can appear in an Indian plate
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# ============================================================
# INDIAN NUMBER PLATE FORMAT
# ============================================================

INDIAN_PLATE_REGEX = (
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
)


# ============================================================
# ANPR OUTPUT
# ============================================================

ANPR_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evidence"
    / "anpr"
)

PLATE_CROP_DIR = (
    ANPR_OUTPUT_DIR
    / "plates"
)

VEHICLE_CROP_DIR = (
    ANPR_OUTPUT_DIR
    / "vehicles"
)

ANPR_EVENTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "events"
    / "anpr_events.json"
)


SAVE_CROPS = True


# ============================================================
# BUS / GPS
# ============================================================

BUS_ID = "BUS_001"

GPS_LATITUDE = 19.0700

GPS_LONGITUDE = 72.8770


# ============================================================
# ANPR VIDEO SETTINGS
# ============================================================

ANPR_INPUT_VIDEO = (
    PROJECT_ROOT
    / "videos"
    / "input"
    / "numplt.mp4"
)

ANPR_OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "outputs"
    / "videos"
    / "anpr_output.mp4"
)


# ============================================================
# DISPLAY
# ============================================================

SHOW_WINDOW = True

WINDOW_NAME = (
    "AI Urban Intelligence - Integrated Pipeline"
)

DISPLAY_WIDTH = 1280

DISPLAY_HEIGHT = 720


# ============================================================
# CREATE DIRECTORIES
# ============================================================

PLATE_CROP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VEHICLE_CROP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANPR_EVENTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# PERFORMANCE
# ============================================================

VEHICLE_IMGSZ = 640
PLATE_IMGSZ = 640

# Run OCR only once every N frames for a tracked vehicle
OCR_FRAME_INTERVAL = 5

# Minimum OCR confidence
MIN_OCR_CONFIDENCE = 0.50