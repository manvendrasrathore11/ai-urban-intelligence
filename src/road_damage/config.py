from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_INPUT_DIR = PROJECT_ROOT / "videos" / "input"

VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"

EVENT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "events"


# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_REPO = "rezzzq/yolo12s-road-damage-rdd2022"

MODEL_FILENAME = "yolo12s_RDD2022_best.pt"


# --------------------------------------------------
# DETECTION SETTINGS
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.40


# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)