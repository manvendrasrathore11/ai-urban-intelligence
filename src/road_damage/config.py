from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_INPUT_DIR = PROJECT_ROOT / "videos" / "input"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"
EVENT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "events"


<<<<<<< HEAD
# Evidence photographs
EVIDENCE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evidence"
ROAD_DAMAGE_EVIDENCE_DIR = EVIDENCE_OUTPUT_DIR / "road_damage"

=======
>>>>>>> 6e1c54d4c6a35f7b69d5b9cc1a36de6fbef33b31
# ============================================================
# MODEL
# ============================================================

MODEL_REPO = "rezzzq/yolo12s-road-damage-rdd2022"

MODEL_FILENAME = "yolo12s_RDD2022_best.pt"


# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.40


<<<<<<< HEAD
# EVIDENCE SETTINGS
EVIDENCE_WINDOW_SECONDS = 5
MIN_DISTINCT_DAMAGE_OBJECTS = 3

=======
>>>>>>> 6e1c54d4c6a35f7b69d5b9cc1a36de6fbef33b31
# ============================================================
# DISPLAY SETTINGS
# ============================================================

# Show the processed video in a window
SHOW_WINDOW = True

# Window name
WINDOW_NAME = "AI Urban Intelligence - Road Damage Detection"

# Press Q or ESC to stop processing
EXIT_KEYS = [ord("q"), 27]




# 

BUS_ID = "BUS_001"

GPS_START_LATITUDE = 26.9124
GPS_START_LONGITUDE = 75.7873



# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

VIDEO_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVENT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
<<<<<<< HEAD
)

ROAD_DAMAGE_EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
=======
>>>>>>> 6e1c54d4c6a35f7b69d5b9cc1a36de6fbef33b31
)