from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_INPUT_DIR = PROJECT_ROOT / "videos" / "input"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"
EVENT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "events"

WATERLOGGING_EVIDENCE_DIR = (
    PROJECT_ROOT / "outputs" / "evidence" / "waterlogging"
)

WATERLOGGING_EVIDENCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DETECTION REGION
# ============================================================

# Only scan the lower part of the frame (the road surface),
# expressed as a fraction of frame height. Sky / buildings in
# the upper part of a bus-camera frame are not road, so we
# ignore them to cut false positives.
ROAD_REGION_TOP_FRACTION = 0.55


# ============================================================
# HEURISTIC THRESHOLDS
# (see detector.py docstring for what each one controls)
# ============================================================

# Water surfaces are visually smooth (low local variance) and
# often reflect the sky, so they tend to be brighter and less
# saturated than wet/dry asphalt around them.
TEXTURE_VARIANCE_MAX = 60.0
BRIGHTNESS_MIN = 90
SATURATION_MAX = 60

# Minimum contour area (in pixels) for a candidate water region
# to be reported as a detection. Filters out small specular
# glints from wet asphalt that are not real waterlogging.
MIN_CONTOUR_AREA = 1800

# A region must persist across this many consecutive sampled
# frames (see FRAME_SAMPLE_RATE) before it is confirmed as a
# waterlogging event, to reduce single-frame false positives
# from glare, headlight reflections, etc.
PERSISTENCE_FRAMES = 4

# Process every Nth frame (mirrors the road_damage module's
# frame-sampling strategy) since waterlogged patches do not
# change frame-to-frame on bus footage.
FRAME_SAMPLE_RATE = 5


# ============================================================
# SEVERITY BANDS
# Based on the fraction of the scanned road region classified
# as standing water.
# ============================================================

SEVERITY_BANDS = {
    "LOW": 0.03,      # >= 3% of road region
    "MODERATE": 0.08,  # >= 8% of road region
    "SEVERE": 0.18,    # >= 18% of road region
}


# ============================================================
# DEMO GPS (overridden by GPSProvider if wired in from pipeline)
# ============================================================

BUS_ID = "BUS_001"

GPS_LOCATION = {
    "latitude": 19.070,
    "longitude": 72.877,
}
