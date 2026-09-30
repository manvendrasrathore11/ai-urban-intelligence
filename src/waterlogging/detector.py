import json
from pathlib import Path

import cv2
import numpy as np

from waterlogging.config import (
    ROAD_REGION_TOP_FRACTION,
    TEXTURE_VARIANCE_MAX,
    BRIGHTNESS_MIN,
    SATURATION_MAX,
    MIN_CONTOUR_AREA,
    PERSISTENCE_FRAMES,
    SEVERITY_BANDS,
    BUS_ID,
    GPS_LOCATION,
)


# ============================================================
# CORE HEURISTIC
# ============================================================

def _detect_water_regions(frame):
    """
    Return (candidate_mask, contours, road_region_area) for ONE
    frame using a classical CV heuristic:

      1. Crop to the lower part of the frame (the road).
      2. Convert to HSV; standing water tends to be brighter
         and less saturated than surrounding wet/dry asphalt
         because it reflects the sky.
      3. Convert to grayscale and compute local variance
         (via Laplacian) in a blurred version; water is visually
         smoother (lower local variance) than textured asphalt.
      4. A pixel is a "water candidate" if it passes the
         brightness/saturation AND smoothness tests together.
      5. Morphologically clean the mask and extract contours
         above MIN_CONTOUR_AREA.
    """

    height, width = frame.shape[:2]

    road_top = int(height * ROAD_REGION_TOP_FRACTION)
    road_region = frame[road_top:height, 0:width]

    hsv = cv2.cvtColor(road_region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    gray = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Local variance approximated with Laplacian magnitude on a
    # blurred image: smooth (water-like) surfaces score low.
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    local_texture = cv2.convertScaleAbs(laplacian)

    smooth_mask = local_texture < TEXTURE_VARIANCE_MAX
    bright_mask = brightness > BRIGHTNESS_MIN
    low_sat_mask = saturation < SATURATION_MAX

    candidate_mask = (
        smooth_mask & bright_mask & low_sat_mask
    ).astype("uint8") * 255

    kernel = np.ones((5, 5), np.uint8)
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_OPEN, kernel
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) >= MIN_CONTOUR_AREA
    ]

    road_region_area = road_region.shape[0] * road_region.shape[1]

    return candidate_mask, contours, road_region_area, road_top


def _severity_from_ratio(covered_ratio):
    if covered_ratio >= SEVERITY_BANDS["SEVERE"]:
        return "SEVERE"
    if covered_ratio >= SEVERITY_BANDS["MODERATE"]:
        return "MODERATE"
    if covered_ratio >= SEVERITY_BANDS["LOW"]:
        return "LOW"
    return None


# ============================================================
# PER-FRAME ENTRY POINT
# ============================================================

def process_waterlogging_frame(
    model,
    frame,
    frame_number,
    timestamp_seconds,
    gps_location,
    waterlogging_track_history,
):
    """
    Process ONE video frame for waterlogging detection.

    This mirrors process_road_damage_frame /
    process_accident_frame in shape: it takes the shared
    per-frame state (waterlogging_track_history) explicitly and
    returns the annotated frame plus this frame's detections, so
    pipeline/main.py can call it the same way.

    Returns:
        annotated_frame
        frame_detections   (list of dicts, empty if none)
    """

    contours = []
    covered_ratio = 0.0
    road_top = int(
        frame.shape[0] * ROAD_REGION_TOP_FRACTION
    )

    if model is not None:
        (
            _mask,
            contours,
            road_region_area,
            road_top,
        ) = _detect_water_regions(frame)

        covered_area = sum(
            cv2.contourArea(c) for c in contours
        )

        covered_ratio = (
            covered_area / road_region_area
            if road_region_area > 0 else 0.0
        )

    severity = _severity_from_ratio(covered_ratio)

    frame_detections = []

    # ------------------------------------------------------
    # DRAW + BUILD DETECTIONS (offset contours back into the
    # full frame's coordinate space, since they were found in
    # the cropped road_region)
    # ------------------------------------------------------

    for contour in contours:
        contour_full = contour + [0, road_top]

        x, y, w, h = cv2.boundingRect(contour_full)

        cv2.drawContours(
            frame,
            [contour_full],
            -1,
            (0, 165, 255),
            2,
        )

        cv2.putText(
            frame,
            "WATERLOGGING",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            2,
        )

        frame_detections.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": timestamp_seconds,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area_px": float(cv2.contourArea(contour)),
                "gps": gps_location or GPS_LOCATION,
                "bus_id": BUS_ID,
            }
        )

    # ------------------------------------------------------
    # TRACK PERSISTENCE (so a single-frame glare glint does not
    # get reported as a confirmed event)
    # ------------------------------------------------------

    if severity is not None:
        waterlogging_track_history["streak"] = (
            waterlogging_track_history.get("streak", 0) + 1
        )
        waterlogging_track_history["last_ratio"] = covered_ratio
        waterlogging_track_history["last_severity"] = severity
        waterlogging_track_history["last_seen_frame"] = frame_number

        if (
            waterlogging_track_history["streak"]
            >= PERSISTENCE_FRAMES
        ):
            waterlogging_track_history.setdefault(
                "confirmed_events", []
            ).append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "severity": severity,
                    "covered_ratio": round(covered_ratio, 4),
                    "gps": gps_location or GPS_LOCATION,
                    "bus_id": BUS_ID,
                }
            )
    else:
        waterlogging_track_history["streak"] = 0

    if severity is not None:
        cv2.putText(
            frame,
            f"Waterlogging: {severity} "
            f"({covered_ratio * 100:.1f}% of road)",
            (25, frame.shape[0] - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )

    return frame, frame_detections


# ============================================================
# FINALIZATION
# ============================================================

def finalize_waterlogging(
    waterlogging_track_history,
    output_events_path,
):
    """
    Finalize waterlogging detection after the complete video
    has been processed. Mirrors finalize_road_damage /
    finalize_accident.
    """

    print("\n" + "=" * 60)
    print("WATERLOGGING DETECTION FINALIZATION")
    print("=" * 60)

    confirmed_events = waterlogging_track_history.get(
        "confirmed_events", []
    )

    if not confirmed_events:
        print("\nNo confirmed waterlogging detected.")
        print("=" * 60)
        return None

    output_events_path = Path(output_events_path)
    output_events_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    worst = max(
        confirmed_events,
        key=lambda e: e["covered_ratio"],
    )

    result = {
        "total_confirmed_frames": len(confirmed_events),
        "worst_case": worst,
        "events": confirmed_events,
    }

    with open(output_events_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"\n{len(confirmed_events)} waterlogged frame(s) confirmed."
    )
    print(
        f"Worst severity: {worst['severity']} "
        f"({worst['covered_ratio'] * 100:.1f}% of road region)"
    )
    print(f"\nSaved to: {output_events_path}")
    print("=" * 60)

    return result
