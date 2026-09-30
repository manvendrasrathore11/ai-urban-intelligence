import json
from pathlib import Path

import cv2

from traffic.config import (
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    CONGESTION_BANDS,
    PERSISTENT_CONGESTION_FRAMES,
    BUS_ID,
    GPS_LOCATION,
)


def _congestion_level(vehicle_count):
    if vehicle_count >= CONGESTION_BANDS["HIGH"]:
        return "HIGH"
    if vehicle_count >= CONGESTION_BANDS["MODERATE"]:
        return "MODERATE"
    return "LOW"


# ============================================================
# PER-FRAME ENTRY POINT
# ============================================================

def process_traffic_frame(
    model,
    frame,
    frame_number,
    timestamp_seconds,
    gps_location,
    traffic_track_history,
):
    """
    Process ONE video frame for traffic congestion estimation.

    Same shape as process_accident_frame / process_road_damage_frame
    so it drops into pipeline/main.py the same way. Detection uses
    plain YOLO inference (not tracking) since we only need a count
    per frame, not persistent vehicle identities.

    Returns:
        annotated_frame
        vehicle_count            (int, this frame)
        congestion_level         ("LOW" / "MODERATE" / "HIGH")
    """

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    vehicle_count = 0

    if results and len(results) > 0:
        result = results[0]

        if result.boxes is not None:
            boxes = result.boxes

            for i in range(len(boxes)):
                confidence = float(boxes.conf[i].item())

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(boxes.cls[i].item())

                if class_id not in VEHICLE_CLASSES:
                    continue

                vehicle_count += 1

                x1, y1, x2, y2 = map(
                    int, boxes.xyxy[i].tolist()
                )

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (255, 200, 0), 1
                )

    congestion_level = _congestion_level(vehicle_count)

    # ------------------------------------------------------
    # PERSISTENCE TRACKING FOR "HIGH" CONGESTION
    # ------------------------------------------------------

    if congestion_level == "HIGH":
        traffic_track_history["high_streak"] = (
            traffic_track_history.get("high_streak", 0) + 1
        )

        if (
            traffic_track_history["high_streak"]
            >= PERSISTENT_CONGESTION_FRAMES
        ):
            traffic_track_history.setdefault(
                "confirmed_events", []
            ).append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "vehicle_count": vehicle_count,
                    "gps": gps_location or GPS_LOCATION,
                    "bus_id": BUS_ID,
                }
            )
    else:
        traffic_track_history["high_streak"] = 0

    cv2.putText(
        frame,
        f"Traffic: {congestion_level} ({vehicle_count} vehicles)",
        (25, frame.shape[0] - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 200, 0),
        2,
    )

    return frame, vehicle_count, congestion_level


# ============================================================
# FINALIZATION
# ============================================================

def finalize_traffic(
    traffic_track_history,
    output_events_path,
):
    print("\n" + "=" * 60)
    print("TRAFFIC CONGESTION FINALIZATION")
    print("=" * 60)

    confirmed_events = traffic_track_history.get(
        "confirmed_events", []
    )

    if not confirmed_events:
        print("\nNo prolonged high-congestion periods detected.")
        print("=" * 60)
        return None

    output_events_path = Path(output_events_path)
    output_events_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    peak = max(
        confirmed_events, key=lambda e: e["vehicle_count"]
    )

    result = {
        "total_confirmed_frames": len(confirmed_events),
        "peak": peak,
        "events": confirmed_events,
    }

    with open(output_events_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"\n{len(confirmed_events)} frame(s) of prolonged "
        f"HIGH congestion confirmed."
    )
    print(
        f"Peak: {peak['vehicle_count']} vehicles at "
        f"{peak['timestamp_seconds']:.1f}s"
    )
    print(f"\nSaved to: {output_events_path}")
    print("=" * 60)

    return result
