from pathlib import Path

import cv2

from road_damage.model import load_road_damage_model
from road_damage.detector import (
    process_road_damage_frame,
    finalize_road_damage
)

from road_damage.config import (
    GPS_START_LATITUDE,
    GPS_START_LONGITUDE
)

from road_damage.gps import GPSProvider

from accident.model import load_accident_model
from accident.detector import (
    process_accident_frame,
    finalize_accident
)

from waterlogging.model import load_waterlogging_model
from waterlogging.detector import (
    process_waterlogging_frame,
    finalize_waterlogging
)

from traffic.model import load_traffic_model
from traffic.detector import (
    process_traffic_frame,
    finalize_traffic
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_INPUT_DIR = PROJECT_ROOT / "videos" / "input"

VIDEO_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "videos"
)

EVENT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "events"
)


# ============================================================
# PIPELINE SETTINGS
# ============================================================

INPUT_VIDEO = VIDEO_INPUT_DIR / "testing2.mp4"

OUTPUT_VIDEO = (
    VIDEO_OUTPUT_DIR /
    "urban_intelligence_output.mp4"
)

ROAD_DAMAGE_OUTPUT = (
    EVENT_OUTPUT_DIR /
    "road_damage_events.json"
)

ACCIDENT_OUTPUT = (
    EVENT_OUTPUT_DIR /
    "accident_events.json"
)

WATERLOGGING_OUTPUT = (
    EVENT_OUTPUT_DIR /
    "waterlogging_events.json"
)

TRAFFIC_OUTPUT = (
    EVENT_OUTPUT_DIR /
    "traffic_events.json"
)


def draw_pipeline_hud(
    frame,
    frame_number,
    timestamp_seconds,
    road_detections,
    vehicle_count,
    collision_count,
    prolonged_collision,
    accident_detected
):
    """
    Draw one unified dashboard on the final pipeline frame.
    """

    height, width = frame.shape[:2]

    # =========================================================
    # TOP HEADER
    # =========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI URBAN INTELLIGENCE",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "MOBILE URBAN INTELLIGENCE PLATFORM",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )

    # =========================================================
    # ROAD DAMAGE PANEL
    # =========================================================

    cv2.rectangle(
        frame,
        (15, 110),
        (width // 2 - 10, 205),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "ROAD DAMAGE",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Count different damage types
    potholes = sum(
        1 for d in road_detections
        if d.get("damage_type") == "D40"
    )

    longitudinal = sum(
        1 for d in road_detections
        if d.get("damage_type") == "D00"
    )

    transverse = sum(
        1 for d in road_detections
        if d.get("damage_type") == "D10"
    )

    alligator = sum(
        1 for d in road_detections
        if d.get("damage_type") == "D20"
    )

    repairs = sum(
        1 for d in road_detections
        if d.get("damage_type") == "Repair"
    )

    cv2.putText(
        frame,
        f"Objects: {len(road_detections)}",
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Potholes: {potholes}",
        (180, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Cracks: {longitudinal + transverse + alligator}",
        (320, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Repairs: {repairs}",
        (500, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # =========================================================
    # ACCIDENT / TRAFFIC PANEL
    # =========================================================

    cv2.rectangle(
        frame,
        (width // 2 + 10, 110),
        (width - 15, 205),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "ACCIDENT / TRAFFIC",
        (width // 2 + 25, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    status = "ACCIDENT DETECTED" if accident_detected else "MONITORING"

    cv2.putText(
        frame,
        f"Status: {status}",
        (width // 2 + 25, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (width // 2 + 230, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Collision: {collision_count}",
        (width // 2 + 370, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # =========================================================
    # BOTTOM STATUS BAR
    # =========================================================

    cv2.rectangle(
        frame,
        (0, height - 55),
        (width, height),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FRAME: {frame_number}",
        (25, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    minutes = int(timestamp_seconds // 60)
    seconds = int(timestamp_seconds % 60)

    cv2.putText(
        frame,
        f"TIME: {minutes:02d}:{seconds:02d}",
        (180, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"VEHICLES: {vehicle_count}",
        (350, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"ROAD EVENTS: {len(road_detections)}",
        (530, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    return frame

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("AI URBAN INTELLIGENCE PIPELINE")
    print("=" * 70)

    # ========================================================
    # LOAD MODELS
    # ========================================================

    print("\n[1/4] Loading Road Damage model...")

    road_damage_model = (
        load_road_damage_model()
    )

    print("\n[2/4] Loading Accident model...")

    accident_model = (
        load_accident_model()
    )

    print(
        "Accident model loaded successfully."
    )

    print("\n[3/4] Loading Waterlogging detector...")

    waterlogging_model = (
        load_waterlogging_model()
    )

    print("\n[4/4] Loading Traffic congestion model...")

    traffic_model = (
        load_traffic_model()
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    print("\nOpening input video:")

    print(INPUT_VIDEO)

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {INPUT_VIDEO}"
        )

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print("\nVideo information:")

    print(f"FPS: {fps}")
    print(f"Resolution: {width} x {height}")
    print(f"Total frames: {total_frames}")

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    VIDEO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EVENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CREATE VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{OUTPUT_VIDEO}"
        )

    # ========================================================
    # ROAD DAMAGE STATE
    # ========================================================

    road_track_history = {}

    gps_provider = GPSProvider(
        start_latitude=GPS_START_LATITUDE,
        start_longitude=GPS_START_LONGITUDE
    )

    # ========================================================
    # ACCIDENT STATE
    # ========================================================

    collision_counter = 0

    prolonged_collision_counter = 0

    accident_detected = False

    accident_event = None
    
    # ========================================================
    # WATERLOGGING STATE
    # ========================================================

    waterlogging_track_history = {}

    # ========================================================
    # TRAFFIC STATE
    # ========================================================

    traffic_track_history = {}    

    # ========================================================
    # FRAME LOOP
    # ========================================================

    frame_number = 0

    print("\nStarting integrated processing...")
    print("-" * 70)

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # VIDEO TIMESTAMP
        # ----------------------------------------------------

        timestamp_seconds = (
            (frame_number - 1) / fps
        )

        # ----------------------------------------------------
        # GPS
        # ----------------------------------------------------

        gps_location = (
            gps_provider.get_location(
                frame_number
            )
        )

        # ====================================================
        # ROAD DAMAGE
        # ====================================================

        (
            frame,
            road_detections
        ) = process_road_damage_frame(
            road_damage_model,
            frame,
            frame_number,
            timestamp_seconds,
            gps_location,
            road_track_history
        )
        


        # ====================================================
        # ACCIDENT
        # ====================================================

        (
            frame,
            collision_counter,
            prolonged_collision_counter,
            accident_detected,
            vehicle_count,
            frame_collision_count,
            current_accident_event
        ) = process_accident_frame(
            accident_model,
            frame,
            frame_number,
            timestamp_seconds,
            collision_counter,
            prolonged_collision_counter,
            accident_detected
        )
        
        frame = draw_pipeline_hud(
        frame=frame,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        road_detections=road_detections,
        vehicle_count=vehicle_count,
        collision_count=collision_counter,
        prolonged_collision=prolonged_collision_counter,
        accident_detected=accident_detected
        )
        
      
        # ====================================================
        # WATERLOGGING
        # ====================================================

        (
            frame,
            waterlogging_detections
        ) = process_waterlogging_frame(
            waterlogging_model,
            frame,
            frame_number,
            timestamp_seconds,
            gps_location,
            waterlogging_track_history
        )

        # ====================================================
        # TRAFFIC CONGESTION
        # ====================================================

        (
            frame,
            traffic_vehicle_count,
            congestion_level
        ) = process_traffic_frame(
            traffic_model,
            frame,
            frame_number,
            timestamp_seconds,
            gps_location,
            traffic_track_history
        )      
        
        

        # ====================================================
        # STORE ACCIDENT EVENT
        # ====================================================

        if (
            current_accident_event is not None
            and accident_event is None
        ):

            accident_event = (
                current_accident_event
            )

        # ====================================================
        # WRITE COMBINED FRAME
        # ====================================================

        writer.write(frame)

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "AI Urban Intelligence Pipeline",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        # ====================================================
        # PROGRESS
        # ====================================================

        if frame_number % 25 == 0:

            print(
                f"Processed frame "
                f"{frame_number}/{total_frames}"
            )

    # ========================================================
    # RELEASE VIDEO RESOURCES
    # ========================================================

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 70)

    # ========================================================
    # FINALIZE ROAD DAMAGE
    # ========================================================

    print("\nFinalizing Road Damage analysis...")

    road_damage_result = (
        finalize_road_damage(
            road_track_history,
            ROAD_DAMAGE_OUTPUT
        )
    )

    # ========================================================
    # FINALIZE ACCIDENT
    # ========================================================

    print("\nFinalizing Accident analysis...")

    accident_result = (
        finalize_accident(
            accident_event,
            ACCIDENT_OUTPUT
        )
    )
    
 
    # ========================================================
    # FINALIZE WATERLOGGING
    # ========================================================

    print("\nFinalizing Waterlogging analysis...")

    waterlogging_result = (
        finalize_waterlogging(
            waterlogging_track_history,
            WATERLOGGING_OUTPUT
        )
    )

    # ========================================================
    # FINALIZE TRAFFIC
    # ========================================================

    print("\nFinalizing Traffic congestion analysis...")

    traffic_result = (
        finalize_traffic(
            traffic_track_history,
            TRAFFIC_OUTPUT
        )
    ) 
    
    

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput video:\n{OUTPUT_VIDEO}"
    )

    print(
        f"\nRoad damage events:\n"
        f"{ROAD_DAMAGE_OUTPUT}"
    )

    print(
        f"\nAccident events:\n"
        f"{ACCIDENT_OUTPUT}"
    )
    
    
    print(
        f"\nWaterlogging events:\n"
        f"{WATERLOGGING_OUTPUT}"
    )

    print(
        f"\nTraffic congestion events:\n"
        f"{TRAFFIC_OUTPUT}"
    )    


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()