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

from anpr.model import load_anpr_model
from anpr.detector import (
    process_anpr_frame,
    finalize_anpr
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

from waterlogging.config import FRAME_SAMPLE_RATE as WATERLOGGING_FRAME_SAMPLE_RATE
from traffic.config import FRAME_SAMPLE_RATE as TRAFFIC_FRAME_SAMPLE_RATE


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

INPUT_VIDEO = VIDEO_INPUT_DIR / "r1.mp4"

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

ANPR_OUTPUT = (
    EVENT_OUTPUT_DIR /
    "anpr_events.json"
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
    accident_detected,
    unique_plate_count,
    latest_plate_text,
    waterlogging_detections,
    waterlogging_severity,
    traffic_congestion_level
):
    """
    Compact dashboard HUD - keeps total overlay height small so
    the video itself stays visible.
    """

    height, width = frame.shape[:2]

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (190, 190, 190)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    ORANGE = (0, 165, 255)

    # =========================================================
    # LAYOUT (all short, on purpose)
    # =========================================================

    header_height = 65
    panel_top = 70
    panel_height = 85
    panel_bottom = panel_top + panel_height   # 155
    panel_gap = 12

    strip_top = panel_bottom + 5              # 160
    strip_bottom = strip_top + 34             # 194

    status_height = 45

    panel_width = (width - 4 * panel_gap) // 3

    # =========================================================
    # ROAD DAMAGE COUNTS
    # =========================================================

    potholes = sum(1 for d in road_detections if d.get("damage_type") == "D40")
    longitudinal = sum(1 for d in road_detections if d.get("damage_type") == "D00")
    transverse = sum(1 for d in road_detections if d.get("damage_type") == "D10")
    alligator = sum(1 for d in road_detections if d.get("damage_type") == "D20")
    repairs = sum(1 for d in road_detections if d.get("damage_type") == "Repair")
    cracks = longitudinal + transverse + alligator

    # =========================================================
    # HEADER
    # =========================================================

    cv2.rectangle(frame, (0, 0), (width, header_height), BLACK, -1)

    cv2.putText(
        frame, "AI URBAN INTELLIGENCE", (25, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2
    )

    cv2.putText(
        frame, "MOBILE URBAN INTELLIGENCE PLATFORM", (25, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1
    )

    cv2.circle(frame, (width - 90, 28), 7, GREEN, -1)
    cv2.putText(
        frame, "LIVE", (width - 75, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2
    )

    # =========================================================
    # PANEL POSITIONS
    # =========================================================

    road_x1 = panel_gap
    road_x2 = road_x1 + panel_width

    accident_x1 = road_x2 + panel_gap
    accident_x2 = accident_x1 + panel_width

    anpr_x1 = accident_x2 + panel_gap
    anpr_x2 = width - panel_gap

    for x1, x2 in ((road_x1, road_x2), (accident_x1, accident_x2), (anpr_x1, anpr_x2)):
        cv2.rectangle(frame, (x1, panel_top), (x2, panel_bottom), BLACK, -1)

    # ---------------------------------------------------------
    # ROAD DAMAGE (title + 1 combined counts line + repairs line)
    # ---------------------------------------------------------

    cv2.putText(
        frame, "ROAD DAMAGE", (road_x1 + 15, panel_top + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2
    )
    cv2.putText(
        frame, f"Obj:{len(road_detections)}  Pot:{potholes}  Crk:{cracks}",
        (road_x1 + 15, panel_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW if (potholes or cracks) else WHITE, 1
    )
    cv2.putText(
        frame, f"Repairs: {repairs}",
        (road_x1 + 15, panel_top + 74),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1
    )

    # ---------------------------------------------------------
    # ACCIDENT / TRAFFIC (title + status line + veh/col/congestion)
    # ---------------------------------------------------------

    status_text = "ACCIDENT DETECTED" if accident_detected else "MONITORING"
    status_color = RED if accident_detected else GREEN

    cv2.putText(
        frame, "ACCIDENT / TRAFFIC", (accident_x1 + 15, panel_top + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2
    )
    cv2.putText(
        frame, status_text, (accident_x1 + 15, panel_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2
    )
    cv2.putText(
        frame,
        f"Veh:{vehicle_count}  Col:{collision_count}  Cong:{traffic_congestion_level}",
        (accident_x1 + 15, panel_top + 74),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
        RED if collision_count > 0 else WHITE, 1
    )

    # ---------------------------------------------------------
    # ANPR (title + unique/plate combined line)
    # ---------------------------------------------------------

    latest_text = latest_plate_text if latest_plate_text else "NO PLATE"

    cv2.putText(
        frame, "NUMBER PLATE / ANPR", (anpr_x1 + 15, panel_top + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2
    )
    cv2.putText(
        frame, f"Unique: {unique_plate_count}", (anpr_x1 + 15, panel_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2
    )
    cv2.putText(
        frame, f"Plate: {latest_text}", (anpr_x1 + 15, panel_top + 74),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW, 1
    )

    # =========================================================
    # WATERLOGGING STRIP - directly under the panels, not floating
    # at the bottom, so it reads as part of the same HUD block
    # =========================================================

    cv2.rectangle(frame, (panel_gap, strip_top), (width - panel_gap, strip_bottom), BLACK, -1)

    water_status = waterlogging_severity if waterlogging_severity else "CLEAR"
    water_color = ORANGE if waterlogging_severity else GREEN

    cv2.putText(
        frame, "WATERLOGGING:", (25, strip_bottom - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2
    )
    cv2.putText(
        frame, water_status, (230, strip_bottom - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, water_color, 2
    )
    cv2.putText(
        frame, f"Regions: {len(waterlogging_detections)}", (400, strip_bottom - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1
    )

    # =========================================================
    # BOTTOM STATUS BAR (shorter, single line)
    # =========================================================

    cv2.rectangle(frame, (0, height - status_height), (width, height), BLACK, -1)

    minutes = int(timestamp_seconds // 60)
    seconds = int(timestamp_seconds % 60)
    bottom_y = height - 15

    cv2.putText(frame, f"F:{frame_number}", (20, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, f"T:{minutes:02d}:{seconds:02d}", (150, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, f"Veh:{vehicle_count}", (280, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, f"RoadEvt:{len(road_detections)}", (400, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, f"Plates:{unique_plate_count}", (580, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, "SYSTEM: ACTIVE", (width - 220, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

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

    print("\n[1/5] Loading Road Damage model...")

    road_damage_model = (
        load_road_damage_model()
    )

    print(
        "Road Damage model loaded successfully."
    )

    print("\n[2/5] Loading Accident model...")

    accident_model = (
        load_accident_model()
    )

    print(
        "Accident model loaded successfully."
    )

    print("\n[3/5] Loading ANPR model...")

    anpr_model = (
        load_anpr_model()
    )

    print(
        "ANPR model loaded successfully."
    )
    
    
    print("\n[4/5] Loading Waterlogging detector...")

    waterlogging_model = (
        load_waterlogging_model()
    )

    print("\n[5/5] Loading Traffic congestion model...")

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
    # ANPR STATE
    # ========================================================

    plate_history = {}

    latest_plate_text = None
    
    # ========================================================
    # WATERLOGGING STATE
    # ========================================================

    waterlogging_track_history = {}
    waterlogging_cache_detections = []
    waterlogging_cache_severity = None

    traffic_cache_count = 0
    traffic_cache_congestion = "LOW"    
    

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

        # ====================================================
        # ANPR
        # ====================================================

        (
            frame,
            plate_reads_this_frame,
            unique_plate_count
        ) = process_anpr_frame(
            anpr_model,
            frame,
            frame_number,
            timestamp_seconds,
            plate_history
        )

        if plate_reads_this_frame:
            for read in plate_reads_this_frame:
                if read["plate_text"]:
                    latest_plate_text = read["plate_text"]
                    
     
     
        # ====================================================
        # WATERLOGGING
        # ====================================================

        # ====================================================
        # WATERLOGGING (sampled - not every frame)
        # ====================================================

        if frame_number % WATERLOGGING_FRAME_SAMPLE_RATE == 0:
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
            waterlogging_cache_detections = waterlogging_detections
            waterlogging_cache_severity = waterlogging_track_history.get("last_severity")
        else:
            waterlogging_detections = waterlogging_cache_detections

        # ====================================================
        # TRAFFIC CONGESTION
        # ====================================================

        # ====================================================
        # TRAFFIC CONGESTION (sampled - not every frame)
        # ====================================================

        if frame_number % TRAFFIC_FRAME_SAMPLE_RATE == 0:
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
            traffic_cache_count = traffic_vehicle_count
            traffic_cache_congestion = congestion_level
        else:
            traffic_vehicle_count = traffic_cache_count
            congestion_level = traffic_cache_congestion
     
     
                    
                    

        frame = draw_pipeline_hud(
        frame=frame,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        road_detections=road_detections,
        vehicle_count=vehicle_count,
        collision_count=collision_counter,
        prolonged_collision=prolonged_collision_counter,
        accident_detected=accident_detected,
        unique_plate_count=unique_plate_count,
        latest_plate_text=latest_plate_text,
        waterlogging_detections=waterlogging_detections,
        waterlogging_severity=waterlogging_track_history.get("last_severity"),
        traffic_congestion_level=congestion_level
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



        if frame_number == 1:

            cv2.namedWindow(
                "AI Urban Intelligence - Integrated Pipeline",
                cv2.WINDOW_NORMAL
            )

            cv2.resizeWindow(
                "AI Urban Intelligence - Integrated Pipeline",
                1600,
                900
            )


        if True:

            cv2.imshow(
                "AI Urban Intelligence - Integrated Pipeline",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):

                print(
                    "\nUser requested pipeline stop."
                )

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
    # FINALIZE ANPR
    # ========================================================

    print("\nFinalizing ANPR analysis...")

    anpr_result = (
        finalize_anpr(
            plate_history,
            ANPR_OUTPUT
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
        f"\nANPR events:\n"
        f"{ANPR_OUTPUT}"
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
Explain me how to create video in the laptop by covering the front end back end. What are the platform by to record the screen? How to do screen recording in the laptop windows? Is there any inbuilt screen recording features? Can we record them or yeah, suggest that that how to do that and yep. And also structured how to show the all project to the SIH by doing this. How to do these things. Like how able to do this.