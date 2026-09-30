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
    latest_plate_text
    ):
    """
    Large and clearly readable dashboard HUD.
    Designed for 4K video displayed in a smaller window.
    """

    height, width = frame.shape[:2]

    # =========================================================
    # COLORS
    # =========================================================

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (190, 190, 190)

    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    BLUE = (255, 120, 0)

    # =========================================================
    # LAYOUT
    # =========================================================

    header_height = 150

    panel_top = 165
    panel_bottom = 385

    panel_gap = 20

    panel_width = (
        width - 4 * panel_gap
    ) // 3

    # =========================================================
    # HEADER
    # =========================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (width, header_height),
        BLACK,
        -1
    )

    cv2.putText(
        frame,
        "AI URBAN INTELLIGENCE",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        WHITE,
        3
    )

    cv2.putText(
        frame,
        "MOBILE URBAN INTELLIGENCE PLATFORM",
        (42, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        GRAY,
        2
    )

    # LIVE indicator

    cv2.circle(
        frame,
        (width - 125, 50),
        10,
        GREEN,
        -1
    )

    cv2.putText(
        frame,
        "LIVE",
        (width - 105, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        GREEN,
        3
    )

    # =========================================================
    # ROAD DAMAGE COUNTS
    # =========================================================

    potholes = sum(
        1
        for d in road_detections
        if d.get("damage_type") == "D40"
    )

    longitudinal = sum(
        1
        for d in road_detections
        if d.get("damage_type") == "D00"
    )

    transverse = sum(
        1
        for d in road_detections
        if d.get("damage_type") == "D10"
    )

    alligator = sum(
        1
        for d in road_detections
        if d.get("damage_type") == "D20"
    )

    repairs = sum(
        1
        for d in road_detections
        if d.get("damage_type") == "Repair"
    )

    cracks = (
        longitudinal
        + transverse
        + alligator
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

    # =========================================================
    # DRAW PANELS
    # =========================================================

    cv2.rectangle(
        frame,
        (road_x1, panel_top),
        (road_x2, panel_bottom),
        BLACK,
        -1
    )

    cv2.rectangle(
        frame,
        (accident_x1, panel_top),
        (accident_x2, panel_bottom),
        BLACK,
        -1
    )

    cv2.rectangle(
        frame,
        (anpr_x1, panel_top),
        (anpr_x2, panel_bottom),
        BLACK,
        -1
    )

    # =========================================================
    # ROAD DAMAGE
    # =========================================================

    cv2.putText(
        frame,
        "ROAD DAMAGE",
        (road_x1 + 25, panel_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        WHITE,
        3
    )

    cv2.putText(
        frame,
        f"TOTAL OBJECTS: {len(road_detections)}",
        (road_x1 + 25, panel_top + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"POTHOLES: {potholes}",
        (road_x1 + 25, panel_top + 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        YELLOW if potholes else WHITE,
        2
    )

    cv2.putText(
        frame,
        f"CRACKS: {cracks}",
        (road_x1 + 250, panel_top + 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        YELLOW if cracks else WHITE,
        2
    )

    cv2.putText(
        frame,
        f"REPAIRS: {repairs}",
        (road_x1 + 25, panel_top + 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2
    )

    # =========================================================
    # ACCIDENT / TRAFFIC
    # =========================================================

    cv2.putText(
        frame,
        "ACCIDENT / TRAFFIC",
        (accident_x1 + 25, panel_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        WHITE,
        3
    )

    # ---------------------------------------------------------
    # ACCIDENT STATUS
    # ---------------------------------------------------------

    if accident_detected:

        status_text = "ACCIDENT DETECTED"
        status_color = RED

    else:

        status_text = "MONITORING"
        status_color = GREEN

    cv2.putText(
        frame,
        f"STATUS: {status_text}",
        (accident_x1 + 25, panel_top + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_color,
        3
    )

    # ---------------------------------------------------------
    # VEHICLE COUNT
    # ---------------------------------------------------------

    cv2.putText(
        frame,
        f"VEHICLES: {vehicle_count}",
        (accident_x1 + 25, panel_top + 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        WHITE,
        2
    )

    # ---------------------------------------------------------
    # COLLISION COUNT
    # ---------------------------------------------------------

    cv2.putText(
        frame,
        f"COLLISIONS: {collision_count}",
        (accident_x1 + 25, panel_top + 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        RED if collision_count > 0 else WHITE,
        2
    )

    # =========================================================
    # NUMBER PLATE / ANPR
    # =========================================================

    cv2.putText(
        frame,
        "NUMBER PLATE / ANPR",
        (anpr_x1 + 25, panel_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        WHITE,
        3
    )

    # ---------------------------------------------------------
    # UNIQUE VEHICLES
    # ---------------------------------------------------------

    cv2.putText(
        frame,
        "UNIQUE VEHICLES",
        (anpr_x1 + 25, panel_top + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        GRAY,
        2
    )

    cv2.putText(
        frame,
        str(unique_plate_count),
        (anpr_x1 + 25, panel_top + 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        GREEN,
        3
    )

    # ---------------------------------------------------------
    # LATEST PLATE
    # ---------------------------------------------------------

    cv2.putText(
        frame,
        "LATEST PLATE",
        (anpr_x1 + 180, panel_top + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        GRAY,
        2
    )

    latest_text = (
        latest_plate_text
        if latest_plate_text
        else "NO PLATE"
    )

    cv2.putText(
        frame,
        latest_text,
        (anpr_x1 + 180, panel_top + 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        YELLOW,
        3
    )

    # =========================================================
    # BOTTOM STATUS BAR
    # =========================================================

    status_height = 80

    cv2.rectangle(
        frame,
        (0, height - status_height),
        (width, height),
        BLACK,
        -1
    )

    minutes = int(
        timestamp_seconds // 60
    )

    seconds = int(
        timestamp_seconds % 60
    )

    bottom_y = height - 30

    cv2.putText(
        frame,
        f"FRAME: {frame_number}",
        (30, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"TIME: {minutes:02d}:{seconds:02d}",
        (280, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"VEHICLES: {vehicle_count}",
        (520, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"ROAD EVENTS: {len(road_detections)}",
        (770, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        f"PLATES: {unique_plate_count}",
        (1060, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        "SYSTEM: ACTIVE",
        (1350, bottom_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        GREEN,
        2
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

    print("\n[1/3] Loading Road Damage model...")

    road_damage_model = (
        load_road_damage_model()
    )

    print(
        "Road Damage model loaded successfully."
    )

    print("\n[2/3] Loading Accident model...")

    accident_model = (
        load_accident_model()
    )

    print(
        "Accident model loaded successfully."
    )

    print("\n[3/3] Loading ANPR model...")

    anpr_model = (
        load_anpr_model()
    )

    print(
        "ANPR model loaded successfully."
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
        latest_plate_text=latest_plate_text
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
                1280,
                720
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
