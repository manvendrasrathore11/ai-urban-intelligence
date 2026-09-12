import cv2
import json

from .config import (
    CONFIDENCE_THRESHOLD,
    SHOW_WINDOW,
    WINDOW_NAME,
    EXIT_KEYS
)


# ============================================================
# FRIENDLY CLASS NAMES
# ============================================================

CLASS_NAMES = {
    "D00": "LONGITUDINAL CRACK",
    "D10": "TRANSVERSE CRACK",
    "D20": "ALLIGATOR CRACK",
    "D40": "POTHOLE",
    "Repair": "REPAIRED AREA"
}


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(seconds):
    """
    Convert seconds into MM:SS format.
    """

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


# ============================================================
# DRAW HUD
# ============================================================

def draw_hud(
    frame,
    frame_number,
    timestamp_seconds,
    detections
):
    """
    Draw information on the video frame.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # TOP HEADER
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "AI URBAN INTELLIGENCE",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "ROAD DAMAGE DETECTION",
        (20, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # --------------------------------------------------------
    # BOTTOM INFORMATION BAR
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, height - 55),
        (width, height),
        (0, 0, 0),
        -1
    )

    video_time = format_time(timestamp_seconds)

    cv2.putText(
        frame,
        f"FRAME: {frame_number}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"TIME: {video_time}",
        (220, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"DETECTIONS: {len(detections)}",
        (400, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ============================================================
# ROAD DAMAGE DETECTOR
# ============================================================

def detect_road_damage(
    model,
    video_path,
    output_video_path,
    output_events_path
):

    print("\nOpening video...")

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            f"Could not open video: {video_path}"
        )


    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Resolution: {width} x {height}")


    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )


    # --------------------------------------------------------
    # EVENT STORAGE
    # --------------------------------------------------------

    events = []

    frame_number = 0

    print("\nStarting detection...\n")


    # ========================================================
    # FRAME PROCESSING LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # READ ONE FRAME
        # ----------------------------------------------------

        success, frame = cap.read()

        if not success:
            break


        frame_number += 1


        # ----------------------------------------------------
        # CALCULATE VIDEO TIME
        # ----------------------------------------------------

        timestamp_seconds = (
            (frame_number - 1) / fps
        )


        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        results = model(
            frame,
            verbose=False
        )

        result = results[0]


        # Store detections from this frame
        frame_detections = []


        # ====================================================
        # PROCESS EACH DETECTION
        # ====================================================

        for box in result.boxes:

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )


            # Ignore weak detections
            if confidence < CONFIDENCE_THRESHOLD:
                continue


            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[
                class_id
            ]


            friendly_name = CLASS_NAMES.get(
                class_name,
                class_name
            )


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )


            # ------------------------------------------------
            # CREATE EVENT
            # ------------------------------------------------

            event = {

                "event_type":
                    "ROAD_DAMAGE",

                "damage_type":
                    class_name,

                "damage_name":
                    friendly_name,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "bounding_box": {

                    "x1":
                        round(x1, 2),

                    "y1":
                        round(y1, 2),

                    "x2":
                        round(x2, 2),

                    "y2":
                        round(y2, 2)
                },

                "frame_number":
                    frame_number,

                "timestamp_seconds":
                    round(
                        timestamp_seconds,
                        3
                    )
            }


            # Add event
            events.append(event)

            frame_detections.append(event)


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,

                (
                    int(x1),
                    int(y1)
                ),

                (
                    int(x2),
                    int(y2)
                ),

                (0, 255, 0),

                2
            )


            # ------------------------------------------------
            # DRAW LABEL
            # ------------------------------------------------

            label = (
                f"{friendly_name} "
                f"{confidence * 100:.0f}%"
            )


            cv2.putText(

                frame,

                label,

                (
                    int(x1),
                    max(
                        int(y1) - 10,
                        20
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 0),

                2
            )


        # ====================================================
        # DRAW HUD
        # ====================================================

        draw_hud(
            frame,
            frame_number,
            timestamp_seconds,
            frame_detections
        )


        # ====================================================
        # SAVE FRAME TO OUTPUT VIDEO
        # ====================================================

        writer.write(frame)


        # ====================================================
        # SHOW LIVE WINDOW
        # ====================================================

        if SHOW_WINDOW:

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # ------------------------------------------------
            # KEYBOARD CONTROL
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key in EXIT_KEYS:

                print(
                    "\nProcessing stopped by user."
                )

                break


        # ====================================================
        # TERMINAL PROGRESS
        # ====================================================

        if frame_number % 25 == 0:

            progress = (
                frame_number /
                total_frames
            ) * 100

            print(
                f"Processing: "
                f"{progress:.1f}% | "
                f"Frame: "
                f"{frame_number}/{total_frames}"
            )


    # ========================================================
    # RELEASE RESOURCES
    # ========================================================

    cap.release()

    writer.release()

    cv2.destroyAllWindows()


    # ========================================================
    # SAVE EVENTS
    # ========================================================

    with open(
        output_events_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            events,
            file,
            indent=4
        )


    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print(
        "\nVideo processing completed."
    )

    print(
        f"\nAnnotated video saved to:\n"
        f"{output_video_path}"
    )

    print(
        f"\nEvents saved to:\n"
        f"{output_events_path}"
    )

    print(
        f"\nTotal frame-level detections: "
        f"{len(events)}"
    )


    return events