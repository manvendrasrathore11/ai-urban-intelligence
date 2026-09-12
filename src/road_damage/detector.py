import cv2

from .config import CONFIDENCE_THRESHOLD


def detect_road_damage(
    model,
    video_path,
    output_video_path,
    output_events_path
):
    """
    Process a video frame-by-frame using the road damage model.

    The function:
    1. Opens the input video.
    2. Runs YOLO on every frame.
    3. Draws detections.
    4. Saves an annotated video.
    5. Saves structured detection events as JSON.
    """

    print("\nOpening video...")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    # --------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

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

    # --------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------
    # EVENT STORAGE
    # --------------------------------------------------

    events = []

    frame_number = 0

    print("\nStarting detection...\n")

    # --------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # --------------------------------------------------
        # TIMESTAMP
        # --------------------------------------------------

        timestamp_seconds = (
            frame_number / fps
        )

        # --------------------------------------------------
        # YOLO INFERENCE
        # --------------------------------------------------

        results = model(
            frame,
            verbose=False
        )

        result = results[0]

        # --------------------------------------------------
        # PROCESS DETECTIONS
        # --------------------------------------------------

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[
                class_id
            ]

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )

            # --------------------------------------------------
            # CREATE EVENT
            # --------------------------------------------------

            event = {
                "event_type": "ROAD_DAMAGE",
                "damage_type": class_name,
                "confidence": round(
                    confidence,
                    4
                ),
                "bounding_box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                },
                "frame_number": frame_number,
                "timestamp_seconds": round(
                    timestamp_seconds,
                    3
                )
            }

            events.append(event)

            # --------------------------------------------------
            # DRAW BOUNDING BOX
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # --------------------------------------------------
            # DRAW LABEL
            # --------------------------------------------------

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(
                f"Frame {frame_number} | "
                f"{class_name} | "
                f"confidence={confidence:.2f}"
            )

        # --------------------------------------------------
        # SAVE FRAME
        # --------------------------------------------------

        writer.write(frame)

    # --------------------------------------------------
    # CLEAN UP
    # --------------------------------------------------

    cap.release()
    writer.release()

    print("\nVideo processing completed.")

    # --------------------------------------------------
    # SAVE JSON EVENTS
    # --------------------------------------------------

    import json

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

    print(
        f"Annotated video saved to:\n"
        f"{output_video_path}"
    )

    print(
        f"Events saved to:\n"
        f"{output_events_path}"
    )

    print(
        f"\nTotal detections: "
        f"{len(events)}"
    )

    return events