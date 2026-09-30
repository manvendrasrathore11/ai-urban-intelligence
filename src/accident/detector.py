import cv2
import json
import time
from datetime import datetime
from pathlib import Path
from emergency_services.osm import alert_nearest_services
from alerts.email_alert import send_accident_email

from .config import (
    CONFIDENCE_THRESHOLD,
    IOU_COLLISION_THRESHOLD,
    PROLONGED_COLLISION_FRAMES,
    SHOW_WINDOW,
    WINDOW_NAME,
    EXIT_KEYS,
    VEHICLE_CLASSES,
    BUS_ID,
    ACCIDENT_EVIDENCE_DIR,
    GPS_LOCATION,
)


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    box1_area = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    box2_area = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0

    return intersection_area / union_area


# ============================================================
# SEVERITY
# ============================================================

def estimate_severity(collision_count, vehicle_count):

    if collision_count >= 5:
        return "HIGH"

    if collision_count >= 2:
        return "MEDIUM"

    if vehicle_count >= 4:
        return "MEDIUM"

    return "LOW"


# ============================================================
# EVENT CREATION
# ============================================================

def create_accident_event(
    frame_number,
    timestamp_seconds,
    collision_count,
    vehicle_count,
    severity,
    evidence_path=None,
):

    return {
        "event_type": "ACCIDENT",

        "event_id": f"ACC-{int(time.time())}",

        "bus_id": BUS_ID,

        "timestamp": datetime.now().isoformat(),

        "frame_number": frame_number,

        "timestamp_seconds": round(timestamp_seconds, 2),

        "collision_count": collision_count,

        "vehicle_count": vehicle_count,

        "severity": severity,

        "gps": {
            "latitude": GPS_LOCATION["latitude"],
            "longitude": GPS_LOCATION["longitude"],
        },

        "evidence": {
            "image": str(evidence_path) if evidence_path else None
        },
    }


# ============================================================
# SAVE EVENT
# ============================================================

def save_event(event, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_events = []

    if output_path.exists():

        try:

            with open(output_path, "r") as f:
                existing_events = json.load(f)

        except json.JSONDecodeError:

            existing_events = []

    existing_events.append(event)

    with open(output_path, "w") as f:

        json.dump(
            existing_events,
            f,
            indent=4
        )
        
        
def process_accident_frame(
    model,
    frame,
    frame_number,
    timestamp_seconds,
    collision_counter,
    prolonged_collision_counter,
    accident_detected
):
    """
    Process ONE video frame for accident detection.

    This function does not open or read the video.

    Returns:
        annotated_frame
        collision_counter
        prolonged_collision_counter
        accident_detected
        frame_vehicle_count
        frame_collision_count
    """

    # ============================================================
    # RUN YOLO TRACKING
    # ============================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    vehicles = []

    # ============================================================
    # PROCESS DETECTIONS
    # ============================================================

    if results and len(results) > 0:

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence = float(
                    boxes.conf[i].item()
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # ------------------------------------------------
                # CLASS ID
                # ------------------------------------------------

                class_id = int(
                    boxes.cls[i].item()
                )

                # ------------------------------------------------
                # ONLY VEHICLES
                # ------------------------------------------------

                if class_id not in VEHICLE_CLASSES:
                    continue

                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    boxes.xyxy[i].tolist()
                )

                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------

                track_id = None

                if boxes.id is not None:
                    track_id = int(
                        boxes.id[i].item()
                    )

                if track_id is None:
                    track_id = i

                # ------------------------------------------------
                # VEHICLE INFORMATION
                # ------------------------------------------------

                vehicle_name = VEHICLE_CLASSES[
                    class_id
                ]

                vehicle = {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": vehicle_name,
                    "confidence": confidence,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }

                vehicles.append(vehicle)

                # ------------------------------------------------
                # DRAW VEHICLE BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                label = (
                    f"{vehicle_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2
                )

    # ============================================================
    # COLLISION DETECTION
    # ============================================================

    collision_pairs = []

    for i in range(len(vehicles)):

        for j in range(i + 1, len(vehicles)):

            box1 = vehicles[i]["box"]
            box2 = vehicles[j]["box"]

            iou = calculate_iou(
                box1,
                box2
            )

            if iou >= IOU_COLLISION_THRESHOLD:

                collision_pairs.append(
                    {
                        "vehicle_1": vehicles[i]["track_id"],
                        "vehicle_2": vehicles[j]["track_id"],
                        "iou": round(iou, 4)
                    }
                )

    # ============================================================
    # UPDATE COLLISION COUNTERS
    # ============================================================

    frame_collision_count = len(
        collision_pairs
    )

    if frame_collision_count > 0:

        collision_counter += (
            frame_collision_count
        )

        prolonged_collision_counter += 1

    else:

        # Gradually decrease prolonged
        # collision state when vehicles
        # separate.

        prolonged_collision_counter = max(
            0,
            prolonged_collision_counter - 1
        )

    # ============================================================
    # ACCIDENT CONFIRMATION
    # ============================================================

    accident_event = None

    if (
        prolonged_collision_counter
        >= PROLONGED_COLLISION_FRAMES
        and not accident_detected
    ):

        accident_detected = True

        # --------------------------------------------------------
        # ESTIMATE ACCIDENT SEVERITY
        # --------------------------------------------------------

        severity = estimate_severity(
            collision_counter,
            len(vehicles)
        )

        # --------------------------------------------------------
        # SAVE ACCIDENT EVIDENCE IMAGE
        # --------------------------------------------------------

        evidence_path = (
            ACCIDENT_EVIDENCE_DIR /
            f"accident_frame_{frame_number}.jpg"
        )

        cv2.imwrite(
            str(evidence_path),
            frame
        )

        print(
            f"Accident evidence saved: {evidence_path}"
        )

        # --------------------------------------------------------
        # CREATE ACCIDENT EVENT
        # --------------------------------------------------------

        accident_event = create_accident_event(
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            collision_count=collision_counter,
            vehicle_count=len(vehicles),
            severity=severity,
            evidence_path=evidence_path
        )

    # ============================================================
    # DRAW COLLISION INFORMATION
    # ============================================================

    # cv2.putText(
    #     frame,
    #     f"Vehicles: {len(vehicles)}",
    #     (20, 35),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 255, 255),
    #     2
    # )

    # cv2.putText(
    #     frame,
    #     f"Collisions: {frame_collision_count}",
    #     (20, 65),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 255, 255),
    #     2
    # )

    # cv2.putText(
    #     frame,
    #     f"Prolonged: {prolonged_collision_counter}",
    #     (20, 95),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 255, 255),
    #     2
    # )

    # ============================================================
    # ACCIDENT ALERT
    # ============================================================

    if accident_detected:

        cv2.putText(
            frame,
            "ACCIDENT DETECTED",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    # ============================================================
    # RETURN
    # ============================================================

    return (
        frame,
        collision_counter,
        prolonged_collision_counter,
        accident_detected,
        len(vehicles),
        frame_collision_count,
        accident_event
    )        
        
def finalize_accident(
    accident_event,
    output_events_path
):
    """
    Finalize accident detection after the complete video
    has been processed.

    This function does not read video frames.
    """

    print("\n" + "=" * 60)
    print("ACCIDENT DETECTION FINALIZATION")
    print("=" * 60)

    # ============================================================
    # NO ACCIDENT
    # ============================================================

    if accident_event is None:

        print("\nNo confirmed accident detected.")
        print("=" * 60)

        return None

    # ============================================================
    # OUTPUT PATH
    # ============================================================

    output_events_path = Path(
        output_events_path
    )

    output_events_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # FIND NEAREST EMERGENCY SERVICES
    # ============================================================

    print("\nFinding nearest emergency services...")

    accident_location = (
        accident_event["gps"]["latitude"],
        accident_event["gps"]["longitude"]
    )

    nearest_police, nearest_hospital = (
        alert_nearest_services(
            accident_location
        )
    )

    # ============================================================
    # ADD SERVICES TO EVENT
    # ============================================================

    accident_event["nearest_services"] = {

        "police": nearest_police,

        "hospital": nearest_hospital
    }

    # ============================================================
    # SEND EMAIL ALERT
    # ============================================================

    print("\nSending accident alert email...")

    email_sent = send_accident_email(
        accident_event
    )

    accident_event["email_alert"] = {
        "sent": email_sent
    }

    # ============================================================
    # SAVE EVENT
    # ============================================================

    save_event(
        accident_event,
        output_events_path
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\nAccident event confirmed.")

    print(
        f"Event ID: "
        f"{accident_event.get('event_id')}"
    )

    print(
        f"Frame: "
        f"{accident_event.get('frame_number')}"
    )

    print(
        f"Time: "
        f"{accident_event.get('timestamp_seconds')} seconds"
    )

    print(
        f"Collision count: "
        f"{accident_event.get('collision_count')}"
    )

    print(
        f"Vehicle count: "
        f"{accident_event.get('vehicle_count')}"
    )

    print(
        f"Severity: "
        f"{accident_event.get('severity')}"
    )

    print(
        f"GPS: "
        f"{accident_event.get('gps')}"
    )

    print(
        f"\nAccident event saved to:"
        f"\n{output_events_path}"
    )

    print("=" * 60)
    print("ACCIDENT FINALIZATION COMPLETE")
    print("=" * 60)

    return accident_event        


# ============================================================
# ACCIDENT DETECTION
# ============================================================

def detect_accidents(model, video_path, output_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print()
    print("=" * 60)
    print("ACCIDENT DETECTION STARTED")
    print("=" * 60)

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width} x {height}")

    # --------------------------------------------------------
    # Output video
    # --------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    frame_number = 0

    collision_counter = 0

    prolonged_collision_counter = 0

    accident_detected = False

    accident_event = None

    evidence_path = None

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        timestamp_seconds = frame_number / fps

        # ----------------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        boxes = result.boxes

        vehicle_boxes = []

        vehicle_ids = []

        # ----------------------------------------------------
        # Extract vehicles
        # ----------------------------------------------------

        if boxes is not None:

            for i in range(len(boxes)):

                cls_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                if cls_id not in VEHICLE_CLASSES:
                    continue

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()

                vehicle_boxes.append(xyxy)

                if boxes.id is not None:

                    vehicle_id = int(
                        boxes.id[i].item()
                    )

                else:

                    vehicle_id = i

                vehicle_ids.append(vehicle_id)

        vehicle_count = len(vehicle_boxes)

        # ----------------------------------------------------
        # Collision detection
        # ----------------------------------------------------

        collision_detected = False

        collision_pairs = []

        for i in range(len(vehicle_boxes)):

            for j in range(i + 1, len(vehicle_boxes)):

                iou = calculate_iou(
                    vehicle_boxes[i],
                    vehicle_boxes[j]
                )

                if iou >= IOU_COLLISION_THRESHOLD:

                    collision_detected = True

                    collision_pairs.append(
                        (
                            vehicle_ids[i],
                            vehicle_ids[j]
                        )
                    )

        # ----------------------------------------------------
        # Prolonged collision
        # ----------------------------------------------------

        if collision_detected:

            collision_counter += 1

            prolonged_collision_counter += 1

        else:

            prolonged_collision_counter = max(
                0,
                prolonged_collision_counter - 1
            )

        # ----------------------------------------------------
        # Accident confirmation
        # ----------------------------------------------------

        if (
            prolonged_collision_counter
            >= PROLONGED_COLLISION_FRAMES
            and not accident_detected
        ):

            accident_detected = True

            severity = estimate_severity(
                collision_counter,
                vehicle_count
            )

            print()
            print("🚨 ACCIDENT DETECTED")
            print(
                f"Frame: {frame_number}"
            )
            print(
                f"Vehicles: {vehicle_count}"
            )
            print(
                f"Collision frames: "
                f"{collision_counter}"
            )
            print(
                f"Severity: {severity}"
            )

            # ------------------------------------------------
            # Evidence
            # ------------------------------------------------

            evidence_dir = (
                output_path.parent /
                "accident_evidence"
            )

            evidence_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            evidence_path = (
                evidence_dir /
                f"accident_frame_{frame_number}.jpg"
            )

            cv2.imwrite(
                str(evidence_path),
                frame
            )

            # ------------------------------------------------
            # Event JSON
            # ------------------------------------------------

            event_output = (
                output_path.parent.parent /
                "events" /
                "accident_events.json"
            )

            accident_event = create_accident_event(
                frame_number,
                timestamp_seconds,
                collision_counter,
                vehicle_count,
                severity,
                evidence_path
            )

            save_event(
                accident_event,
                event_output
            )

            print(
                f"Evidence saved: {evidence_path}"
            )

            print(
                f"Event saved: {event_output}"
            )

        # ====================================================
        # VISUALIZATION
        # ====================================================

        annotated_frame = frame.copy()

        # ----------------------------------------------------
        # Draw vehicle boxes
        # ----------------------------------------------------

        if boxes is not None:

            for i in range(len(boxes)):

                cls_id = int(
                    boxes.cls[i].item()
                )

                if cls_id not in VEHICLE_CLASSES:
                    continue

                confidence = float(
                    boxes.conf[i].item()
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                xyxy = boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                if boxes.id is not None:

                    vehicle_id = int(
                        boxes.id[i].item()
                    )

                else:

                    vehicle_id = i

                label = (
                    f"{VEHICLE_CLASSES[cls_id]} "
                    f"ID:{vehicle_id} "
                    f"{confidence:.2f}"
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        # ====================================================
        # TOP INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            annotated_frame,
            (0, 0),
            (width, 145),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            annotated_frame,
            "AI URBAN INTELLIGENCE",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            "EVENT 2: ACCIDENT DETECTION",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Vehicles: {vehicle_count}",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Collision Frames: {collision_counter}",
            (230, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        status = (
            "ACCIDENT DETECTED"
            if accident_detected
            else "MONITORING"
        )

        status_position = (
            500,
            92
        )

        cv2.putText(
            annotated_frame,
            status,
            status_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255)
            if accident_detected
            else (0, 255, 0),
            2
        )

        # ====================================================
        # COLLISION INFORMATION
        # ====================================================

        if collision_detected:

            cv2.putText(
                annotated_frame,
                "⚠ COLLISION DETECTED",
                (20, height - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

        if accident_detected:

            cv2.putText(
                annotated_frame,
                "🚨 ACCIDENT CONFIRMED",
                (20, height - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            cv2.putText(
                annotated_frame,
                f"GPS: {GPS_LOCATION['latitude']}, "
                f"{GPS_LOCATION['longitude']}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # Write output
        # ----------------------------------------------------

        writer.write(
            annotated_frame
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        if SHOW_WINDOW:

            cv2.imshow(
                WINDOW_NAME,
                annotated_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in EXIT_KEYS:
                break

    # ========================================================
    # Cleanup
    # ========================================================

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("ACCIDENT DETECTION COMPLETE")
    print("=" * 60)

    print(
        f"Output video: {output_path}"
    )

    print(
        f"Accident detected: {accident_detected}"
    )

    if accident_event:

        print(
            f"Severity: "
            f"{accident_event['severity']}"
        )

    return accident_event