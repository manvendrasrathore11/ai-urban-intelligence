import cv2
import json

from .config import (
    CONFIDENCE_THRESHOLD,
    SHOW_WINDOW,
    WINDOW_NAME,
    EXIT_KEYS,
    BUS_ID,
    GPS_START_LATITUDE,
    GPS_START_LONGITUDE
)

from .gps import GPSProvider


# ============================================================
# ROAD DAMAGE CLASS NAMES
# ============================================================

CLASS_NAMES = {
    "D00": "LONGITUDINAL CRACK",
    "D10": "TRANSVERSE CRACK",
    "D20": "ALLIGATOR CRACK",
    "D40": "POTHOLE",
    "Repair": "REPAIRED AREA"
}


# ============================================================
# TIME FORMATTER
# ============================================================

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


# ============================================================
# HUD
# ============================================================

def draw_hud(frame, frame_number, timestamp_seconds, detections):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # TOP BAR
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
        "ROAD DAMAGE DETECTION + TRACKING",
        (20, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # BOTTOM BAR
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
        f"OBJECTS: {len(detections)}",
        (400, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ============================================================
# UPDATE TRACK HISTORY
# ============================================================

def update_track_history(track_history, detection):

    track_id = detection["track_id"]

    # --------------------------------------------------------
    # No tracking ID
    # --------------------------------------------------------

    if track_id is None:
        return

    # --------------------------------------------------------
    # First time we see this track
    # --------------------------------------------------------

    if track_id not in track_history:

        track_history[track_id] = {

            "track_id": track_id,

            "damage_type": detection["damage_type"],

            "damage_name": detection["damage_name"],

            "first_seen_frame":
                detection["frame_number"],

            "last_seen_frame":
                detection["frame_number"],

            "first_seen_timestamp":
                detection["timestamp_seconds"],

            "last_seen_timestamp":
                detection["timestamp_seconds"],

            "max_confidence":
                detection["confidence"],

            "observation_count": 1,
            
            "gps_observations": [

                {
                    "frame_number":
                        detection["frame_number"],

                    "timestamp_seconds":
                        detection["timestamp_seconds"],

                    "latitude":
                        detection["location"]["latitude"],

                    "longitude":
                        detection["location"]["longitude"]
                }

            ]
        }

        return

    # --------------------------------------------------------
    # Existing track
    # --------------------------------------------------------

    track = track_history[track_id]

    track["last_seen_frame"] = \
        detection["frame_number"]

    track["last_seen_timestamp"] = \
        detection["timestamp_seconds"]

    track["max_confidence"] = max(
        track["max_confidence"],
        detection["confidence"]
    )

    track["observation_count"] += 1
    
    
    track["gps_observations"].append({

        "frame_number":
            detection["frame_number"],

        "timestamp_seconds":
            detection["timestamp_seconds"],

        "latitude":
            detection["location"]["latitude"],

        "longitude":
            detection["location"]["longitude"]
    })    


# ============================================================
# CREATE DAMAGE CANDIDATES
# ============================================================

def create_damage_candidates(track_history):

    candidates = []

    for track_id, track in track_history.items():

        candidate = {

            "candidate_id":
                f"DAMAGE_CANDIDATE_{track_id}",

            "track_id":
                track["track_id"],

            "damage_type":
                track["damage_type"],

            "damage_name":
                track["damage_name"],

            "first_seen_frame":
                track["first_seen_frame"],

            "last_seen_frame":
                track["last_seen_frame"],

            "first_seen_timestamp":
                track["first_seen_timestamp"],

            "last_seen_timestamp":
                track["last_seen_timestamp"],

            "max_confidence":
                track["max_confidence"],

            "observation_count":
                track["observation_count"],
                
            "bus_id":
                BUS_ID,    
                
            "gps_observations":
               track["gps_observations"],    
                

            # ------------------------------------------------
            # Evidence reference
            # ------------------------------------------------

            "evidence": {

                "start_frame":
                    track["first_seen_frame"],

                "end_frame":
                    track["last_seen_frame"],

                "start_timestamp":
                    track["first_seen_timestamp"],

                "end_timestamp":
                    track["last_seen_timestamp"]
            }
        }

        candidates.append(candidate)

    return candidates


# ============================================================
# CREATE VIDEO SUMMARY
# ============================================================

def create_video_summary(candidates):

    damage_by_type = {}

    for candidate in candidates:

        damage_name = candidate["damage_name"]

        if damage_name not in damage_by_type:

            damage_by_type[damage_name] = 0

        damage_by_type[damage_name] += 1

    summary = {

        "total_damage_candidates":
            len(candidates),

        "damage_by_type":
            damage_by_type
    }

    return summary


# ============================================================
# MAIN ROAD DAMAGE DETECTION
# ============================================================

def detect_road_damage(
    model,
    video_path,
    output_video_path,
    output_events_path
):

    print("\nOpening video...")

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
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
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

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(
        f"Resolution: "
        f"{width} x {height}"
    )

    # --------------------------------------------------------
    # OUTPUT VIDEO
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
    # TRACK MEMORY
    # --------------------------------------------------------

    track_history = {}

    frame_number = 0
    
    gps_provider = GPSProvider(
    start_latitude=GPS_START_LATITUDE,
    start_longitude=GPS_START_LONGITUDE
    )

    print(
        "\nStarting "
        "YOLO + ByteTrack + "
        "Road Damage Evidence Analysis...\n"
    )

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        timestamp_seconds = (
            frame_number - 1
        ) / fps
        
        gps_location = gps_provider.get_location(
            frame_number
      )

        # ----------------------------------------------------
        # YOLO + BYTETRACK
        # ----------------------------------------------------

        results = model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=CONFIDENCE_THRESHOLD,

            verbose=False
        )

        result = results[0]

        frame_detections = []

        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        if result.boxes is not None:

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

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
                # TRACK ID
                # ------------------------------------------------

                track_id = None

                if box.id is not None:

                    track_id = int(
                        box.id[0]
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
                # OBSERVATION
                # ------------------------------------------------

                detection = {

                    "event_type":
                        "ROAD_DAMAGE_OBSERVATION",

                    "damage_type":
                        class_name,

                    "damage_name":
                        friendly_name,

                    "track_id":
                        track_id,

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
                        ),
                    "bus_id":
                        BUS_ID,

                    "location": {

                        "latitude":
                            gps_location["latitude"],

                        "longitude":
                           gps_location["longitude"]    
                }
            }

                frame_detections.append(
                    detection
                )

                # ------------------------------------------------
                # UPDATE TRACK MEMORY
                # ------------------------------------------------

                update_track_history(
                    track_history,
                    detection
                )

                # =================================================
                # DRAW DETECTION
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
                # LABEL
                # ------------------------------------------------

                if track_id is not None:

                    label = (
                        f"{friendly_name} | "
                        f"ID {track_id} | "
                        f"{confidence * 100:.0f}%"
                    )

                else:

                    label = (
                        f"{friendly_name} | "
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

                    0.5,

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

        # ----------------------------------------------------
        # SAVE FRAME
        # ----------------------------------------------------

        writer.write(frame)

        # ====================================================
        # SHOW LIVE VIDEO
        # ====================================================

        if SHOW_WINDOW:

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in EXIT_KEYS:

                print(
                    "\nProcessing "
                    "stopped by user."
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
                f"{frame_number}/"
                f"{total_frames} | "

                f"Damage candidates: "
                f"{len(track_history)}"
            )

    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    # ========================================================
    # STAGE 4A
    # CREATE DAMAGE CANDIDATES
    # ========================================================

    damage_candidates = (
        create_damage_candidates(
            track_history
        )
    )

    # ========================================================
    # CREATE SUMMARY
    # ========================================================

    video_summary = (
        create_video_summary(
            damage_candidates
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    final_output = {

        "analysis_type":
            "ROAD_DAMAGE_EVIDENCE",

        "video_summary":
            video_summary,

        "damage_candidates":
            damage_candidates
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        output_events_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_output,
            file,
            indent=4
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "ROAD DAMAGE EVIDENCE ANALYSIS COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        f"\nTotal damage candidates: "
        f"{len(damage_candidates)}"
    )

    print(
        "\nDamage by type:"
    )

    for damage_type, count in (
        video_summary[
            "damage_by_type"
        ].items()
    ):

        print(
            f"  {damage_type}: {count}"
        )

    print(
        f"\nAnnotated video saved to:"
        f"\n{output_video_path}"
    )

    print(
        f"\nEvidence JSON saved to:"
        f"\n{output_events_path}"
    )

    # ========================================================
    # CANDIDATE DETAILS
    # ========================================================

    print(
        "\nDamage Candidate Details:"
    )

    for candidate in damage_candidates:

        print(

            f"\nCandidate: "
            f"{candidate['candidate_id']}"

        )

        print(

            f"  Type: "
            f"{candidate['damage_name']}"

        )

        print(

            f"  Frames: "
            f"{candidate['first_seen_frame']}"
            f"-"
            f"{candidate['last_seen_frame']}"

        )

        print(

            f"  Observations: "
            f"{candidate['observation_count']}"

        )

        print(

            f"  Best confidence: "
            f"{candidate['max_confidence']:.2f}"

        )

    return final_output