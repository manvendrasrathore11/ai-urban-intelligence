import cv2
import json
import math
from datetime import datetime
from pathlib import Path

from .config import (
    CONFIDENCE_THRESHOLD,
    SHOW_WINDOW,
    WINDOW_NAME,
    EXIT_KEYS,
    BUS_ID,
    GPS_START_LATITUDE,
    GPS_START_LONGITUDE,
    EVENT_OUTPUT_DIR,
    ROAD_DAMAGE_EVIDENCE_DIR,
    EVIDENCE_WINDOW_SECONDS,
    MIN_DISTINCT_DAMAGE_OBJECTS
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
# STAGE 4C CONFIGURATION
# ============================================================

# Maximum distance for considering two candidates
# as the same physical road damage.
SPATIAL_MATCH_DISTANCE_METERS = 10.0

EVIDENCE_WINDOW_FRAMES = None


# ============================================================
# STAGE 4D CONFIGURATION
# ============================================================

# History file used to remember previous bus/day confirmations.
HISTORY_FILE = EVENT_OUTPUT_DIR / "road_damage_history.json"


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

def draw_hud(
    frame,
    frame_number,
    timestamp_seconds,
    detections
):

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
    
    
    
    
    
def save_road_damage_evidence(
    frame,
    damage_name,
    track_id,
    frame_number,
    confidence,
    timestamp_seconds,
    gps_location
):
    """
    Save a photograph of detected road damage
    and return its metadata.
    """

    # Convert damage name into a safe filename
    safe_name = damage_name.lower().replace(" ", "_")

    filename = (
        f"{safe_name}_ID_{track_id}_"
        f"frame_{frame_number}.jpg"
    )

    evidence_path = ROAD_DAMAGE_EVIDENCE_DIR / filename

    # Save image
    cv2.imwrite(
        str(evidence_path),
        frame
    )

    # Return metadata
    return {
        "image": str(evidence_path),
        "frame_number": frame_number,
        "timestamp_seconds": round(timestamp_seconds, 2),
        "confidence": round(confidence, 4),
        "location": gps_location
    }    
    
    
    
    


# ============================================================
# UPDATE TRACK HISTORY
# ============================================================

def update_track_history(
    track_history,
    detection
):

    track_id = detection["track_id"]

    # --------------------------------------------------------
    # NO TRACKING ID
    # --------------------------------------------------------

    if track_id is None:
        return

    # --------------------------------------------------------
    # FIRST TIME WE SEE THIS TRACK
    # --------------------------------------------------------

    if track_id not in track_history:

        track_history[track_id] = {

            "track_id":
                track_id,

            "damage_type":
                detection["damage_type"],

            "damage_name":
                detection["damage_name"],

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

            "observation_count":
                1,

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
    # EXISTING TRACK
    # --------------------------------------------------------

    track = track_history[track_id]

    track["last_seen_frame"] = (
        detection["frame_number"]
    )

    track["last_seen_timestamp"] = (
        detection["timestamp_seconds"]
    )

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
# STAGE 4A
# CREATE DAMAGE CANDIDATES
# ============================================================

def create_damage_candidates(track_history):

    candidates = []

    for track_id, track in track_history.items():
        
            # Skip evidence-window metadata.
        if track_id == "_evidence_window":
            continue

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

            # This is tracking evidence only.
            "observation_count":
                track["observation_count"],

            "bus_id":
                BUS_ID,

            "gps_observations":
                track["gps_observations"],

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
# STAGE 4B
# ADD REPRESENTATIVE GPS LOCATION
# ============================================================

def add_representative_locations(candidates):

    for candidate in candidates:

        gps_observations = (
            candidate["gps_observations"]
        )

        if not gps_observations:

            candidate["location"] = None

            continue

        average_latitude = (
            sum(
                observation["latitude"]
                for observation in gps_observations
            )
            / len(gps_observations)
        )

        average_longitude = (
            sum(
                observation["longitude"]
                for observation in gps_observations
            )
            / len(gps_observations)
        )

        candidate["location"] = {

            "latitude":
                round(average_latitude, 6),

            "longitude":
                round(average_longitude, 6)
        }

    return candidates


# ============================================================
# GPS DISTANCE
# ============================================================

def calculate_gps_distance(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2
):

    """
    Calculate approximate distance between two GPS
    coordinates using the Haversine formula.

    Returns:
        Distance in meters.
    """

    EARTH_RADIUS = 6371000

    latitude_1 = math.radians(latitude_1)
    longitude_1 = math.radians(longitude_1)

    latitude_2 = math.radians(latitude_2)
    longitude_2 = math.radians(longitude_2)

    delta_latitude = (
        latitude_2 - latitude_1
    )

    delta_longitude = (
        longitude_2 - longitude_1
    )

    a = (

        math.sin(delta_latitude / 2) ** 2

        +

        math.cos(latitude_1)
        * math.cos(latitude_2)
        * math.sin(delta_longitude / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    distance = EARTH_RADIUS * c

    return distance


# ============================================================
# CHECK WHETHER TWO CANDIDATES MATCH
# ============================================================

def candidates_match(
    candidate_a,
    candidate_b
):

    # --------------------------------------------------------
    # DAMAGE TYPE MUST MATCH
    # --------------------------------------------------------

    if (
        candidate_a["damage_type"]
        !=
        candidate_b["damage_type"]
    ):

        return False

    # --------------------------------------------------------
    # GET LOCATIONS
    # --------------------------------------------------------

    location_a = candidate_a.get(
        "location"
    )

    location_b = candidate_b.get(
        "location"
    )

    if location_a is None:
        return False

    if location_b is None:
        return False

    # --------------------------------------------------------
    # CALCULATE DISTANCE
    # --------------------------------------------------------

    distance = calculate_gps_distance(

        location_a["latitude"],
        location_a["longitude"],

        location_b["latitude"],
        location_b["longitude"]
    )

    # --------------------------------------------------------
    # MATCH DECISION
    # --------------------------------------------------------

    return (
        distance
        <=
        SPATIAL_MATCH_DISTANCE_METERS
    )


# ============================================================
# STAGE 4C
# CREATE PHYSICAL DAMAGE ENTITIES
# ============================================================

def create_physical_damage_entities(
    candidates
):

    physical_damage_entities = []

    # --------------------------------------------------------
    # PROCESS EVERY CANDIDATE
    # --------------------------------------------------------

    for candidate in candidates:

        matched_entity = None

        # ----------------------------------------------------
        # SEARCH EXISTING ENTITIES
        # ----------------------------------------------------

        for entity in physical_damage_entities:

            for existing_candidate in (
                entity["source_candidates"]
            ):

                if candidates_match(
                    candidate,
                    existing_candidate
                ):

                    matched_entity = entity

                    break

            if matched_entity is not None:
                break

        # ----------------------------------------------------
        # MATCH FOUND
        # ----------------------------------------------------

        if matched_entity is not None:

            matched_entity[
                "source_candidates"
            ].append(candidate)

            matched_entity[
                "candidate_count"
            ] += 1

            matched_entity[
                "bus_ids"
            ].append(
                candidate["bus_id"]
            )

        # ----------------------------------------------------
        # NO MATCH
        # ----------------------------------------------------

        else:

            damage_id = (

                f"ROAD_DAMAGE_"
                f"{len(physical_damage_entities) + 1:03d}"

            )

            new_entity = {

                "damage_id":
                    damage_id,

                "damage_type":
                    candidate["damage_type"],

                "damage_name":
                    candidate["damage_name"],

                "location":
                    candidate["location"],

                "candidate_count":
                    1,

                "bus_ids": [

                    candidate["bus_id"]

                ],

                "source_candidates": [

                    candidate

                ]
            }

            physical_damage_entities.append(
                new_entity
            )

    # --------------------------------------------------------
    # REMOVE DUPLICATE BUS IDs
    # --------------------------------------------------------

    for entity in physical_damage_entities:

        entity["bus_ids"] = sorted(
            list(
                set(
                    entity["bus_ids"]
                )
            )
        )

    return physical_damage_entities


# ============================================================
# CREATE VIDEO SUMMARY
# ============================================================

def create_video_summary(
    candidates,
    physical_damage_entities
):

    damage_by_type = {}

    # --------------------------------------------------------
    # COUNT CANDIDATES
    # --------------------------------------------------------

    for candidate in candidates:

        damage_name = (
            candidate["damage_name"]
        )

        if damage_name not in damage_by_type:

            damage_by_type[
                damage_name
            ] = 0

        damage_by_type[
            damage_name
        ] += 1

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "total_damage_candidates":
            len(candidates),

        "total_physical_damage_entities":
            len(physical_damage_entities),

        "damage_by_type":
            damage_by_type
    }

    return summary


# ============================================================
# STAGE 4D
# LOAD HISTORICAL DAMAGE DATA
# ============================================================

def load_damage_history():

    if not HISTORY_FILE.exists():

        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)

        if not isinstance(history, list):

            return []

        return history

    except (
        json.JSONDecodeError,
        OSError
    ):

        print(
            "\nWarning: Could not read "
            "damage history. Starting fresh."
        )

        return []


# ============================================================
# SAVE HISTORICAL DAMAGE DATA
# ============================================================

def save_damage_history(
    history
):

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


# ============================================================
# FIND MATCHING HISTORICAL DAMAGE
# ============================================================

def find_historical_match(
    current_entity,
    history
):

    for historical_damage in history:

        # ----------------------------------------------------
        # DAMAGE TYPE
        # ----------------------------------------------------

        if (
            historical_damage["damage_type"]
            !=
            current_entity["damage_type"]
        ):

            continue

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        current_location = (
            current_entity.get("location")
        )

        historical_location = (
            historical_damage.get("location")
        )

        if (
            current_location is None
            or
            historical_location is None
        ):

            continue

        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        distance = calculate_gps_distance(

            current_location["latitude"],
            current_location["longitude"],

            historical_location["latitude"],
            historical_location["longitude"]
        )

        if (
            distance
            <=
            SPATIAL_MATCH_DISTANCE_METERS
        ):

            return historical_damage

    return None


# ============================================================
# STAGE 4D
# UPDATE MULTI-BUS / MULTI-DAY CONFIRMATION
# ============================================================

def update_confirmation(
    physical_damage_entities,
    history
):

    current_date = (
        datetime.now()
        .strftime("%Y-%m-%d")
    )

    confirmed_entities = []

    for entity in physical_damage_entities:

        historical_match = (
            find_historical_match(
                entity,
                history
            )
        )

        # ----------------------------------------------------
        # NEW PHYSICAL DAMAGE
        # ----------------------------------------------------

        if historical_match is None:

            confirmation = {

                "first_detected_date":
                    current_date,

                "last_detected_date":
                    current_date,

                "unique_bus_count":
                    len(
                        set(
                            entity["bus_ids"]
                        )
                    ),

                "unique_day_count":
                    1,

                "total_confirmations":
                    entity[
                        "candidate_count"
                    ]
            }

            entity["confirmation"] = (
                confirmation
            )

            confirmed_entities.append(
                entity
            )

            continue

        # ----------------------------------------------------
        # EXISTING PHYSICAL DAMAGE
        # ----------------------------------------------------

        previous_bus_ids = set(
            historical_match.get(
                "bus_ids",
                []
            )
        )

        current_bus_ids = set(
            entity.get(
                "bus_ids",
                []
            )
        )

        combined_bus_ids = (
            previous_bus_ids
            |
            current_bus_ids
        )

        previous_days = set(
            historical_match.get(
                "detection_dates",
                []
            )
        )

        previous_days.add(
            current_date
        )

        total_confirmations = (

            historical_match.get(
                "total_confirmations",
                0
            )

            +

            entity[
                "candidate_count"
            ]
        )

        confirmation = {

            "first_detected_date":
                historical_match.get(
                    "first_detected_date",
                    current_date
                ),

            "last_detected_date":
                current_date,

            "unique_bus_count":
                len(combined_bus_ids),

            "unique_day_count":
                len(previous_days),

            "total_confirmations":
                total_confirmations
        }

        entity["confirmation"] = (
            confirmation
        )

        confirmed_entities.append(
            entity
        )

    return confirmed_entities


# ============================================================
# STAGE 4D
# UPDATE HISTORY FILE
# ============================================================

def update_damage_history(
    physical_damage_entities,
    history
):

    current_date = (
        datetime.now()
        .strftime("%Y-%m-%d")
    )

    for entity in physical_damage_entities:

        historical_match = (
            find_historical_match(
                entity,
                history
            )
        )

        # ----------------------------------------------------
        # NEW DAMAGE
        # ----------------------------------------------------

        if historical_match is None:

            history.append({

                "damage_id":
                    entity["damage_id"],

                "damage_type":
                    entity["damage_type"],

                "damage_name":
                    entity["damage_name"],

                "location":
                    entity["location"],

                "bus_ids":
                    entity["bus_ids"],

                "detection_dates": [
                    current_date
                ],

                "first_detected_date":
                    current_date,

                "last_detected_date":
                    current_date,

                "total_confirmations":
                    entity[
                        "candidate_count"
                    ],
                
                "status": "CONFIRMED"    
                    
            })

        # ----------------------------------------------------
        # EXISTING DAMAGE
        # ----------------------------------------------------

        else:

            # Keep the original physical damage ID.
            entity["damage_id"] = (
                historical_match[
                    "damage_id"
                ]
            )

            # Update bus list.
            historical_bus_ids = set(
                historical_match.get(
                    "bus_ids",
                    []
                )
            )

            current_bus_ids = set(
                entity.get(
                    "bus_ids",
                    []
                )
            )

            historical_match[
                "bus_ids"
            ] = sorted(
                list(
                    historical_bus_ids
                    |
                    current_bus_ids
                )
            )

            # Update dates.
            detection_dates = set(
                historical_match.get(
                    "detection_dates",
                    []
                )
            )

            detection_dates.add(
                current_date
            )

            historical_match[
                "detection_dates"
            ] = sorted(
                list(detection_dates)
            )

            historical_match[
                "last_detected_date"
            ] = current_date

            historical_match[
                "total_confirmations"
            ] = (

                historical_match.get(
                    "total_confirmations",
                    0
                )

                +

                entity[
                    "candidate_count"
                ]
            )

    save_damage_history(history)

    return history


# ============================================================
# PRINT STAGE 4 SUMMARY
# ============================================================

def print_stage_4_summary(
    candidates,
    physical_damage_entities
):

    print(
        "\n=========================================="
    )

    print(
        "STAGE 4 ROAD DAMAGE PIPELINE"
    )

    print(
        "=========================================="
    )

    print(
        f"\n4A - Damage candidates: "
        f"{len(candidates)}"
    )

    print(
        f"4C - Physical damage entities: "
        f"{len(physical_damage_entities)}"
    )

    print(
        "\n4D - Confirmation details:"
    )

    for entity in physical_damage_entities:

        confirmation = entity.get(
            "confirmation",
            {}
        )

        print(
            f"\n  {entity['damage_id']}"
        )

        print(
            f"    Type: "
            f"{entity['damage_name']}"
        )

        print(
            f"    Buses: "
            f"{confirmation.get('unique_bus_count', 1)}"
        )

        print(
            f"    Days: "
            f"{confirmation.get('unique_day_count', 1)}"
        )

        print(
            f"    Confirmations: "
            f"{confirmation.get('total_confirmations', 1)}"
        )

def process_road_damage_frame(
    model,
    frame,
    frame_number,
    timestamp_seconds,
    gps_location,
    track_history
):
    """
    Process ONE video frame for road damage detection.

    This function does not open or read the video.
    It receives an already-read frame from the main pipeline.

    Returns:
        annotated_frame
        frame_detections
    """

    # ============================================================
    # RUN ROAD DAMAGE MODEL
    # ============================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    frame_detections = []
        
    # ------------------------------------------------------------
    # 5-SECOND EVIDENCE WINDOW
    # ------------------------------------------------------------

    window_start = (
        int(timestamp_seconds // EVIDENCE_WINDOW_SECONDS)
        * EVIDENCE_WINDOW_SECONDS
    )

    window_data = track_history.setdefault(
        "_evidence_window",
        {
            "start_time": window_start,
            "tracks": set(),
            "best_confidence": 0.0,
            "best_frame": None,
            "best_detection": None
        }
    )
    
    
        
    # ------------------------------------------------------------
    # CHECK WHETHER PREVIOUS 5-SECOND WINDOW IS COMPLETE
    # ------------------------------------------------------------

    new_window = (
        window_data["start_time"] != window_start
    )

    if new_window:

        distinct_objects = len(
            window_data["tracks"]
        )

        # Save only if at least 3 different
        # road-damage objects were detected.
        if (
            distinct_objects >= MIN_DISTINCT_DAMAGE_OBJECTS
            and window_data["best_frame"] is not None
        ):

            best_detection = window_data["best_detection"]

            evidence = save_road_damage_evidence(
                frame=window_data["best_frame"],
                damage_name=best_detection["damage_name"],
                track_id=best_detection["track_id"],
                frame_number=best_detection["frame_number"],
                confidence=best_detection["confidence"],
                timestamp_seconds=best_detection["timestamp_seconds"],
                gps_location=best_detection["location"]
            )

            print("\nROAD DAMAGE EVIDENCE SAVED")
            print(
                f"Window: "
                f"{window_data['start_time']} - "
                f"{window_start} seconds"
            )
            print(
                f"Distinct objects: {distinct_objects}"
            )
            print(
                f"Best confidence: "
                f"{best_detection['confidence']:.2f}"
            )
            print(
                f"Image: {evidence['image']}"
            )

        # --------------------------------------------------------
        # START NEW 5-SECOND WINDOW
        # --------------------------------------------------------

        window_data["start_time"] = window_start
        window_data["tracks"] = set()
        window_data["best_confidence"] = 0.0
        window_data["best_frame"] = None
        window_data["best_detection"] = None    
    
    


    

    # ============================================================
    # PROCESS MODEL RESULTS
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
                # CLASS NAME
                # ------------------------------------------------

                if hasattr(result, "names"):
                    class_name = result.names.get(
                        class_id,
                        str(class_id)
                    )
                else:
                    class_name = str(class_id)

                # ------------------------------------------------
                # FRIENDLY DAMAGE NAME
                # ------------------------------------------------

                damage_name = CLASS_NAMES.get(
                    class_name,
                    class_name
                )

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

                # If tracker does not provide an ID,
                # create a temporary frame-level ID.
                if track_id is None:
                    track_id = i

                # ------------------------------------------------
                # CREATE OBSERVATION
                # ------------------------------------------------

                detection = {
                    "event_type": "ROAD_DAMAGE",
                    "damage_type": class_name,
                    "damage_name": damage_name,
                    "track_id": track_id,
                    "confidence": round(confidence, 4),
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "frame_number": frame_number,
                    "timestamp_seconds": round(
                        timestamp_seconds,
                        2
                    ),
                    "bus_id": BUS_ID,
                    "location": gps_location
                }

                frame_detections.append(detection)
                
                
                # Add this damage object to the current
                # 5-second evidence window.
                window_data["tracks"].add(
                    track_id
                )

                # Keep the highest-confidence detection
                # seen in this 5-second window.
                if confidence > window_data["best_confidence"]:

                    window_data["best_confidence"] = confidence
                    window_data["best_detection"] = detection.copy()                
                

                # ------------------------------------------------
                # UPDATE TRACK HISTORY
                # ------------------------------------------------



                update_track_history(
                    track_history,
                    detection
                )


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{damage_name} | "
                    f"ID {track_id} | "
                    f"{confidence * 100:.0f}%"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )
                
                # Save the current annotated frame as the
                # candidate best evidence image.
                if (
                    confidence
                    == window_data["best_confidence"]
                ):
                    window_data["best_frame"] = frame.copy()                




    # ============================================================
    # RETURN RESULTS
    # ============================================================

    return frame, frame_detections


def finalize_road_damage(
    track_history,
    output_events_path
):
    """
    Run Stage 4 road-damage analysis after all video frames
    have been processed.

    This function does NOT read video frames.
    It only analyzes the accumulated track history.
    """

    print("\n" + "=" * 60)
    print("ROAD DAMAGE STAGE 4 FINALIZATION")
    print("=" * 60)

    # ============================================================
    # STAGE 4A — CREATE DAMAGE CANDIDATES
    # ============================================================

    print("\n[STAGE 4A] Creating damage candidates...")

    damage_candidates = create_damage_candidates(
        track_history
    )

    print(
        f"Damage candidates created: "
        f"{len(damage_candidates)}"
    )

    # ============================================================
    # STAGE 4B — ADD REPRESENTATIVE GPS LOCATIONS
    # ============================================================

    print("\n[STAGE 4B] Adding representative locations...")

    add_representative_locations(
        damage_candidates
    )

    # ============================================================
    # STAGE 4C — CREATE PHYSICAL DAMAGE ENTITIES
    # ============================================================

    print(
        "\n[STAGE 4C] Creating physical damage entities..."
    )

    physical_damage_entities = (
        create_physical_damage_entities(
            damage_candidates
        )
    )

    print(
        f"Physical damage entities created: "
        f"{len(physical_damage_entities)}"
    )

    # ============================================================
    # LOAD HISTORICAL DAMAGE
    # ============================================================

    print("\nLoading historical damage...")

    damage_history = load_damage_history()

    print(
        f"Historical damage records loaded: "
        f"{len(damage_history)}"
    )

    # ============================================================
    # STAGE 4D — CONFIRMATION
    # ============================================================

    print("\n[STAGE 4D] Updating confirmations...")

    update_confirmation(
        physical_damage_entities,
        damage_history
    )

    # ============================================================
    # UPDATE HISTORICAL DATABASE
    # ============================================================

    print("\nUpdating damage history...")

    update_damage_history(
        physical_damage_entities,
        damage_history
    )

    # ============================================================
    # SAVE UPDATED HISTORY
    # ============================================================

    save_damage_history(
        damage_history
    )

    # ============================================================
    # VIDEO SUMMARY
    # ============================================================

    video_summary = create_video_summary(
        damage_candidates,
        physical_damage_entities
    )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    final_output = {
        "analysis_type": "ROAD_DAMAGE_EVIDENCE",
        "pipeline_stage": "STAGE_4_COMPLETE",
        "analysis_timestamp": datetime.now().isoformat(),
        "bus_id": BUS_ID,
        "video_summary": video_summary,
        "damage_candidates": damage_candidates,
        "physical_damage_entities": physical_damage_entities,
        "historical_damage_count": len(
            damage_history
        )
    }

    # ============================================================
    # SAVE JSON
    # ============================================================

    output_events_path = Path(
        output_events_path
    )

    output_events_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_events_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_output,
            f,
            indent=4
        )

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print_stage_4_summary(
        video_summary,
        physical_damage_entities
    )

    print(
        f"\nRoad damage results saved to:"
        f"\n{output_events_path}"
    )

    print("=" * 60)
    print("ROAD DAMAGE FINALIZATION COMPLETE")
    print("=" * 60)

    return final_output




# ============================================================
# MAIN ROAD DAMAGE DETECTION
# ============================================================

def detect_road_damage(
    model,
    video_path,
    output_video_path,
    output_events_path
):

    print(
        "\nOpening video..."
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise ValueError(
            f"Could not open video: "
            f"{video_path}"
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

    print(
        f"Video FPS: {fps}"
    )

    print(
        f"Total frames: {total_frames}"
    )

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

    # --------------------------------------------------------
    # GPS PROVIDER
    # --------------------------------------------------------

    gps_provider = GPSProvider(

        start_latitude=
            GPS_START_LATITUDE,

        start_longitude=
            GPS_START_LONGITUDE
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

        gps_location = (
            gps_provider.get_location(
                frame_number
            )
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

                friendly_name = (
                    CLASS_NAMES.get(
                        class_name,
                        class_name
                    )
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
                            gps_location[
                                "latitude"
                            ],

                        "longitude":
                            gps_location[
                                "longitude"
                            ]
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
                frame_number
                /
                total_frames
            ) * 100

            print(

                f"Processing: "
                f"{progress:.1f}% | "

                f"Frame: "
                f"{frame_number}/"
                f"{total_frames} | "

                f"Tracks: "
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
    # STAGE 4B
    # REPRESENTATIVE GPS LOCATIONS
    # ========================================================

    damage_candidates = (
        add_representative_locations(
            damage_candidates
        )
    )

    # ========================================================
    # STAGE 4C
    # PHYSICAL DAMAGE DEDUPLICATION
    # ========================================================

    physical_damage_entities = (
        create_physical_damage_entities(
            damage_candidates
        )
    )

    # ========================================================
    # STAGE 4D
    # LOAD HISTORICAL DATA
    # ========================================================

    damage_history = (
        load_damage_history()
    )

    # ========================================================
    # STAGE 4D
    # MULTI-BUS / MULTI-DAY CONFIRMATION
    # ========================================================

    physical_damage_entities = (
        update_confirmation(
            physical_damage_entities,
            damage_history
        )
    )

    # ========================================================
    # UPDATE HISTORICAL DATABASE
    # ========================================================

    damage_history = (
        update_damage_history(
            physical_damage_entities,
            damage_history
        )
    )

    # ========================================================
    # CREATE SUMMARY
    # ========================================================

    video_summary = (
        create_video_summary(
            damage_candidates,
            physical_damage_entities
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    final_output = {

        "analysis_type":
            "ROAD_DAMAGE_EVIDENCE",

        "pipeline_stage":
            "STAGE_4_COMPLETE",

        "analysis_timestamp":
            datetime.now().isoformat(),

        "bus_id":
            BUS_ID,

        "video_summary":
            video_summary,

        # ----------------------------------------------
        # 4A
        # ----------------------------------------------

        "damage_candidates":
            damage_candidates,

        # ----------------------------------------------
        # 4C
        # ----------------------------------------------

        "physical_damage_entities":
            physical_damage_entities,

        # ----------------------------------------------
        # 4D
        # ----------------------------------------------

        "historical_damage_count":
            len(damage_history)
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
        "ROAD DAMAGE STAGE 4 COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        f"\n4A Damage candidates: "
        f"{len(damage_candidates)}"
    )

    print(
        f"4C Physical damages: "
        f"{len(physical_damage_entities)}"
    )

    print(
        f"4D Historical damages: "
        f"{len(damage_history)}"
    )

    print_stage_4_summary(

        damage_candidates,

        physical_damage_entities
    )

    print(
        f"\nAnnotated video saved to:"
        f"\n{output_video_path}"
    )

    print(
        f"\nEvidence JSON saved to:"
        f"\n{output_events_path}"
    )

    print(
        f"\nHistorical database saved to:"
        f"\n{HISTORY_FILE}"
    )

    return final_output