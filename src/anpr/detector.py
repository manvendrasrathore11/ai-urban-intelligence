import json
import re
from collections import Counter
from datetime import datetime

import cv2

from anpr import config


# ============================================================
# PLATE VALIDATION
# ============================================================

_PLATE_PATTERN = re.compile(
    config.INDIAN_PLATE_REGEX
)


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_plate_text(raw_text):

    if not raw_text:
        return ""

    text = raw_text.upper()

    text = "".join(
        ch
        for ch in text
        if ch.isalnum()
    )

    return text


# ============================================================
# VALIDATE INDIAN PLATE
# ============================================================

def _is_valid_plate(text):

    if not text:
        return False

    return bool(
        _PLATE_PATTERN.match(text)
    )


# ============================================================
# OCR
# ============================================================

def _read_plate(
    ocr_reader,
    plate_crop
):

    if plate_crop is None:
        return None, 0.0

    if plate_crop.size == 0:
        return None, 0.0


    # --------------------------------------------------------
    # Upscale small plate
    # --------------------------------------------------------

    h = plate_crop.shape[0]

    if (
        h > 0
        and h < config.OCR_MIN_CROP_HEIGHT_PX
    ):

        scale = (
            config.OCR_MIN_CROP_HEIGHT_PX
            / h
        )

        plate_crop = cv2.resize(
            plate_crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    results = ocr_reader.readtext(

        plate_crop,

        allowlist=config.OCR_ALLOWLIST,

        detail=1
    )


    if not results:
        return None, 0.0


    candidates = []


    for result in results:

        text = result[1]

        confidence = float(
            result[2]
        )

        text = _clean_plate_text(
            text
        )

        if not text:
            continue

        candidates.append(
            (
                text,
                confidence
            )
        )


    if not candidates:
        return None, 0.0


    # --------------------------------------------------------
    # Prefer a valid Indian plate
    # --------------------------------------------------------

    valid_candidates = [
        item
        for item in candidates
        if _is_valid_plate(item[0])
    ]


    if valid_candidates:

        return max(
            valid_candidates,
            key=lambda x: x[1]
        )


    # No valid plate
    return None, 0.0


# ============================================================
# SAVE PLATE CROP
# ============================================================

def _save_plate_crop(
    plate_crop,
    track_id,
    frame_number,
    plate_text
):

    if not config.SAVE_CROPS:
        return None

    if plate_crop is None:
        return None

    if plate_crop.size == 0:
        return None


    filename = (
        f"{plate_text}_"
        f"track_{track_id}_"
        f"frame_{frame_number}.jpg"
    )


    path = (
        config.PLATE_CROP_DIR
        / filename
    )


    cv2.imwrite(
        str(path),
        plate_crop
    )


    return str(path)


# ============================================================
# PROCESS ANPR FRAME
# ============================================================

def process_anpr_frame(
    model,
    frame,
    frame_number,
    timestamp_seconds,
    plate_history
):

    vehicle_model = (
        model["vehicle_model"]
    )

    plate_model = (
        model["plate_model"]
    )

    ocr_reader = (
        model["ocr_reader"]
    )

    device = model["device"]


    frame_reads = []


    # ========================================================
    # VEHICLE TRACKING
    # ========================================================

    vehicle_results = (
        vehicle_model.track(

            source=frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=config.VEHICLE_CONF_THRESHOLD,

            classes=list(
                config.VEHICLE_CLASS_IDS.keys()
            ),

            imgsz=config.VEHICLE_IMGSZ,

            device=device,

            verbose=False
        )[0]
    )


    if (
        vehicle_results.boxes is None
        or len(vehicle_results.boxes) == 0
    ):

        return (
            frame,
            frame_reads,
            len(plate_history)
        )


    boxes = vehicle_results.boxes


    # ========================================================
    # PROCESS EACH VEHICLE
    # ========================================================

    for index, vbox in enumerate(boxes):

        vx1, vy1, vx2, vy2 = [
            int(v)
            for v
            in vbox.xyxy[0].tolist()
        ]


        vclass_idx = int(
            vbox.cls.item()
        )


        vclass = (
            config.VEHICLE_CLASS_IDS.get(
                vclass_idx,
                str(vclass_idx)
            )
        )


        # ----------------------------------------------------
        # TRACK ID
        # ----------------------------------------------------

        if boxes.id is not None:

            track_id = int(
                boxes.id[index]
                .cpu()
                .item()
            )

        else:

            track_id = index


        # ----------------------------------------------------
        # Clamp vehicle box
        # ----------------------------------------------------

        vx1 = max(
            0,
            vx1
        )

        vy1 = max(
            0,
            vy1
        )

        vx2 = min(
            frame.shape[1],
            vx2
        )

        vy2 = min(
            frame.shape[0],
            vy2
        )


        if vx2 <= vx1 or vy2 <= vy1:
            continue


        # ----------------------------------------------------
        # Vehicle crop
        # ----------------------------------------------------

        vehicle_crop = frame[
            vy1:vy2,
            vx1:vx2
        ]


        if vehicle_crop.size == 0:
            continue


        # ====================================================
        # PLATE DETECTION
        # ====================================================

        plate_results = (
            plate_model.predict(

                source=vehicle_crop,

                conf=config.PLATE_CONF_THRESHOLD,

                imgsz=config.PLATE_IMGSZ,

                device=device,

                verbose=False
            )[0]
        )


        if (
            plate_results.boxes is None
            or len(plate_results.boxes) == 0
        ):
            continue


        # ====================================================
        # PROCESS PLATES
        # ====================================================

        for pbox in plate_results.boxes:

            px1, py1, px2, py2 = [
                int(v)
                for v
                in pbox.xyxy[0].tolist()
            ]


            plate_confidence = float(
                pbox.conf.item()
            )


            # ------------------------------------------------
            # Vehicle crop coordinates
            # → frame coordinates
            # ------------------------------------------------

            abs_x1 = vx1 + px1
            abs_y1 = vy1 + py1

            abs_x2 = vx1 + px2
            abs_y2 = vy1 + py2


            # ------------------------------------------------
            # Clamp
            # ------------------------------------------------

            abs_x1 = max(
                0,
                abs_x1
            )

            abs_y1 = max(
                0,
                abs_y1
            )

            abs_x2 = min(
                frame.shape[1],
                abs_x2
            )

            abs_y2 = min(
                frame.shape[0],
                abs_y2
            )


            if (
                abs_x2 <= abs_x1
                or abs_y2 <= abs_y1
            ):
                continue


            # ------------------------------------------------
            # Plate crop
            # ------------------------------------------------

            plate_crop = frame[
                abs_y1:abs_y2,
                abs_x1:abs_x2
            ].copy()


            if plate_crop.size == 0:
                continue


            # =================================================
            # OCR
            # =================================================

            # OCR is expensive, so don't run it on every frame.
            # Run OCR periodically for each tracked vehicle.

            should_run_ocr = (
                frame_number % config.OCR_FRAME_INTERVAL == 0
            )

            if should_run_ocr:

                plate_text, ocr_confidence = (
                    _read_plate(
                        ocr_reader,
                        plate_crop
                    )
                )

            else:

                plate_text = None
                ocr_confidence = 0.0


            valid = (
                plate_text is not None
                and _is_valid_plate(
                    plate_text
                )
            )


            # =================================================
            # DRAW VEHICLE
            # =================================================

            cv2.rectangle(

                frame,

                (vx1, vy1),

                (vx2, vy2),

                (255, 0, 0),

                4
            )


            vehicle_label = (
                f"{vclass.upper()} "
                f"ID:{track_id}"
            )


            cv2.putText(

                frame,

                vehicle_label,

                (vx1, max(vy1 - 15, 30)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.80,

                (255, 0, 0),

                3
            )


            # =================================================
            # DRAW PLATE
            # =================================================

            if valid:

                box_color = (
                    0,
                    255,
                    0
                )

                label = (
                    f"{plate_text} "
                    f"ID:{track_id}"
                )

            else:

                box_color = (
                    128,
                    128,
                    128
                )

                label = (
                    f"PLATE ID:{track_id}"
                )


            cv2.rectangle(

                frame,

                (abs_x1, abs_y1),

                (abs_x2, abs_y2),

                box_color,

                4
            )


            cv2.putText(

                frame,

                label,

                (
                    abs_x1,
                    max(
                        abs_y1 - 15,
                        30
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.80,

                box_color,

                3
            )


            # =================================================
            # CREATE READ RECORD
            # =================================================

            read_record = {

                "frame_number":
                    frame_number,

                "timestamp_seconds":
                    timestamp_seconds,

                "vehicle_track_id":
                    track_id,

                "vehicle_class":
                    vclass,

                "vehicle_bbox": [
                    vx1,
                    vy1,
                    vx2,
                    vy2
                ],

                "plate_bbox": [
                    abs_x1,
                    abs_y1,
                    abs_x2,
                    abs_y2
                ],

                "plate_text":
                    plate_text,

                "ocr_confidence":
                    round(
                        ocr_confidence,
                        3
                    ),

                "plate_detector_confidence":
                    round(
                        plate_confidence,
                        3
                    ),

                "is_valid_format":
                    valid
            }


            frame_reads.append(
                read_record
            )


            # =================================================
            # IGNORE INVALID OCR
            # =================================================

            if not valid:
                continue


            # =================================================
            # SAVE CROP
            # =================================================

            crop_path = (
                _save_plate_crop(

                    plate_crop,

                    track_id,

                    frame_number,

                    plate_text
                )
            )


            read_record[
                "crop_path"
            ] = crop_path


            # =================================================
            # CREATE TRACK HISTORY
            # =================================================

            if track_id not in plate_history:

                plate_history[track_id] = {

                    "vehicle_track_id":
                        track_id,

                    "vehicle_class":
                        vclass,

                    "first_frame":
                        frame_number,

                    "first_timestamp_seconds":
                        timestamp_seconds,

                    "last_frame":
                        frame_number,

                    "last_timestamp_seconds":
                        timestamp_seconds,

                    "observations":
                        [],

                    "best_plate":
                        None,

                    "best_confidence":
                        0.0,

                    "best_crop_path":
                        None
                }


            entry = plate_history[
                track_id
            ]


            # ------------------------------------------------
            # Add observation
            # ------------------------------------------------

            entry[
                "observations"
            ].append({

                "plate_text":
                    plate_text,

                "ocr_confidence":
                    ocr_confidence,

                "frame_number":
                    frame_number,

                "timestamp_seconds":
                    timestamp_seconds,

                "crop_path":
                    crop_path
            })


            entry[
                "last_frame"
            ] = frame_number


            entry[
                "last_timestamp_seconds"
            ] = timestamp_seconds


            # =================================================
            # SELECT BEST PLATE
            # =================================================

            observations = (
                entry["observations"]
            )


            plate_counts = Counter(
                obs["plate_text"]
                for obs in observations
            )


            best_plate = max(
                plate_counts,
                key=plate_counts.get
            )


            best_observations = [
                obs
                for obs in observations
                if obs["plate_text"]
                == best_plate
            ]


            best_observation = max(
                best_observations,
                key=lambda x:
                    x["ocr_confidence"]
            )


            entry[
                "best_plate"
            ] = best_plate


            entry[
                "best_confidence"
            ] = best_observation[
                "ocr_confidence"
            ]


            entry[
                "best_crop_path"
            ] = best_observation[
                "crop_path"
            ]


    return (
        frame,
        frame_reads,
        len(plate_history)
    )


# ============================================================
# FINALIZE ANPR
# ============================================================

def finalize_anpr(
    plate_history,
    output_path
):

    plates = []


    for track_id, entry in (
        plate_history.items()
    ):

        observations = (
            entry["observations"]
        )


        if not observations:
            continue


        best_plate = (
            entry["best_plate"]
        )


        if not best_plate:
            continue


        plate_counts = Counter(
            obs["plate_text"]
            for obs in observations
        )


        record = {

            "event_type":
                "vehicle_plate_sighting",

            "vehicle_track_id":
                track_id,

            "vehicle_class":
                entry["vehicle_class"],

            "plate_text":
                best_plate,

            "ocr_confidence":
                round(
                    entry["best_confidence"],
                    3
                ),

            "read_count":
                len(observations),

            "plate_vote_count":
                plate_counts[
                    best_plate
                ],

            "first_frame":
                entry["first_frame"],

            "last_frame":
                entry["last_frame"],

            "first_timestamp_seconds":
                entry[
                    "first_timestamp_seconds"
                ],

            "last_timestamp_seconds":
                entry[
                    "last_timestamp_seconds"
                ],

            "bus_id":
                config.BUS_ID,

            "gps": {

                "latitude":
                    config.GPS_LATITUDE,

                "longitude":
                    config.GPS_LONGITUDE
            },

            "evidence_image":
                entry["best_crop_path"],

            "timestamp":
                datetime.now().isoformat()
        }


        plates.append(
            record
        )


    # ========================================================
    # FINAL JSON
    # ========================================================

    result = {

        "pipeline":
            "AI Urban Intelligence ANPR",

        "generated_at":
            datetime.now().isoformat(),

        "total_unique_vehicles":
            len(plates),

        "plates":
            plates
    }


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )


    print(
        "\n============================================================"
    )

    print(
        "ANPR FINALIZATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Unique vehicles with valid plates: "
        f"{len(plates)}"
    )


    for plate in plates:

        print(
            f"  Vehicle ID: "
            f"{plate['vehicle_track_id']} | "
            f"Plate: "
            f"{plate['plate_text']} | "
            f"Reads: "
            f"{plate['read_count']}"
        )


    print(
        f"\nANPR JSON saved to:\n"
        f"{output_path}"
    )


    return result