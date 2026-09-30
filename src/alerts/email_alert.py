import os
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


# ============================================================
# SEND ACCIDENT EMAIL
# ============================================================

def send_accident_email(
    accident_event
):
    """
    Send an accident alert email.

    The email contains:
    - Accident ID
    - Severity
    - GPS
    - Vehicle count
    - Collision count
    - Nearest police
    - Nearest hospital
    - Evidence image
    """

    # ========================================================
    # CHECK CREDENTIALS
    # ========================================================

    if not SENDER_EMAIL:
        print("Email error: SENDER_EMAIL not configured.")
        return False

    if not RECEIVER_EMAIL:
        print("Email error: RECEIVER_EMAIL not configured.")
        return False

    if not EMAIL_PASSWORD:
        print("Email error: EMAIL_PASSWORD not configured.")
        return False

    # ========================================================
    # ACCIDENT INFORMATION
    # ========================================================

    event_id = accident_event.get(
        "event_id",
        "UNKNOWN"
    )

    severity = accident_event.get(
        "severity",
        "UNKNOWN"
    )

    vehicle_count = accident_event.get(
        "vehicle_count",
        0
    )

    collision_count = accident_event.get(
        "collision_count",
        0
    )

    timestamp = accident_event.get(
        "timestamp",
        "UNKNOWN"
    )

    timestamp_seconds = accident_event.get(
        "timestamp_seconds",
        0
    )

    bus_id = accident_event.get(
        "bus_id",
        "UNKNOWN"
    )

    gps = accident_event.get(
        "gps",
        {}
    )

    latitude = gps.get(
        "latitude",
        "UNKNOWN"
    )

    longitude = gps.get(
        "longitude",
        "UNKNOWN"
    )

    # ========================================================
    # EMERGENCY SERVICES
    # ========================================================

    services = accident_event.get(
        "nearest_services",
        {}
    )

    police = services.get(
        "police"
    )

    hospital = services.get(
        "hospital"
    )

    # --------------------------------------------------------
    # Police information
    # --------------------------------------------------------

    if police:

        police_name = police.get(
            "name",
            "Unknown"
        )

        police_distance = police.get(
            "distance_km",
            "Unknown"
        )

    else:

        police_name = "No nearby police station found"

        police_distance = "Unknown"

    # --------------------------------------------------------
    # Hospital information
    # --------------------------------------------------------

    if hospital:

        hospital_name = hospital.get(
            "name",
            "Unknown"
        )

        hospital_distance = hospital.get(
            "distance_km",
            "Unknown"
        )

    else:

        hospital_name = "No nearby hospital found"

        hospital_distance = "Unknown"

    # ========================================================
    # EMAIL SUBJECT
    # ========================================================

    subject = (
        f"🚨 ACCIDENT ALERT | "
        f"{severity} | "
        f"{event_id}"
    )

    # ========================================================
    # EMAIL BODY
    # ========================================================

    body = f"""
AI URBAN INTELLIGENCE
ACCIDENT ALERT
==============================

Accident ID:
{event_id}

Severity:
{severity}

Bus ID:
{bus_id}

Detection Time:
{timestamp}

Video Timestamp:
{timestamp_seconds} seconds


ACCIDENT INFORMATION
------------------------------

Vehicles detected:
{vehicle_count}

Collision count:
{collision_count}


ACCIDENT LOCATION
------------------------------

Latitude:
{latitude}

Longitude:
{longitude}


NEAREST POLICE STATION
------------------------------

Name:
{police_name}

Distance:
{police_distance} km


NEAREST HOSPITAL
------------------------------

Name:
{hospital_name}

Distance:
{hospital_distance} km


This alert was generated automatically
by the AI Urban Intelligence system.
"""

    # ========================================================
    # CREATE EMAIL
    # ========================================================

    message = MIMEMultipart()

    message["From"] = SENDER_EMAIL
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = subject

    message.attach(
        MIMEText(
            body,
            "plain"
        )
    )

    # ========================================================
    # ATTACH EVIDENCE IMAGE
    # ========================================================

    evidence = accident_event.get(
        "evidence",
        {}
    )

    evidence_path = evidence.get(
        "image"
    )

    if evidence_path:

        evidence_file = Path(
            evidence_path
        )

        if evidence_file.exists():

            try:

                with open(
                    evidence_file,
                    "rb"
                ) as image_file:

                    image = MIMEImage(
                        image_file.read()
                    )

                image.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=evidence_file.name
                )

                message.attach(image)

                print(
                    f"Evidence attached: "
                    f"{evidence_file}"
                )

            except Exception as e:

                print(
                    f"Could not attach evidence: {e}"
                )

        else:

            print(
                f"Evidence file not found: "
                f"{evidence_file}"
            )

    # ========================================================
    # SEND EMAIL
    # ========================================================

    try:

        print("\nConnecting to Gmail...")

        with smtplib.SMTP(
            "smtp.gmail.com",
            587
        ) as server:

            server.starttls()

            server.login(
                SENDER_EMAIL,
                EMAIL_PASSWORD
            )

            server.send_message(
                message
            )

        print(
            "\nEMAIL ALERT SENT SUCCESSFULLY"
        )

        print(
            f"From: {SENDER_EMAIL}"
        )

        print(
            f"To: {RECEIVER_EMAIL}"
        )

        return True

    except Exception as e:

        print(
            "\nEMAIL ALERT FAILED"
        )

        print(
            f"Error: {e}"
        )

        return False