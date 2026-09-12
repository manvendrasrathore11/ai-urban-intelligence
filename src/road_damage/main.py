from pathlib import Path

from .config import (
    VIDEO_INPUT_DIR,
    VIDEO_OUTPUT_DIR,
    EVENT_OUTPUT_DIR
)

from .model import load_road_damage_model

from .detector import detect_road_damage


def main():

    # --------------------------------------------------
    # INPUT VIDEO
    # --------------------------------------------------

    input_video = (
        VIDEO_INPUT_DIR /
        "road_test.mp4"
    )

    # --------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------

    output_video = (
        VIDEO_OUTPUT_DIR /
        "road_damage_detected.mp4"
    )

    output_events = (
        EVENT_OUTPUT_DIR /
        "road_damage_events.json"
    )

    # --------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------

    if not input_video.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{input_video}\n\n"
            f"Put your video inside:\n"
            f"videos/input/"
        )

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    model = load_road_damage_model()

    # --------------------------------------------------
    # RUN DETECTION
    # --------------------------------------------------

    events = detect_road_damage(
        model=model,
        video_path=input_video,
        output_video_path=output_video,
        output_events_path=output_events
    )

    print("\n================================")
    print("ROAD DAMAGE PIPELINE COMPLETE")
    print("================================")

    print(
        f"Total events: {len(events)}"
    )


if __name__ == "__main__":
    main()