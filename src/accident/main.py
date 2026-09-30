import argparse

from .model import load_accident_model
from .detector import detect_accidents


def main():

    parser = argparse.ArgumentParser(
        description="AI Urban Intelligence - Accident Detection"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--output",
        default="../outputs/videos/accident_detection.mp4",
        help="Output video path"
    )

    args = parser.parse_args()

    model = load_accident_model()

    detect_accidents(
        model,
        args.video,
        args.output
    )


if __name__ == "__main__":
    main()