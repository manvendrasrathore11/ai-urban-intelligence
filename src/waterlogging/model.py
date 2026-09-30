"""
Waterlogging "model" loader.

IMPORTANT — read this before wiring the module in:

There is no widely-available, pretrained GitHub checkpoint for
"waterlogging / standing water on a road" the way there is for
road damage (RDD2022 checkpoints) or vehicles (COCO-trained
YOLO). So this module ships an OpenCV colour/texture HEURISTIC
detector instead of a neural network. It needs no weights file
and no download.

This keeps the same load_x_model() / process_x_frame() /
finalize_x() shape as road_damage and accident, so it can be
dropped into pipeline/main.py exactly like the other modules -
but be upfront with your team/judges that this is a classical
computer-vision heuristic, not a trained detector.

Upgrade path (optional, later):
    If you collect and label your own frames (road vs.
    standing-water mask), you can train a YOLOv8-seg or a
    binary segmentation model (e.g. a small U-Net) and swap
    the body of this function for:

        from ultralytics import YOLO
        model = YOLO("path/to/your_waterlogging_seg.pt")

    detector.py would then need its inner logic replaced with
    real inference, but process_waterlogging_frame's INPUT and
    OUTPUT contract can stay the same, so pipeline/main.py would
    not need to change again.
"""


def load_waterlogging_model():
    print("Loading waterlogging detector (heuristic, no weights)...")

    # No neural network to load. We return a small config object
    # so the pipeline's "load model" step still has something to
    # hold onto and print progress for, matching the other modules.
    model = {"type": "heuristic-cv"}

    print("Waterlogging detector ready.")

    return model
