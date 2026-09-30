from ultralytics import YOLO


def load_accident_model():
    print("Loading accident detection model...")

    model = YOLO("yolo11n.pt")

    print("Accident model loaded successfully.")

    return model