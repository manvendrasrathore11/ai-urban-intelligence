import logging
import shutil

import torch
from ultralytics import YOLO

from anpr import config


logger = logging.getLogger(__name__)


def _resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_plate_checkpoint():
    """
    Downloads the plate-detector checkpoint from Hugging Face the first
    time it's needed, if it isn't already sitting at PLATE_MODEL_PATH.
    """

    if config.PLATE_MODEL_PATH.exists():
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Plate detector checkpoint not found at {config.PLATE_MODEL_PATH}, "
            "and huggingface_hub isn't installed to auto-download it. "
            "Run: pip install huggingface_hub  — or manually place a "
            "YOLO-format license-plate .pt file at that path."
        )

    print(
        f"[ANPR] Plate checkpoint missing — downloading "
        f"'{config.PLATE_MODEL_HF_FILENAME}' from "
        f"'{config.PLATE_MODEL_HF_REPO}' on Hugging Face..."
    )

    try:
        downloaded_path = hf_hub_download(
            repo_id=config.PLATE_MODEL_HF_REPO,
            filename=config.PLATE_MODEL_HF_FILENAME,
        )
        config.PLATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded_path, config.PLATE_MODEL_PATH)
        print(f"[ANPR] Downloaded plate checkpoint to {config.PLATE_MODEL_PATH}")

    except Exception as e:
        raise FileNotFoundError(
            f"Auto-download failed ({e}). Manually download a YOLO-format "
            f"license-plate checkpoint and place it at "
            f"{config.PLATE_MODEL_PATH}."
        )


def load_anpr_model():
    """
    Loads everything the ANPR pipeline needs and returns it as one bundle,
    the same way load_road_damage_model() / load_accident_model() return
    a single loaded-model object for their pipelines.
    """

    device = _resolve_device()

    print(f"[ANPR] Loading vehicle detector on device={device} ...")
    vehicle_model = YOLO(config.VEHICLE_MODEL_PATH)
    vehicle_model.to(device)

    _ensure_plate_checkpoint()

    print(f"[ANPR] Loading plate detector on device={device} ...")
    plate_model = YOLO(str(config.PLATE_MODEL_PATH))
    plate_model.to(device)

    print("[ANPR] Loading OCR reader (EasyOCR) ...")
    import easyocr
    ocr_reader = easyocr.Reader(
        config.OCR_LANGUAGES,
        gpu=(device == "cuda"),
    )

    return {
        "vehicle_model": vehicle_model,
        "plate_model": plate_model,
        "ocr_reader": ocr_reader,
        "device": device,
    }
