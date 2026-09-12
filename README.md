# AI Urban Intelligence

This project is a beginner-friendly starting point for an AI Urban Intelligence system that detects road damage from bus-camera videos.

## Project Structure

- `models/road_damage/` - Place model files here when ready.
- `videos/input/` - Add input video files here.
- `outputs/videos/` - Generated processed videos will be saved here.
- `outputs/events/` - Detection event summaries or JSON output can be saved here.
- `src/road_damage/` - Python source code for the road-damage detection pipeline.

## Current Status

This repository currently contains only the initial project skeleton:
- Basic package structure
- Placeholder Python files with simple comments/docstrings
- Empty folders for models, videos, and outputs
- A `requirements.txt` file for dependencies to be added later

## Next Steps

1. Add Python dependencies in `requirements.txt`.
2. Implement configuration values in `src/road_damage/config.py`.
3. Add model loading logic in `src/road_damage/model.py`.
4. Build detection logic in `src/road_damage/detector.py`.
5. Run the application entry point in `src/road_damage/main.py`.

## Notes

- No model download is performed here.
- No YOLO implementation is included yet.
- This is intended as a clean starting structure for future development.

rrd  pipline 

                    YOUR STAGE 1
                         │
                         ▼
                 ┌───────────────┐
                 │   Road Video  │
                 │ road_test.mp4 │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ OpenCV        │
                 │ VideoCapture  │
                 └───────┬───────┘
                         │
                    frame 1
                    frame 2
                    frame 3
                       ...
                         │
                         ▼
                 ┌───────────────┐
                 │ YOLOv12s      │
                 │ RDD2022 Model │
                 └───────┬───────┘
                         │
                         ▼
             ┌────────────────────────┐
             │ Detection              │
             │                        │
             │ class                  │
             │ confidence             │
             │ bounding box           │
             └──────────┬─────────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      ┌──────────────┐      ┌──────────────┐
      │ Annotated    │      │ JSON Events  │
      │ Video        │      │              │
      └──────────────┘      └──────────────┘


      detect file pipline

Input Video
    ↓
Read frame
    ↓
YOLO model
    ↓
Find road damage
    ↓
Filter low-confidence detections
    ↓
Create event data
    ↓
Draw bounding boxes + labels
    ↓
Save annotated video
    ↓
Save events as JSON      