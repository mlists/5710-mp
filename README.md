# AR Chess

## Install requirements
```powershell
pip install -r requirements.txt
```

## Run the interactive AR program
```powershell
python main.py
```

### Useful command-line options

| Option | Default | What it does |
|---|---|---|
| `--source PATH_OR_NUMBER` | `0` | Input video filename or webcam device number. |
| `--width PIXELS` | `1920` | Request webcam capture width. Ignored for prerecorded files. |
| `--height PIXELS` | `1080` | Request webcam capture height. Ignored for prerecorded files. |

## Project breakdown
pinch_benchmarking and pose_benchmarking contain copies of the files used for those sections of the report and do not directly depend on the main code body.
### Benchmarking/debugging files
- debug_pose.py - Draw pose information on a video or camera stream
- test_find_board.py - Annotation/evaluation program for manually marking images to test the board detector, generate corner_annotations.json
- calibration.py - Run calibration of the camera, generate cal_data.pkl
### Main code body
- board_detector.py - Dectector capable of detecting partially occluded chessboards
- board_tracker.py - Optical flow tracker to avoid running detection every frame, attempts to preserve consistent board rotation
- get_pose.py - Calculate the pose of the camera from board observations
- pose_quality.py - Check accuracy of pose solutions
- pinch_detector.py - Detect pinches in pixel space
- render_board.py - Rendering code to place the models on the board
- game_gui.py - Gamestate management and GUI code
- main.py - Run the whole program