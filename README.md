# Chessboard Tracking and 3D Pose, v8

Reference for `find_board_video_v8.py`.

The program uses the **original `find_board.py` detector**, the **original `board_tracker.py` optical flow through `board_tracker_v2.py`** (orientation-safe IDs), and the **unchanged four-corner `get_pose_v6.py` solver**. The v8 addition is a pose acceptance check scaled to the chessboard's apparent square size. No smoothing is applied.

## 1. Setup

In project folder, keep these files together:

```text
5710-mp/
  find_board.py                 # original detector
  board_tracker.py              # original optical-flow tracker
  board_tracker_v2.py           # orientation-safe corner association
  get_pose_v6.py                # raw four-corner PnP pose
  pose_quality_v8.py            # relative/pixel acceptance gate
  find_board_video_v8.py        # run this file
  cal_data.pkl                  # camera calibration
  videos/
    5.mp4                      # example recorded input
  outputs/                      # CSV, PNG and MP4 results
```

Install dependencies in your activated virtual environment, if needed:

```powershell
pip install numpy opencv-python
```

Run commands from the project directory. Paths containing spaces should be quoted.

## 2. Copy-and-paste examples

**Live webcam at requested 1920 x 1080, default relative gate:**

```powershell
python find_board_video_v8.py --webcam 0 --calibration ".\cal_data.pkl" --display-scale 0.5
```

**Prerecorded 1080p video, display plus CSV, screenshot and processed MP4:**

```powershell
python find_board_video_v8.py --source ".\training-videos\5.mp4" --calibration ".\cal_data.pkl" --csv ".\outputs\5_diagnostics.csv" --save-preview ".\outputs\5_preview.png" --output-video ".\outputs\5_annotated.mp4"
```

**Process without opening a window (useful for batches):**

```powershell
python find_board_video_v8.py --source ".\training-videos\5.mp4" --no-display --csv ".\outputs\5_diagnostics.csv" --output-video ".\outputs\5_annotated.mp4"
```

**Compare the fixed 14-pixel rule:**

```powershell
python find_board_video_v8.py --source ".\training-videos\5.mp4" --pose-gate pixels --max-corner-error 14
```

**Only the first 200 frames, with less console output:**

```powershell
python find_board_video_v8.py --source ".\training-videos\5.mp4" --max-frames 200 --print-every 10
```

## 3. All command-line options

| Option | Default | What it does |
|---|---|---|
| `--source PATH_OR_NUMBER` | `0` | Input video filename or webcam device number. Mutually exclusive with `--webcam`. |
| `--webcam NUMBER` | Not specified | Explicit webcam device index, usually `0`. |
| `--calibration PATH` | `cal_data.pkl` | Load matching camera intrinsics `K` and distortion `D` from `.pkl` or `.npz`. |
| `--camera-width PIXELS` | `1920` | Request webcam capture width. Ignored for prerecorded files. |
| `--camera-height PIXELS` | `1080` | Request webcam capture height. Ignored for prerecorded files. |
| `--square-size METRES` | `0.057` | Physical chessboard square pitch, 57 mm. Affects the metric 3D pose. |
| `--detect-every N` | `60` | Run full detection every N frames, also on initialisation or tracking loss. |
| `--display-scale FACTOR` | `0.5` | Resize the two-panel display, saved PNG and output MP4. **Does not resize the source images used by tracking/PnP.** Allowed range: above `0`, up to `2.0`. |
| `--pose-gate relative\|pixels` | `relative` | Choose how the final pose reprojection error is accepted. |
| `--max-corner-error-squares VALUE` | `0.35` | With `relative`, allow up to this fraction of one **local projected square** at *each* board corner. |
| `--max-corner-error PIXELS` | `14.0` | With `pixels`, allow this many pixels at *each* board corner. **Ignored in relative mode.** |
| `--print-every N` | `1` | Print one diagnostic line every N frames (always prints frame 1 and major errors). |
| `--csv PATH` | Off | Save per-frame tracking, orientation, pose, errors and stage timings to CSV. |
| `--max-frames N` | `0` | Stop after N processed frames; `0` means process until video ends or user quits. |
| `--no-display` | Off | Disable the OpenCV viewer. Still permits CSV, PNG and MP4 output. |
| `--save-preview PATH` | Off | Save a two-panel PNG from the **first valid pose**, or the last diagnostic frame if no valid pose occurs. |
| `--output-video PATH` | Off | Write an annotated side-by-side MP4 with tracking on the left and pose on the right. |

View the built-in help at any time:

```powershell
python find_board_video_v8.py --help
```

## 4. Understanding the output

- **Left panel:** current tracking geometry, numbered corners, tracked intersections and projected square centres.
- **Right panel:** tracker outline in amber, PnP-projected outline in magenta and X/Y/Z axes only when the pose is accepted.
- **Corner ID3:** must be on a black A1/H8 square; the 180-degree A1/H8 ambiguity is allowed. Positive Z must face out of the printed board.
- **POSE VALID:** tracking/orientation checks and the selected error gate passed.
- **POSE REJECTED:** a pose was calculated but failed a validity or error check; no accepted pose axes are drawn.
- **POSE UNAVAILABLE:** insufficient current tracking geometry or another prerequisite failed.
- **CSV:** includes pixel errors (`corner0_px` to `corner3_px`), normalized errors (`corner0_squares` to `corner3_squares`), `origin_px`, `gate_mode`, `gate_limit`, `pose_status`, tracked point counts and stage timings in milliseconds.

For **relative** mode, each corner's pixel error is divided by the projected local square pitch at that corner. At a 0.25-square threshold, a 40-pixel square permits approximately 10 pixels of error, while a 100-pixel square permits approximately 25 pixels. This makes the acceptance rule fairer across board distances; it does not change the actual PnP pose.

**Viewer controls:** press `q` to quit and `r` to reset tracking and attempt reinitialisation on the next frame.