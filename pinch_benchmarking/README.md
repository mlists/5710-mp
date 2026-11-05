# AMME5710 | Standalone pinch-gesture benchmarking framework

This is a complete **board-free** capture, ground-truth annotation, replay, scoring and plotting workflow. It is built around the original `pinch_detector.py` and the MediaPipe **Tasks Hand Landmarker** already supplied in the starter pack. You do **not** need the physical chessboard, PnP pose estimation, or any chess project imports.

**No experimental footage, ground-truth annotations, or results were supplied.** The software is ready to collect and analyse your own recordings. Automated tests use artificial landmarks **only for checking the software**. They are not evidence of gesture-recognition performance.

## 1. Install once

Recommended: Python 3.11 or 3.12, Windows (or another desktop OS with a supported webcam and MediaPipe installation). Open a terminal **inside this extracted folder**.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. A local `hand_landmarker.task` model is included in the archive, so the capture program does **not** attempt a model download. Run the software with a real camera on your own computer; this pack does not require any network service after installation.

**Quick verification:**

```powershell
python -m unittest discover -s tests -v
```

The tests simulate a camera/MediaPipe sequence and validate video-frame alignment, detector behavior, replay, annotation-policy safeguards, CSV/JSON/LaTeX outputs and figures. Passing synthetic tests is **not** a validation of MediaPipe on your actual camera.

## 2. Easiest workflow: interactive recording of all conditions

First see exactly what will be recorded:

```powershell
python pinch_benchmark_run_protocol.py --session session01 --camera 1 --dry-run
```

Then record the full main experiment set:

```powershell
python pinch_benchmark_run_protocol.py --session session01 --camera 1
```

**Camera view is horizontally mirrored by default, like a selfie camera.** You see and annotate the same mirrored image that the model processes, and the video and landmarks remain spatially aligned. Add `--no-mirror` to any capture or batch-recording command if you want the original, unflipped camera orientation. Do **not** flip saved video again before annotating; older clips stay in their recorded orientation. The `mirror` column in `session01/manifest.csv` records the setting for each clip.

The script prompts you before each clip. Press Enter when the hand and camera are positioned, then perform the displayed action. Each condition is a separate video and has a fresh MediaPipe tracker and pinch detector. The program stops that clip after the planned duration, or sooner if you press `Q` or `Esc`. Camera index **1** follows your starter pack; use `--camera 0` if that is your webcam. Windows users can add `--backend msmf` if DirectShow has trouble. Add `--video-codec MJPG` to save AVI instead of MP4 if the machine cannot open an `mp4v` VideoWriter.

The default plan captures **one pilot**, **three distances**, **three orientations** and **four negative controls**, 11 clips total. Edit `protocol/experiment_plan.csv` if you need longer durations or additional repetitions. Use `--experiments distance orientation negative` to skip the pilot. Optional experiments can be selected explicitly with `--experiments optional`.

### Experiment conditions

| Group | Conditions | Suggested recording |
|---|---|---|
| Distance | 35 cm, 60 cm, 100 cm | 10 full pinch/release cycles at each distance, palm front-facing |
| Orientation | 0°, 45°, 75° sideways | 10 full cycles at each orientation at approximately 60 cm |
| Negative controls | Open palm, pointing, closed fist, waving without pinching | 25 seconds of each control, with **no physical contact gesture** |
| Parameter ablation | Confirmation windows of 1, 3, 5 processed frames | **No additional recording**. Reuse exactly the same stored landmarks |

Before each positive clip, hold the hand OPEN. Make a clear thumb-index contact, hold briefly, release clearly and hold OPEN. Aim for 10 cycles per recording and finish OPEN. Keep the camera, lighting, participant and dominant hand consistent except for the factor explicitly being studied. Distances and angles are approximate unless physically measured.

### Alternative: record just one clip

```powershell
python pinch_benchmark_capture.py --session session01 --clip-id distance_35cm_01 --experiment distance --condition 35cm --distance-cm 35 --expected-cycles 10 --camera 1 --duration 25
```

Negative example:

```powershell
python pinch_benchmark_capture.py --session session01 --clip-id negative_pointing_01 --experiment negative --condition pointing --camera 1 --duration 25
```

Capture defaults are the **documented final system**: pinch threshold **0.30**, release threshold **0.35**, three consecutive frames and hand-loss cancellation at six missing observations. The copied standalone `pinch_webcam_v2.py` and original `pinch_detector.py` have different constructor/standalone defaults (**0.25/0.30**). The original detector logic remains untouched, and the standalone webcam runner has only its **camera mirror CLI default** changed to on. The new capture and replay programs explicitly use **0.30/0.35/3** unless you override them.

### What is saved per clip

```text
session01/
  manifest.csv                           # Experimental condition, camera, actual settings, versions, SHA-256
  recordings/distance_35cm_01.mp4        # RAW video: no detector or predicted-event graphics
  landmarks/distance_35cm_01.jsonl       # Frame-indexed 21 (x,y,z) landmarks, ratios, states, events, timings
  frames/distance_35cm_01.csv           # Easy-to-inspect flat per-frame metrics (same frame IDs)
  ground_truth_events.csv               # Created by human annotation
  ground_truth_visibility.csv           # Visually PRESENT/ABSENT/UNKNOWN frame intervals
  annotation_status.csv                 # DRAFT or COMPLETE; evaluated only when COMPLETE
  results/                               # Generated CSV, JSON, LaTeX and plots
```

The mirror transform, when enabled, is applied **once before MediaPipe inference and video saving**. Consequently, the stored 21 landmarks and event/cursor pixel coordinates correspond to the displayed and saved frames. The annotation GUI shows stored video as-is, with no additional flip. A horizontal reflection does **not** change the pinch-distance ratio or its scalar thresholds.

Every saved video frame has **one matching JSONL row and CSV row**, indexed starting at zero. `time_s` is a monotonic timestamp taken when the camera read returns. `video_time_s = frame_index / video_fps_nominal` is **only an encoded-video playback coordinate**, not a substitute for physical capture time. Because webcam loop speed may vary, the annotation and evaluation use the recorded `time_s`, not playback seconds. Real exposure timing can precede the camera-read timestamp if frames are buffered.

`hand_detected` records whether MediaPipe returned a hand. `valid_landmarks` records whether the palm measurement was valid. If palm size is below one pixel, the event detector receives a missing-hand observation rather than crashing, while the original landmark data is preserved. A `PINCH_CANCEL` after six missing observations is tracked separately from `PINCH_RELEASE`.

## 3. Annotate each RAW clip before scoring

For each recording, open the human annotation GUI. **Do not use the capture preview as ground truth**, since it showed algorithm predictions. Annotate the saved raw video independently.

```powershell
python pinch_benchmark_annotate.py --session session01 --clip-id distance_35cm_01 --assume-visible
```

The `--assume-visible` flag initializes a provisional "hand PRESENT" label for every frame. **Only use it if you visually review the full clip and confirm the hand is visible throughout.** If the hand enters/leaves the scene, mark transitions using the visibility keys. Without that flag, visibility defaults to UNKNOWN until you explicitly label it, and hand detection rate will show **N/A** rather than a fabricated percentage.

In the annotation GUI, the raw video occupies the left side and the controls appear in a separate right-hand panel, so no predicted labels cover the fingertips.

| Key | Action |
|---|---|
| `a`, `d` | Previous / next single frame |
| `j`, `l` | Back / forward 10 frames |
| `-`, `=` | Back / forward 30 frames |
| `Space` | Play / pause |
| `s` | Label first visible thumb-index **contact** (`PINCH_START`) |
| `r` | Label first visible **separation** (`PINCH_RELEASE`) |
| `u` | Toggle UNCERTAIN for event(s) on current frame |
| `x` | Delete event labels on current frame |
| `v` / `n` / `b` | Hand visibly PRESENT / ABSENT / UNKNOWN starting at this frame |
| `g` / `e` | Go to first / last frame |
| `f` | **Finalize** ground truth for this clip (status COMPLETE) |
| `q` / `Esc` | Save and exit (draft remains DRAFT until finalized) |

For negative-control videos, review **the entire video**, create **zero** START/RELEASE events if none occurred, mark hand visibility, then press `F` to explicitly certify the negative clip. A missing, empty or unfinished label file is **never** silently treated as a negative-control result.

For side-on or occluded fingertips, label the best visible frame and press `U` if timing is ambiguous. Uncertain ground truths are excluded from strict scoring; nearby predictions of the same event type (within the prespecified tolerance) are also excluded from FP counting. The excluded counts are reported transparently.

Annotations are saved as soon as you change them. You can revisit a finalized clip to make corrections: any change reverts its status to DRAFT until you press F again.

## 4. Replay the EXACT same landmarks and calculate results

When the trials have been annotated and finalized:

```powershell
python pinch_replay_evaluate.py --session session01 --frames 1 3 5 --pinch 0.30 --release 0.35 --lost-frames 6 --tolerance 0.50
```

The evaluator ignores incomplete recordings and draft annotations. **Pilot clips are excluded by default** to keep setup/calibration trials out of the reported evaluation. Use `--include-pilot` only if you have a clear reason to report them. Use `--plot-clip distance_35cm_01` to select the example ratio timeline. You can rerun evaluations without re-recording the video, since all 21 landmarks are stored.

The evaluator checks whether the replayed baseline (3 frames, 0.30/0.35 and six missing frames) reproduces the original per-frame live events, states and smoothed ratios. Inspect `baseline_replay_consistency.csv`. **If a REPLAY_MISMATCH is reported, investigate before using results in the paper.**

### Metrics and rules

- **Event TP/FP/FN**: `PINCH_START` and `PINCH_RELEASE` are matched separately, one-to-one, in temporal order. A dynamic program maximizes the number of matched events, then minimizes absolute time error. The match tolerance is **±0.50 s**, fixed before evaluating. `PINCH_CANCEL` is reported but is **not** treated as an ordinary release.
- **Precision / recall / F1**: calculated for START, RELEASE and combined ALL. Undefined scores are left blank / N/A, particularly for correctly empty negative controls. They are **never automatically called 100%**.
- **Gesture detection delay**: the *signed* difference between predicted event's `time_s` and the human-labelled physical contact/separation `time_s`. Also reports median absolute delay and p90 absolute delay. Negative signed delays are retained.
- **Negative-control false-start rate**: total predicted `PINCH_START` activations per minute, evaluated separately from positive-trial event F1. A long OPEN interval is **not** counted as thousands of true negatives to inflate accuracy.
- **Hand detection rate**: percentage of manually labelled hand-PRESENT video frames for which MediaPipe produced landmarks. If presence labels remain UNKNOWN, the metric is N/A. Also reports hand detections on visually ABSENT frames when these are labelled.
- **Processing time**: captured `frame_to_gesture_ms` is from the end of camera read until the gesture update returns. It does **not** include waiting for physical contact, the 3-frame confirmation, camera sensor buffering, or display response.
- **Ablation**: `frames=1,3,5` uses the same recorded landmark sequence, the same 0.30/0.35 thresholds and independent fresh detector instances for each clip. Pool TP/FP/FN before computing F1, instead of naively averaging per-clip F1 scores.

### Files created under `session01/results/`

| File | What it contains |
|---|---|
| `per_clip_metrics.csv` | Per-trial START/RELEASE/ALL counts, precision, recall, F1, delays, processing, visibility |
| `per_condition_metrics.csv` | Pooled counts and rates for distance, orientation and negative conditions |
| `confirmation_ablation.csv` | Confirmation window 1/3/5 comparison |
| `event_matching_audit.csv` | Each individual TP, FP, FN, and uncertain-excluded prediction |
| `replay_predictions.csv` | Detector state/ratio/event on every recorded frame for each confirmation window |
| `baseline_replay_consistency.csv` | Frame-by-frame comparison against the live recorded baseline |
| `summary.json` | Machine-readable aggregate metrics, settings and evaluated clip list |
| `RESULTS_README.md` | Automatically generated notes and measured ablation summary |
| `table_conditions.tex`, `table_negative.tex`, `table_ablation.tex` | Ready-to-edit LaTeX tables for your CVPR-style report |
| `figure_f1_by_condition.png/.pdf` | Contact and release F1 across test conditions |
| `figure_negative_false_starts.png/.pdf` | False activation rates for negative actions |
| `figure_ratio_timeline_<clip>.png/.pdf` | Raw/filtered ratio, thresholds, human-labelled and detected events |
| `figure_confirmation_f1.png/.pdf` | F1 tradeoff from varying confirmation window |
| `figure_confirmation_delay.png/.pdf` | Recognition-delay tradeoff from confirmation window |
| `figure_confirmation_false_starts.png/.pdf` | Negative-control false-start tradeoff |
| `figure_processing_times.png/.pdf` | Median/p95 frame-to-gesture computation time per clip |

The figures and tables are generated **only when reviewed real recordings exist**. If no positive/negative conditions have been recorded, some corresponding figures do not appear and their metrics are N/A. All `.pdf` plots are generated by Matplotlib from benchmark data, not the assignment PDF. All reported numeric results must be drawn from your own reviewed experiment session.

To include in a LaTeX paper after copying results into your project:

```latex
\input{results/table_conditions.tex}
\input{results/table_negative.tex}
\input{results/table_ablation.tex}
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{results/figure_confirmation_f1.pdf}
  \caption{Effect of temporal confirmation on event-level F1.}
\end{figure}
```

## 5. Source files

| File | Role |
|---|---|
| `pinch_detector.py` | **Original unmodified** pinch state machine, used during capture and replay |
| `pinch_webcam_v2.py` | Original standalone visual demo; only the camera mirror default/CLI was updated (detector logic and 0.25/0.30 demo thresholds retained) |
| `hand_landmarker.task` | Bundled MediaPipe Tasks model (7.8 MB) |
| `pinch_benchmark_capture.py` | Real-time webcam/video/landmark/CSV logger; no chess dependencies |
| `pinch_benchmark_annotate.py` | Human annotation GUI, visibility intervals, explicit completion status |
| `pinch_replay_evaluate.py` | Landmark replay, scoring, exports and baseline parity checking |
| `pinch_benchmark_common.py` | Shared file formats, camera-frame timebase and detector adapter |
| `pinch_benchmark_metrics.py` | Optimal event matching, uncertainty policy and detection statistics |
| `pinch_benchmark_plots.py` | Saved publication charts, PNG and vector PDF |
| `pinch_benchmark_tables.py` | Automatically populated LaTeX tables |
| `pinch_benchmark_run_protocol.py` | Interactive batch capture from experiment plan |
| `protocol/experiment_plan.csv` | The 11 main tests plus optional extensions |
| `tests/` | Repeatable synthetic correctness/integration tests (never used as results) |
| `chess_live.py`, `methodology_reference.tex`, `Pinch_Gesture_Experiments_Handoff.md` | Source references only; chess application is not run |

## 6. Practical quality controls and limitations

1. Confirm actual recorded resolution and measured FPS printed at capture time. The requested webcam FPS may not equal the measured processing FPS. Do not derive detection delay from nominal video FPS.
2. Review camera framing so that fingertips are visible at contact. Use a camera distance ruler/protractor or clearly describe distances/angles as approximate.
3. Keep annotations independent of the detector output. Complete every reviewed negative clip, even when it has zero events.
4. Inspect `event_matching_audit.csv` for systematic missed releases or early detections. Preserve failures rather than dropping difficult clips.
5. Report user/camera/lighting limitations and the time-sampling uncertainty of frame-based manual labels.
6. If replay differs from live inference (`REPLAY_MISMATCH`), stop and investigate detector settings or corrupted/misaligned frame records.
7. Only test data from the provided/captured hardware and model should support conclusions about real-world performance. Do not present the artificial tests in this package as empirical results.
8. The course requires acknowledgement of AI assistance as specified in its assignment guidance. Review, understand, test and verify all submitted code and analyses yourself.

**Troubleshooting:**
- Camera appears backwards: this package defaults to a horizontally **mirrored** selfie view. Add `--no-mirror` to capture, batch protocol, or standalone webcam demo for an unflipped view. Do not re-flip videos after capture.
- Camera won't open: try `--camera 0`, `--backend msmf` on Windows, or close Teams/Zoom.
- Missing MediaPipe: activate your venv and use `python -m pip install -r requirements.txt`; if wheels are unavailable for your Python build, try Python 3.11/3.12.
- Black/empty video or video writer error: rerun with a NEW clip ID and `--video-codec MJPG` (AVI), check permissions and free disk space.
- GUI cut off on a small monitor: pass `--scale 0.60` to the annotator.
- Need higher-fidelity annotation footage: use `--video-codec FFV1` (lossless AVI) if supported, and allow more disk space.
- Notebook/server without a GUI: capture with `--no-preview --duration 25`; the annotation tool itself needs an interactive desktop window.

## Final note

This framework implements the experiment design specified in `Pinch_Gesture_Experiments_Handoff.md`. **No observed accuracy, latency, or false-positive rate has yet been claimed.** Those numbers will come directly from your recorded, annotated pinch trials once you run the capture and evaluation commands.
