# AMME5710 Major Project | Pinch Gesture Benchmarking Handoff

**Status:** Experiment design and implementation are the next steps. No benchmark videos have been recorded, no hand-labelled ground truth has been supplied, and no accuracy/latency results have been measured yet.

**Scope:** This handoff is **only for standalone pinch gesture recognition**. The physical chessboard is no longer available. Do not require board detection, camera pose, rendered chess pieces, or chess-move validation for the proposed experiments.

## 1. Project context and objective

The team is preparing an eight-page, two-column CVPR-style report for AMME5710 Computer Vision and Image Processing. The paper presents an augmented-reality chess system with board tracking, PnP camera pose, and MediaPipe hand-gesture interaction. My contribution includes pinch detection. The methodology is drafted; the immediate priority is to **design, implement, conduct, and analyse rigorous pinch-only experiments** that provide useful quantitative results and material for the Experimental Setup & Benchmarking, Results, and Discussion sections.

The chosen methodology in `methodology (1).tex` describes 21 MediaPipe hand landmarks, a palm-normalised thumb/index pinch ratio, equal-weight temporal ratio smoothing, hysteresis thresholds of **0.30 (pinch)** and **0.35 (release)**, confirmation over **three consecutive processed frames**, and cancellation after **six missing hand observations**. Preserve those settings as the **final-system benchmark baseline**, unless deliberately testing a documented alternative.

## 2. Source files required for the next chat

The files below are all present in the uploaded `src.zip`. A reduced **Pinch_Benchmark_Starter_Pack.zip** accompanies this handoff and contains them at the archive root.

| File | Required? | Role |
|---|---|---|
| `pinch_detector.py` | **Essential** | Implements normalisation, filtered pinch ratio, two-threshold state machine, frame confirmation, hand-loss cancellation and gesture events. |
| `pinch_webcam_v2.py` | **Essential** | Standalone webcam + MediaPipe test runner with live overlays, CLI options and existing per-frame timing CSV. Recommended starting point for benchmark capture and logging. |
| `hand_landmarker.task` | **Essential** | MediaPipe Tasks Hand Landmarker model used by the Python runners. Must be placed alongside the standalone script, or it will attempt to download the model. |
| `chess_live.py` | **Reference only** | Integrated application's CLI defaults and HandWorker configuration. Do **not** run for these board-free tests: it imports rendering and board-related modules. |
| `methodology_reference.tex` | **Reference only** | Unmodified copy of the user's current `methodology (1).tex`, containing the intended gesture methodology wording. |
| `requirements.txt` | **Helpful** | Minimal Python packages for the standalone runner; exact installed versions should be recorded on the test machine. |

**Not needed for the pinch-only experiments:** `board_tracker.py`, `board_tracker_v2.py`, `find_board.py`, `get_pose_v6.py`, `pose_quality.py`, `real_board_scene.py`, `real_gui.py`, `cal_data.pkl`, or a physical chessboard.

**Dependencies:** Python with `opencv-python` and `mediapipe`. Standard-library modules used include `argparse`, `csv`, `time` and `math`. Record Python version, `cv2.__version__`, MediaPipe version, computer, camera model and operating system for reproducibility. Existing code uses the MediaPipe **Tasks** API (`mp.tasks.vision.HandLandmarker`), not the legacy `mp.solutions.hands` interface.

## 3. Verified current implementation (from `src.zip`)

### MediaPipe input

- The standalone runner (`pinch_webcam_v2.py`) uses `HandLandmarker` in **VIDEO** mode, with **one hand** (`num_hands=1`). Confidence thresholds are detection **0.6**, presence **0.5**, and tracking **0.5**.
- Requested capture defaults: **1280 x 720 at 30 fps**, camera index **1**. On Windows the default backend is `dshow`; the requested resolution/FPS may not be provided by the camera. `--mirror` is optional and **off** by default. Record actual settings.
- Each loop reads a camera frame, calls MediaPipe, then calls `PinchDetector.update(landmarks, width, height)`. The standalone runner is not the threaded AR integration.

### PinchDetector logic

- Thumb tip = landmark **4**; index tip = **8**; wrist = **0**; middle-finger base = **9**; index base = **5**; little-finger base = **17**.
- Palm size = `max(distance(0,9), distance(5,17))` in image pixels. Raw ratio = `distance(4,8) / palm_size`.
- Smoothed ratio = `0.5 * raw_ratio + 0.5 * previous_smoothed_ratio` (initialised from the first valid ratio).
- When OPEN, `ratio < pinch_threshold` increments the consecutive-frame counter. When PINCHED, `ratio > release_threshold` increments it. Otherwise it resets. On reaching `frames`, an event is generated: `PINCH_START` or `PINCH_RELEASE`.
- **Six consecutive absent hand observations** cancel an active pinch with `PINCH_CANCEL`. Cancellation is a distinct event, not a release.
- The cursor uses a point one-quarter of the way from the thumb to the index fingertip and is smoothed separately (`0.55 * current_contact + 0.45 * previous_cursor`). Cursor tracking is not the primary focus of the proposed benchmarks.
- The standalone webcam script currently logs the **smoothed** ratio only to the screen. Its CSV does **not** include raw/smoothed ratio, landmarks, video timestamps, or independent ground-truth labels.

### Important discrepancy between scripts

| Source of defaults | Pinch | Release | Confirmation |
|---|---:|---:|---:|
| Current methodology, `methodology_reference.tex` | **0.30** | **0.35** | **3 frames** |
| Integrated application's CLI, `chess_live.py` | **0.30** | **0.35** | **3 frames** |
| Standalone script CLI, `pinch_webcam_v2.py` | 0.25 | 0.30 | 3 frames |
| `PinchDetector` constructor, `pinch_detector.py` | 0.25 | 0.30 | 3 frames |

**Action:** Run standalone tests with explicit `--pinch 0.30 --release 0.35 --frames 3` so the report is consistent. Do not mistake the standalone defaults for the final integrated settings.

### Existing working baseline command

```bash
python pinch_webcam_v2.py --camera 1 --width 1280 --height 720 --fps 30 --pinch 0.30 --release 0.35 --frames 3 --timing-csv pilot_timings.csv
```

If the webcam is device 0, use `--camera 0`. The current runner provides **visual feedback and processing timings**, but **does not record raw video or all benchmark fields**. It needs extending before the quantitative collection.

## 4. Recommended experiments (board-free)

Run a short pilot first to confirm that the camera, landmarks, recording and annotations work. Freeze the baseline parameters before recording the evaluation set. Control lighting, resolution, hand, and approximate motion speed as far as practical, and vary **one factor at a time**.

| Experiment | Conditions | Suggested sample size | Main question |
|---|---|---|---|
| **A. Distance robustness** | Hand approximately **35, 60, 100 cm** from camera | **10 full pinch/release cycles** per distance | Does palm normalisation maintain detection as apparent hand size changes? |
| **B. Hand orientation** | At ~60 cm: palm facing camera, **45°** sideways, **~75°** sideways; optionally add **90° in-plane rotation** as separate condition | **10 full cycles** per orientation | How do viewing angle and fingertip occlusion affect recognition? |
| **C. Negative controls** | Open palm, pointing, closed fist, moving/waving hand without touching thumb and index finger | **20-30 s** per action | How often does the detector invent a pinch? |
| **D. Temporal confirmation ablation** | Replay **identical recorded landmarks** with `frames=1`, `3`, and `5`; thresholds remain 0.30/0.35 | Reuse all labelled clips | Trade-off between accidental events, missed brief pinches, and recognition delay |

**Capture protocol:** For each positive clip, begin OPEN, perform 10 clear pinch-release cycles, briefly hold each pinch and each open position (e.g. 0.5-1 s), and finish OPEN. Do not rush. Use the same participant/hand initially, unless explicitly evaluating different users. Mark nominal hand distances and angles as approximate unless measured. Save each condition as a **separate file** and keep a small manifest of lighting, camera settings, participant/hand, distance and orientation.

**Optional only if time permits:** a low-light clip; faster and slower pinches; the opposite hand; a short-term hand-loss test to verify `PINCH_CANCEL`. These should not displace the main experiments.

## 5. Ground truth and event evaluation

**Ground truth cannot be inferred from the detector's own thresholds or outputs.** Record the raw hand video first. Annotate the video manually, preferably frame by frame, with the first visible thumb-index fingertip contact as `PINCH_START` and first visible separation as `PINCH_RELEASE`. Mark ambiguous frames (especially side-on views). Negative-control clips should contain no labelled pinch events.

Proposed annotation CSV columns:

```text
clip_id,event_id,event_type,frame_index,time_s,uncertain,notes
```

Recommended scorer:

1. Compare predicted events to ground-truth events **of the same type**, matching **one-to-one** within a prespecified time tolerance (initial proposal **±0.50 s**; do not change this after looking at results). An event can match at most one annotation.
2. Count **TP** (matched), **FP** (unmatched predictions), and **FN** (unmatched annotations) separately for pinch starts and releases. Report precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 = 2PR/(P+R). State how undefined scores are handled when there are no positive examples.
3. For matched events, report **signed event delay** = predicted event timestamp minus annotated event timestamp, median absolute delay, and optionally the 90th percentile. Negative delay should be retained as a sign of annotation/threshold mismatch rather than silently set to zero.
4. For negative-control videos, report **false `PINCH_START` events per minute** and the total number of false activations. An inflated frame-level accuracy over long OPEN intervals is not very informative.
5. Report the **hand detection rate** (% visually hand-present frames where landmarks are returned), especially for side-on orientations. This helps separate MediaPipe landmark failures from failures of the custom state machine.
6. Record `PINCH_CANCEL` events and hand-loss intervals separately. Do not treat cancellation as an ordinary pinch release.

**Two different forms of 'latency' must not be confused:**

- **Gesture detection delay:** time between manually labelled physical contact/separation and the emitted `PINCH_START`/`PINCH_RELEASE`. Requires a common video/frame timeline and annotations.
- **Processing time:** elapsed time from after camera capture to the gesture result, already measured as `frame_to_gesture_ms` in the current standalone script. It does **not** include time waiting for a physical gesture to be confirmed, and must not be presented as the human-to-system recognition latency.

## 6. Required code additions (proposed, not yet implemented)

**First priority: extend `pinch_webcam_v2.py`, while keeping `pinch_detector.py` algorithmically unchanged for the baseline.**

- Record a **raw, unannotated video** per test clip. Optionally write a separate overlay/debug video, but use raw footage for manual ground truth.
- Log one row per **processed** frame: `clip_id`, `frame_index`, video/capture timestamp or a consistent frame-time reference, hand present, 21 normalised landmark `(x,y)` pairs (or a parallel JSONL file), raw ratio, smoothed ratio, state, event, event position, and processing timings.
- Record actual frame dimensions, elapsed time and FPS; don't assume that requested 30 fps equals actual throughput. Ensure events, video and landmark logs use consistent frame IDs and timestamps.
- Save a metadata file/manifest with conditions, camera, OS, model and parameter settings. Reset the detector at the start of each independent clip.
- Prefer capturing **landmark sequences once** and running every ablation through fresh `PinchDetector` instances offline. This holds MediaPipe outputs constant when comparing state-machine settings; using independently reprocessed video might add upstream variation.
- Create a small annotation tool or simple frame-index workflow, then a **replay/scoring script** that outputs per-condition TP/FP/FN, precision, recall, F1, delay and false-event rates. Export plots and tables from real measurements only.
- Small defensive issue: `pinch_detector.py` returns `None` if estimated palm size is under 1 px, whereas the standalone webcam runner assumes a dictionary. Handle or record this exceptional case consistently rather than silently dropping such a frame.

**Suggested files to create:** `pinch_benchmark_capture.py`, `pinch_replay_evaluate.py`, `ground_truth_events.csv`, `recordings/`, `landmarks/`, `results/` and `benchmark_manifest.csv`. These names are suggestions, not existing uploaded files.

Suggested eventual directory structure:

```text
pinch_benchmark/
  pinch_detector.py
  pinch_webcam_v2.py
  hand_landmarker.task
  pinch_benchmark_capture.py          # TO CREATE
  pinch_replay_evaluate.py            # TO CREATE
  benchmark_manifest.csv              # TO CREATE
  recordings/*.mp4                    # TO COLLECT
  landmarks/*.csv or *.jsonl          # TO COLLECT
  ground_truth_events.csv             # TO ANNOTATE
  results/*.csv and figures/*         # TO GENERATE
```

## 7. Results to produce for the CVPR paper

- **Main table:** Positive gesture detection by distance and orientation: number of annotated START/RELEASE events, precision, recall, F1, median event delay, hand detection rate. Add a separate row or mini-table for negative-control false-start rate.
- **Main figure:** A small multi-condition F1 or recall chart, plus a representative time series of raw/filtered pinch ratio with 0.30/0.35 thresholds, ground-truth transitions, and emitted events.
- **Ablation figure or compact table:** Compare confirmation windows `1/3/5` for F1, false events and median event delay.
- **Discussion points:** Why palm normalisation does/does not generalise with distance; landmark occlusion at side orientations; trade-off between filtering robustness and response speed; limited sample size and one-user/one-camera generalisation; distinction between MediaPipe errors and state-machine errors.
- **Do not invent values or claim superior performance before testing.** Record failures as well as successes. Be cautious when reporting an 'accuracy' percentage derived from mostly OPEN frames.

## 8. Immediate next-chat objective and ready-to-paste prompt

> I am continuing an AMME5710 computer vision major project. I can no longer access the physical chessboard. Work **only on standalone pinch gesture recognition benchmarking**. The accompanying handoff documents my final methodology, the exact existing `PinchDetector`, MediaPipe runner, benchmark design, and required files. First, inspect the supplied code and implement a reliable capture/logging workflow that records raw webcam video, all 21 landmarks, per-frame timestamps, raw and smoothed ratios, events, and processing timings. Keep the baseline pinch algorithm unchanged (0.30 close, 0.35 release, 3-frame confirmation, cancel after 6 missed frames). Then help me record/annotate distance, hand-orientation and negative-control trials and implement offline replay + event-matched metrics. Distinguish recognition delay from computation time. Output results suitable for a short CVPR-style Experimental Setup, Results and Discussion. Do not fabricate results or require board-related files.

## 9. Practical checklist

- [ ] Upload `Pinch_Benchmark_Starter_Pack.zip` and this handoff to the new chat.
- [ ] Confirm Python, camera index, MediaPipe model loading, actual resolution/FPS and baseline 0.30/0.35/3 parameters.
- [ ] Extend capture script with lossless event/frame correspondence, source video, landmarks, timestamped logs and metadata.
- [ ] Pilot one clip and verify that physical contacts can be annotated without seeing predicted labels.
- [ ] Record distance, orientation and negative-control clips using a fixed protocol.
- [ ] Label START/RELEASE events independently of model predictions.
- [ ] Run baseline scoring and same-landmark confirmation ablations (1/3/5).
- [ ] Build reproducible charts/tables; only then write numerical Results and Discussion paragraphs.

**Source note:** Facts above were checked against the user's uploaded `src.zip` and `methodology (1).tex`; the experimental protocol, metrics and suggested new scripts are proposals rather than existing results or implemented functionality.
