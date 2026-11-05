"""Record raw, frame-aligned pinch footage, landmarks, gesture states and timings.

Each invocation is one independent clip. Original PinchDetector logic is kept
unchanged; only degenerate landmarks are mapped to a missing-hand observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import cv2

from pinch_detector import PinchDetector
from pinch_benchmark_common import (
    FRAME_CSV_FIELDS, MANIFEST_FIELDS, clean_clip_id, ensure_session,
    manifest_by_id, replace_clip_rows, safe_detector_update, utc_now,
)

MODEL = Path(__file__).resolve().parent / "hand_landmarker.task"


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--session", required=True, help="Directory for all benchmark recordings and annotations")
    p.add_argument("--clip-id", required=True, help="Unique ID, e.g. distance_35_01")
    p.add_argument("--experiment", required=True, help="pilot, distance, orientation, negative, optional")
    p.add_argument("--condition", required=True, help="35cm / 45deg / pointing / open_palm, etc.")
    p.add_argument("--distance-cm", default="", help="Approximate/measured hand-to-camera distance")
    p.add_argument("--orientation-deg", default="", help="Approximate hand yaw relative to camera")
    p.add_argument("--participant", default="P01")
    p.add_argument("--hand", default="right", choices=("right", "left", "unknown"))
    p.add_argument("--lighting", default="normal")
    p.add_argument("--expected-cycles", type=int, default=0, help="10 for positive clips; 0 for controls")
    p.add_argument("--notes", default="")
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0, help="Requested camera and nominal video playback FPS")
    p.add_argument("--backend", choices=("auto", "dshow", "msmf"),
                   default="dshow" if os.name == "nt" else "auto")
    p.add_argument("--fourcc", choices=("MJPG", "YUY2", "auto"), default="MJPG",
                   help="Webcam capture pixel/compression format")
    p.add_argument("--video-codec", choices=("mp4v", "MJPG", "FFV1"), default="mp4v",
                   help="Saved unannotated video format; FFV1 AVI is lossless if available")
    mirror = p.add_mutually_exclusive_group()
    mirror.add_argument("--mirror", dest="mirror", action="store_true", default=True,
                        help="Mirror horizontally (selfie view): preview, saved video and detector input; DEFAULT")
    mirror.add_argument("--no-mirror", dest="mirror", action="store_false",
                        help="Keep the camera's native left-right orientation")
    p.add_argument("--duration", type=float, default=0.0, help="Recording duration (0 = until Q/Esc/Ctrl+C)")
    p.add_argument("--max-frames", type=int, default=0, help="Optional frame cap")
    p.add_argument("--warmup-frames", type=int, default=10, help="Unread/unprocessed warmup frames")
    p.add_argument("--no-preview", action="store_true", help="Headless capture; use --duration")
    p.add_argument("--pinch", type=float, default=0.30)
    p.add_argument("--release", type=float, default=0.35)
    p.add_argument("--frames", type=int, default=3, help="Consecutive-frame confirmation")
    p.add_argument("--lost-frames", type=int, default=6)
    p.add_argument("--detection-confidence", type=float, default=0.6)
    p.add_argument("--presence-confidence", type=float, default=0.5)
    p.add_argument("--tracking-confidence", type=float, default=0.5)
    p.add_argument("--model", type=Path, default=MODEL, help="MediaPipe .task model")
    args = p.parse_args()
    clean_clip_id(args.clip_id)
    if not 0 < args.pinch < args.release or min(args.frames, args.lost_frames) < 1:
        p.error("Need 0 < pinch < release, frames >= 1 and lost-frames >= 1")
    if min(args.width, args.height, args.fps) <= 0 or args.warmup_frames < 0:
        p.error("Width/height/FPS must be positive, warmup-frames >= 0")
    if args.duration < 0 or args.max_frames < 0 or args.expected_cycles < 0:
        p.error("Duration/max-frames/expected-cycles must be nonnegative")
    if args.no_preview and args.duration <= 0 and args.max_frames <= 0:
        p.error("Use --duration or --max-frames when --no-preview is set")
    for c in (args.detection_confidence, args.presence_confidence, args.tracking_confidence):
        if not 0 <= c <= 1:
            p.error("Confidence thresholds must be in [0,1]")
    return args


def open_camera(args):
    opts = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    cam = cv2.VideoCapture(args.camera, opts[args.backend])
    if not cam.isOpened():
        raise RuntimeError(f"Camera {args.camera} not available ({args.backend}). Try --camera 0 or a different backend")
    if args.fourcc != "auto":
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cam.set(cv2.CAP_PROP_FPS, args.fps)
    return cam


def draw_preview(frame, landmarks, gesture, frame_idx, time_s, ms):
    """Only draw on a copy for display, never on the stored video."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    if landmarks is not None and len(landmarks) == 21:
        points = [(round(p.x*width), round(p.y*height)) for p in landmarks]
        edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                 (13,17),(17,18),(18,19),(19,20),(0,17)]
        for a, b in edges:
            cv2.line(canvas, points[a], points[b], (190,190,190), 2)
        for point in points:
            cv2.circle(canvas, point, 3, (0,225,255), -1)
    cv2.rectangle(canvas, (0, 0), (width, 96), (25,25,25), -1)
    text = f"{gesture['state']} | event={gesture['event'] or '-'} | ratio={gesture['ratio'] if gesture['ratio'] is not None else '-'}"
    cv2.putText(canvas, text[:100], (12,32), cv2.FONT_HERSHEY_SIMPLEX, .64, (255,255,255), 2)
    cv2.putText(canvas, f"Frame {frame_idx:05d}  capture t={time_s:.3f}s  processing={ms:.1f}ms",
                (12,62), cv2.FONT_HERSHEY_SIMPLEX, .57, (255,255,255), 1)
    cv2.putText(canvas, "Q / Esc: end this recording", (12,86),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    return canvas


def main():
    args = arguments()
    session = ensure_session(args.session)
    old = manifest_by_id(session)
    if args.clip_id in old:
        raise SystemExit(f"Clip {args.clip_id!r} already in manifest; choose a new --clip-id (no overwrite)")
    ext = ".mp4" if args.video_codec == "mp4v" else ".avi"
    video_rel = f"recordings/{args.clip_id}{ext}"
    log_rel = f"landmarks/{args.clip_id}.jsonl"
    frames_rel = f"frames/{args.clip_id}.csv"
    paths = [session / video_rel, session / log_rel, session / frames_rel]
    if any(path.exists() for path in paths):
        raise SystemExit("Files already exist for this clip. Use a new --clip-id to protect previous recordings")
    if not args.model.is_file():
        raise SystemExit(f"Cannot find MediaPipe model at {args.model}; keep hand_landmarker.task beside this script")

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit("MediaPipe not installed. Run: python -m pip install -r requirements.txt") from exc

    config = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.detection_confidence,
        min_hand_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )
    detector = PinchDetector(pinch=args.pinch, release=args.release,
                             frames=args.frames, lost_frames=args.lost_frames)
    # Camera and model both need to be available before new data files are opened.
    camera = open_camera(args)
    writer = None
    nframes = 0
    first_capture = None
    prev_capture = None
    monotonic_origin = time.monotonic()
    mp_previous_ts = -1
    started_utc = utc_now()
    fatal = None
    n_invalid = 0
    camera_fps = float(camera.get(cv2.CAP_PROP_FPS))
    actual_width = actual_height = 0
    log_handle = csv_handle = None
    print(f"Recording {args.clip_id} ({args.experiment}: {args.condition})")
    print("Raw video is saved without overlays. Start OPEN, then perform your protocol.")
    print("Camera view:", "MIRRORED horizontally (selfie view)" if args.mirror else "NATIVE (not mirrored)")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(config) as landmarker:
            for _ in range(args.warmup_frames):
                camera.read()
            # Exclusive creation prevents destroying an existing benchmark clip.
            log_handle = paths[1].open("x", encoding="utf-8", buffering=1)
            csv_handle = paths[2].open("x", encoding="utf-8", newline="")
            csv_writer = csv.DictWriter(csv_handle, fieldnames=FRAME_CSV_FIELDS)
            csv_writer.writeheader()
            while True:
                t0 = time.perf_counter()
                ok, frame = camera.read()
                t1 = time.perf_counter()
                if not ok or frame is None:
                    print("WARNING: Camera returned no frame; stopping.")
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if writer is None:
                    actual_width, actual_height = w, h
                    writer = cv2.VideoWriter(str(paths[0]),
                                              cv2.VideoWriter_fourcc(*args.video_codec),
                                              args.fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"VideoWriter failed for {paths[0]}. Try --video-codec MJPG")
                    print(f"Actual saved frame: {w} x {h}; reported camera FPS: {camera_fps:.1f}")
                    print(f"Playback video FPS {args.fps:g} is NOMINAL; real capture times are in JSONL/CSV")
                elif (w, h) != (actual_width, actual_height):
                    raise RuntimeError("Camera resolution changed midclip; frame-to-video alignment cannot be guaranteed")

                capture_now = time.monotonic()
                if first_capture is None:
                    first_capture = capture_now
                elapsed = capture_now - first_capture
                if prev_capture is not None and elapsed <= prev_capture:
                    elapsed = prev_capture + 1e-6
                prev_capture = elapsed
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                stamp = max(mp_previous_ts + 1, int((capture_now - monotonic_origin)*1000))
                mp_previous_ts = stamp
                t2 = time.perf_counter()
                result = landmarker.detect_for_video(image, stamp)
                t3 = time.perf_counter()
                raw_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
                hand_detected = raw_landmarks is not None
                # Rare corrupt/NaN model outputs must not destroy an entire clip.
                # Treat unusable landmarks as an absent hand for the state machine.
                finite_points = (raw_landmarks is not None and len(raw_landmarks) == 21
                                 and all(math.isfinite(float(value)) for point in raw_landmarks
                                         for value in (point.x, point.y, point.z)))
                landmarks = raw_landmarks if finite_points else None
                gesture, raw_ratio, valid, palm = safe_detector_update(detector, landmarks, w, h)
                t4 = time.perf_counter()
                if hand_detected and not valid:
                    n_invalid += 1
                t_write = time.perf_counter()
                writer.write(frame)
                t_written = time.perf_counter()
                times = {
                    "capture_ms": (t1-t0)*1000,
                    "preprocess_ms": (t2-t1)*1000,
                    "mediapipe_ms": (t3-t2)*1000,
                    "gesture_ms": (t4-t3)*1000,
                    "frame_to_gesture_ms": (t4-t1)*1000,
                    "video_write_ms": (t_written-t_write)*1000,
                }
                key = -1
                if not args.no_preview:
                    display = draw_preview(frame, landmarks, gesture, nframes, elapsed,
                                           times["frame_to_gesture_ms"])
                    cv2.imshow("Pinch benchmark capture (overlay never saved)", display)
                    key = cv2.waitKey(1) & 0xFF
                # Full-loop measure includes camera, inference, video write and preview;
                # CSV/JSONL serialization follows this timestamp and is not included.
                times["full_loop_ms"] = (time.perf_counter()-t0)*1000
                cursor, event_point = gesture["position"], gesture["event_position"]
                record = {
                    "clip_id": args.clip_id,
                    "frame_index": nframes,
                    "time_s": round(elapsed, 9),
                    "video_time_s": round(nframes/args.fps, 9),
                    "mp_timestamp_ms": stamp,
                    "width": w,
                    "height": h,
                    "hand_detected": hand_detected,
                    "valid_landmarks": valid,
                    "landmarks": [[float(p.x), float(p.y), float(p.z)] for p in landmarks]
                                 if landmarks is not None else None,
                    "palm_size_px": palm,
                    "raw_ratio": raw_ratio,
                    "smoothed_ratio": gesture["ratio"],
                    "state": gesture["state"],
                    "event": gesture["event"],
                    "cursor_x": cursor[0] if cursor else None,
                    "cursor_y": cursor[1] if cursor else None,
                    "event_x": event_point[0] if event_point else None,
                    "event_y": event_point[1] if event_point else None,
                    **times,
                }
                log_handle.write(json.dumps(record, allow_nan=False, separators=(",", ":"))+"\n")
                csv_writer.writerow({key: ("" if value is None else value)
                                     for key, value in record.items() if key in FRAME_CSV_FIELDS})
                nframes += 1
                if gesture["event"]:
                    print(f"{args.clip_id}: frame={nframes-1} time={elapsed:.3f} "
                          f"event={gesture['event']} filtered={gesture['ratio']}")
                if key in (27, ord("q")):
                    break
                if args.duration and elapsed >= args.duration:
                    break
                if args.max_frames and nframes >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print("Capture interrupted by Ctrl+C; saved frames will remain in this clip")
    except Exception as exc:
        fatal = exc
        print(f"ERROR while capturing: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        if log_handle is not None:
            log_handle.close()
        if csv_handle is not None:
            csv_handle.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    if nframes == 0:
        for path in paths:
            path.unlink(missing_ok=True)
        raise SystemExit("No video frames recorded. Nothing was added to the manifest")

    # Read the exact number of serialized frames as a post-capture integrity check.
    with paths[1].open(encoding="utf-8") as stream:
        nlogged = sum(1 for line in stream if line.strip())
    with paths[2].open(encoding="utf-8", newline="") as stream:
        ncsv = sum(1 for _ in csv.DictReader(stream))
    status = "COMPLETE" if fatal is None and nlogged == ncsv == nframes else "INCOMPLETE"
    verify = cv2.VideoCapture(str(paths[0]))
    decoded_count = int(verify.get(cv2.CAP_PROP_FRAME_COUNT)) if verify.isOpened() else 0
    verify.release()
    if decoded_count > 0 and decoded_count != nframes:
        status = "INCOMPLETE"
        print(f"WARNING: video reports {decoded_count} frames but log has {nframes}")
    measured_fps = (nframes - 1)/prev_capture if nframes > 1 and prev_capture and prev_capture > 0 else 0.0
    row = {
        "clip_id": args.clip_id, "experiment": args.experiment,
        "condition": args.condition, "distance_cm": args.distance_cm,
        "orientation_deg": args.orientation_deg, "participant": args.participant,
        "hand": args.hand, "lighting": args.lighting,
        "expected_cycles": args.expected_cycles, "notes": args.notes,
        "video_file": video_rel, "log_file": log_rel, "frames_file": frames_rel,
        "frame_count": nframes, "duration_s": round(prev_capture, 6),
        "capture_fps_measured": round(measured_fps, 4),
        "video_fps_nominal": args.fps, "width": actual_width, "height": actual_height,
        "camera_index": args.camera, "camera_backend": args.backend,
        "camera_fourcc": args.fourcc, "camera_fps_reported": camera_fps,
        "requested_width": args.width, "requested_height": args.height,
        "requested_fps": args.fps, "mirror": args.mirror,
        "pinch": args.pinch, "release": args.release,
        "confirmation_frames": args.frames, "lost_frames": args.lost_frames,
        "min_detection_confidence": args.detection_confidence,
        "min_presence_confidence": args.presence_confidence,
        "min_tracking_confidence": args.tracking_confidence,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "python_version": platform.python_version(), "opencv_version": cv2.__version__,
        "mediapipe_version": mp.__version__, "os_version": platform.platform(),
        "created_utc": started_utc, "status": status,
    }
    replace_clip_rows(session / "manifest.csv", MANIFEST_FIELDS, args.clip_id, [row])
    print(f"Saved {nframes} aligned video/JSONL/CSV frames. Status: {status}")
    print(f"Raw video: {paths[0]} | landmark stream: {paths[1]} | actual processed FPS: {measured_fps:.2f}")
    print(f"Degenerate landmark observations: {n_invalid}. Annotate next, then evaluate.")
    if fatal is not None:
        raise SystemExit(1)
    if status != "COMPLETE":
        raise SystemExit("Capture integrity check failed. Inspect this clip before evaluating")


if __name__ == "__main__":
    main()
