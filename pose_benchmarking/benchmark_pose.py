"""Offline evaluation of the EXISTING 8x8 chessboard pose pipeline.

  python benchmark_pose.py --mode all
  python benchmark_pose.py --mode images
  python benchmark_pose.py --mode videos --max-frames 300
  python benchmark_pose.py --mode annotate --image pose_benchmark/images/clear/p01.jpg
  python benchmark_pose.py --mode synthetic --synthetic-trials 60
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

BOARD_N = 8
SQUARE_M = 0.057
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "pose_benchmark"
IMAGES_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

COLUMNS = [
    "source", "source_type", "condition", "frame", "time_s", "image_width", "image_height",
    "tracking_ok", "tracked_count", "orientation_status", "detector_run", "detector_ok",
    "detector_message", "pose_status", "pose_reason", "pnp_solved", "pose_accepted",
    "corner_rmse_px", "max_corner_error_px", "inner_rmse_px", "max_error_squares",
    "manual_points", "manual_rmse_px", "manual_median_px", "manual_rmse_squares",
    "camera_x_board_m", "camera_y_board_m", "camera_z_board_m",
    "rotation_step_deg", "camera_step_m", "detect_ms", "tracker_ms", "pnp_ms",
    "pose_ms", "gate_ms", "processing_ms",
]


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("all", "images", "videos", "synthetic", "annotate"), default="all")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--calibration", type=Path, default=BASE_DIR / "cal_data.pkl")
    p.add_argument("--square-size", type=float, default=SQUARE_M, help="Board square pitch in metres")
    p.add_argument("--detect-every", type=int, default=60, help="Detection frequency during video; LK runs every frame")
    p.add_argument("--pose-gate", choices=("relative", "pixels"), default="relative")
    p.add_argument("--max-corner-error-squares", type=float, default=0.60, help="Current real_board_scene.py threshold; v8 standalone runner uses 0.25")
    p.add_argument("--max-corner-error", type=float, default=8.0)
    p.add_argument("--max-frames", type=int, default=0, help="Per-video frame limit; 0 means all")
    p.add_argument("--sample-every", type=int, default=120, help="Save one sample every N video frames, plus status transitions; 0 disables periodic samples")
    p.add_argument("--max-samples", type=int, default=16, help="Max representative images per video")
    p.add_argument("--synthetic-trials", type=int, default=60, help="Trials per noise level per viewing condition")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image", type=Path, help="Image file for --mode annotate")
    p.add_argument("--grid-points", default="1:1,1:4,1:7,4:1,4:4,4:7,7:1,7:4,7:7",
                   help="Annotated inner intersections as row:col, starting from corner 0")
    p.add_argument("--no-figures", action="store_true", help="Write CSVs only")
    a = p.parse_args(argv)
    if a.detect_every < 1 or a.max_frames < 0 or a.sample_every < 0 or a.max_samples < 0:
        p.error("Invalid detection interval, frame/sample limit or sample interval")
    if a.synthetic_trials < 1 or a.square_size <= 0 or a.max_corner_error <= 0 or a.max_corner_error_squares <= 0:
        p.error("Square size, limits and synthetic trials must be positive")
    return a


@dataclass
class Pipeline:
    detect: object
    tracker_class: object
    get_pose: object
    load_calibration: object
    pose_result_class: object
    evaluate_gate: object
    object_outer_corners: object


def load_pipeline() -> Pipeline:
    """Import project implementation lazily, so annotation works without it."""
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    try:
        detector = importlib.import_module("find_board")
        tracker_mod = importlib.import_module("board_tracker_v2")
        pose_mod = importlib.import_module("get_pose_v6")
        try:
            quality = importlib.import_module("pose_quality_v8")
        except ModuleNotFoundError as exc:
            if exc.name != "pose_quality_v8":
                raise
            quality = importlib.import_module("pose_quality")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing project module {exc.name!r}. Copy benchmark_pose.py into your main "
            "project folder beside find_board.py, board_tracker.py, board_tracker_v2.py, "
            "get_pose_v6.py and pose_quality_v8.py (or pose_quality.py)."
        ) from exc
    return Pipeline(
        detector.detect_chessboard, tracker_mod.BoardTracker,
        pose_mod.get_board_pose, pose_mod.load_calibration, pose_mod.PoseResult,
        quality.evaluate_corner_gate, pose_mod.object_outer_corners,
    )


def folders(a):
    for name in ("images/clear", "images/occluded", "videos", "annotations", "results",
                 "figures/images", "figures/videos"):
        (a.data_dir / name).mkdir(parents=True, exist_ok=True)


def relative_key(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def files_under(path: Path, extensions):
    return sorted(x for x in path.rglob("*") if x.is_file() and x.suffix.lower() in extensions)


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved {path} ({len(rows)} rows)")


def load_manual_points(data_dir: Path):
    """Human image-space labels, never used to solve or choose a pose."""
    labels = {}
    for row in read_csv(data_dir / "annotations/image_points.csv"):
        if not (row.get("image") or "").strip():
            continue
        key = row["image"].strip().replace("\\", "/")
        try:
            rr, cc = int(row["grid_row"]), int(row["grid_col"])
            uv = np.array([float(row["x_px"]), float(row["y_px"])], dtype=float)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid manual point in {key}: {row}") from exc
        if not 1 <= rr <= 7 or not 1 <= cc <= 7 or not np.isfinite(uv).all():
            raise ValueError(f"Manual labels require row/col 1..7 and finite pixel coordinates: {row}")
        group = labels.setdefault(key, {})
        if (rr, cc) in group:
            raise ValueError(f"Duplicate manual point {key}, row {rr}, col {cc}")
        group[(rr, cc)] = uv
    return labels


def load_occlusion_intervals(data_dir: Path):
    by_video = {}
    for row in read_csv(data_dir / "annotations/occlusion_intervals.csv"):
        key = (row.get("video") or "").strip().replace("\\", "/")
        if not key:
            continue
        try:
            start, end = float(row["start_s"]), float(row["end_s"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid occlusion interval: {row}") from exc
        if not 0 <= start < end:
            raise ValueError(f"Each occlusion interval must have 0 <= start < end: {row}")
        by_video.setdefault(key, []).append((start, end))
    for key, ranges in by_video.items():
        ranges.sort()
        if any(x[1] > y[0] for x, y in zip(ranges, ranges[1:])):
            raise ValueError(f"Overlapping occlusion intervals in {key}")
    return by_video


def video_phase(t, ranges):
    if not ranges:
        return "unlabelled"
    return "occluded" if any(start <= t < end for start, end in ranges) else "clear"


def median(v):
    data = np.array([float(x) for x in v if x is not None], dtype=float)
    data = data[np.isfinite(data)]
    return float(np.median(data)) if len(data) else float("nan")


def p95(v):
    data = np.array([float(x) for x in v if x is not None], dtype=float)
    data = data[np.isfinite(data)]
    return float(np.percentile(data, 95)) if len(data) else float("nan")


def rotation_difference_deg(R1, R2):
    if R1 is None or R2 is None:
        return float("nan")
    a = float(np.clip((np.trace(R1 @ R2.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(a)))


def check_image_geometry(frame, calibration, source):
    image_size = getattr(calibration, "image_size", None)
    actual = (frame.shape[1], frame.shape[0])
    if image_size is not None and tuple(image_size) != actual:
        raise ValueError(f"Calibration resolution {image_size} != {actual} for {source}. "
                         "No automatic resizing, cropping, or intrinsics scaling is performed.")


def detect_or_track(frame, tracker, pipe, run_detector):
    """Copy the v8 runner's detector scheduling and tracker update sequence."""
    detection_ok = False
    result = None
    detect_ms = float("nan")
    tracker_ms = float("nan")
    message = "Optical flow only"
    if run_detector:
        start = time.perf_counter()
        try:
            detection = pipe.detect(frame, video_mode=False,
                                    display_output=False, display_debug=False)
            detect_ms = (time.perf_counter() - start) * 1000
            start_update = time.perf_counter()
            result = (tracker.update(frame, detection=detection)
                      if tracker.initialised else tracker.initialize(frame, detection))
            tracker_ms = (time.perf_counter() - start_update) * 1000
            detection_ok = True
            message = "Detector initialised/corrected tracker"
        except (RuntimeError, ValueError, TypeError, cv2.error) as exc:
            detect_ms = (time.perf_counter() - start) * 1000
            message = f"Detector failed: {str(exc)[:110]}"
    if result is None and tracker.initialised:
        start_update = time.perf_counter()
        try:
            result = tracker.update(frame)
            tracker_ms = (time.perf_counter() - start_update) * 1000
        except (RuntimeError, ValueError, TypeError, cv2.error) as exc:
            message = f"Tracker failed: {str(exc)[:110]}"
    return result, detection_ok, message, detect_ms, tracker_ms


def solve_existing_pose(tracker_result, calibration, pipe, a):
    pose = pipe.pose_result_class(reason="Tracker unavailable or LOST")
    gate = None
    gate_ms = float("nan")
    if tracker_result is None or not bool(tracker_result.tracking_ok):
        return pose, gate, gate_ms

    # v8 relative-mode behaviour: let the solver compute its raw candidate,
    # then apply the existing local-square gate. No modification to PnP geometry.
    pixel_limit = a.max_corner_error if a.pose_gate == "pixels" else 1_000_000.0
    pose = pipe.get_pose(tracker_result, calibration.K, calibration.D,
                         square_size=a.square_size, max_corner_error_px=pixel_limit)
    if pose.solved:
        start = time.perf_counter()
        try:
            gate = pipe.evaluate_gate(
                tracker_result.homography, pose.corner_errors_px,
                mode=a.pose_gate, max_corner_error_squares=a.max_corner_error_squares,
                max_corner_error_px=a.max_corner_error,
            )
            if pose.success:
                pose.success = bool(gate.accepted)
                pose.reason = gate.reason
        except (RuntimeError, ValueError, TypeError, cv2.error) as exc:
            pose.success = False
            pose.reason = f"Pose gate invalid: {str(exc)[:110]}"
        gate_ms = (time.perf_counter() - start) * 1000
        if pose.camera_position_board is not None:
            zz = float(pose.camera_position_board[2])
            if not np.isfinite(zz) or zz <= 0:
                pose.success = False
                pose.reason = f"Front-face guard failed: camera board Z={zz:.3f} m"
    return pose, gate, gate_ms


def manual_reprojection(pose, points, K, D, square_size):
    if not (pose.solved and pose.rvec is not None and points):
        return 0, float("nan"), float("nan"), float("nan")
    locations = sorted(points)
    obj = np.array([[col * square_size, (BOARD_N-row) * square_size, 0.]
                    for row, col in locations], dtype=np.float64)
    measured = np.array([points[key] for key in locations], dtype=np.float64)
    estimated = cv2.projectPoints(obj, pose.rvec, pose.tvec, K, D)[0].reshape(-1, 2)
    dist = np.linalg.norm(estimated - measured, axis=1)
    # Perspective-aware pitch at each intersection (one square on each axis).
    o_x = obj.copy()
    o_y = obj.copy()
    o_x[:, 0] += square_size
    o_y[:, 1] += square_size
    e_x = cv2.projectPoints(o_x, pose.rvec, pose.tvec, K, D)[0].reshape(-1, 2)
    e_y = cv2.projectPoints(o_y, pose.rvec, pose.tvec, K, D)[0].reshape(-1, 2)
    pitch = np.sqrt(np.linalg.norm(e_x - estimated, axis=1)
                    * np.linalg.norm(e_y - estimated, axis=1))
    valid = pitch > 1e-8
    return (len(locations), float(np.sqrt(np.mean(dist ** 2))),
            float(np.median(dist)),
            float(np.sqrt(np.mean((dist[valid] / pitch[valid]) ** 2)))
            if valid.any() else float("nan"))


def make_row(key, kind, condition, frame_idx, t, frame, result, detector_run, detector_ok,
             detector_message, pose, gate, times, manual=None, calibration=None,
             square_size=SQUARE_M, previous_pose=None):
    tracking_ok = bool(result is not None and result.tracking_ok)
    cp = getattr(pose, "camera_position_board", None)
    if cp is not None:
        cp = np.asarray(cp).reshape(3)
    n, mr, med, mn = manual_reprojection(pose, manual,
                                         calibration.K, calibration.D, square_size) if manual is not None else (0, float("nan"), float("nan"), float("nan"))
    pos_delta = (float(np.linalg.norm(cp-np.asarray(previous_pose.camera_position_board).reshape(3)))
                 if previous_pose is not None and cp is not None and
                 previous_pose.camera_position_board is not None else float("nan"))
    rot_delta = rotation_difference_deg(pose.R, previous_pose.R) if previous_pose is not None else float("nan")
    return {
        "source": key, "source_type": kind, "condition": condition, "frame": frame_idx,
        "time_s": t, "image_width": frame.shape[1], "image_height": frame.shape[0],
        "tracking_ok": int(tracking_ok),
        "tracked_count": int(result.tracked_count) if result is not None else 0,
        "orientation_status": getattr(result, "orientation_status", "UNAVAILABLE"),
        "detector_run": int(detector_run), "detector_ok": int(detector_ok),
        "detector_message": detector_message,
        "pose_status": pose.status, "pose_reason": pose.reason,
        "pnp_solved": int(pose.solved), "pose_accepted": int(pose.success),
        "corner_rmse_px": pose.corner_rmse_px,
        "max_corner_error_px": pose.max_corner_error_px,
        "inner_rmse_px": pose.inner_reprojection_rmse_px,
        "max_error_squares": gate.max_error_squares if gate is not None else float("nan"),
        "manual_points": n, "manual_rmse_px": mr, "manual_median_px": med,
        "manual_rmse_squares": mn,
        "camera_x_board_m": cp[0] if cp is not None else float("nan"),
        "camera_y_board_m": cp[1] if cp is not None else float("nan"),
        "camera_z_board_m": cp[2] if cp is not None else float("nan"),
        "rotation_step_deg": rot_delta, "camera_step_m": pos_delta,
        "detect_ms": times.get("detect_ms", float("nan")),
        "tracker_ms": times.get("tracker_ms", float("nan")),
        "pnp_ms": pose.pnp_ms, "pose_ms": pose.pose_ms,
        "gate_ms": times.get("gate_ms", float("nan")),
        "processing_ms": times.get("processing_ms", float("nan")),
    }


def render_diagnostic(frame, result, pose, title, manual=None, max_width=1150):
    """Export-only overlay. It never affects image processing or evaluation."""
    out = frame.copy()
    if result is not None and getattr(result, "board_corners", None) is not None:
        pts = np.asarray(result.board_corners)
        if pts.shape == (4, 2) and np.isfinite(pts).all():
            cv2.polylines(out, [np.round(pts).astype(np.int32).reshape(-1,1,2)], True,
                          (0, 210, 255), 3, cv2.LINE_AA)
            for i, (x, y) in enumerate(pts):
                cv2.circle(out, (int(round(x)), int(round(y))), 5,
                           (0, 210, 255), -1, cv2.LINE_AA)
                cv2.putText(out, str(i), (int(x)+8, int(y)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 210, 255), 2, cv2.LINE_AA)
    if pose.solved and pose.projected_outer_corners is not None:
        pts = np.asarray(pose.projected_outer_corners)
        if np.isfinite(pts).all():
            cv2.polylines(out, [np.round(pts).astype(np.int32).reshape(-1,1,2)], True,
                          (255, 0, 255), 2, cv2.LINE_AA)
    if pose.success and pose.projected_axes is not None:
        axes = np.asarray(pose.projected_axes)
        if axes.shape == (4, 2) and np.isfinite(axes).all():
            pix = np.round(axes).astype(np.int32)
            o = tuple(pix[0])
            for j, color in ((1, (40, 40, 255)), (2, (40, 255, 40)), (3, (255, 100, 30))):
                cv2.line(out, o, tuple(pix[j]), color, 4, cv2.LINE_AA)
    if manual:
        for (row, col), (x, y) in manual.items():
            cv2.drawMarker(out, (int(round(x)), int(round(y))), (20, 255, 255),
                           cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
    # Header bar drawn AFTER geometry; labels large enough in saved image.
    cv2.rectangle(out, (0, 0), (out.shape[1], 76), (22, 24, 28), -1)
    cv2.putText(out, title[:105], (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .7,
                (245, 245, 245), 2, cv2.LINE_AA)
    status_color = (85, 220, 90) if pose.success else (70, 90, 255)
    err = pose.corner_rmse_px
    cv2.putText(out, f"{pose.status} | fitted corner RMS: {err:.2f}px | magenta=PnP amber=tracker",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, .61, status_color, 2, cv2.LINE_AA)
    if max_width and out.shape[1] > max_width:
        scale = max_width/out.shape[1]
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out


def save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"Could not write {path}")


def process_photos(a, pipe, calib, manual_labels):
    rows = []
    inputs = files_under(a.data_dir / "images", IMAGES_EXT)
    if not inputs:
        print("No photographs found under pose_benchmark/images/clear or /occluded")
    for i, path in enumerate(inputs, start=1):
        key = relative_key(path, a.data_dir)
        condition = "occluded" if "occluded" in path.parts else "clear" if "clear" in path.parts else "unlabelled"
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"SKIP unreadable photo {key}")
            continue
        check_image_geometry(frame, calib, key)
        start = time.perf_counter()
        tracker = pipe.tracker_class()  # A still has no preceding tracking history.
        result, det_ok, msg, detect_ms, tracker_ms = detect_or_track(frame, tracker, pipe, True)
        pose, gate, gate_ms = solve_existing_pose(result, calib, pipe, a)
        processing_ms = (time.perf_counter() - start)*1000
        times = dict(detect_ms=detect_ms, tracker_ms=tracker_ms,
                     gate_ms=gate_ms, processing_ms=processing_ms)
        manual = manual_labels.get(key, {})
        row = make_row(key, "image", condition, 1, 0, frame, result,
                       True, det_ok, msg, pose, gate, times, manual, calib, a.square_size)
        rows.append(row)
        if not a.no_figures:
            target = (a.data_dir / "figures/images" /
                      path.relative_to(a.data_dir / "images")).with_suffix(".jpg")
            save_image(target, render_diagnostic(frame, result, pose, f"{key} | {condition}", manual))
        print(f"[photo {i}/{len(inputs)}] {key}: {pose.status}, "
              f"fit={pose.corner_rmse_px:.2f}px, manual_n={row['manual_points']}, "
              f"manual_rmse={row['manual_rmse_px']:.2f}px")
    write_csv(a.data_dir / "results/image_metrics.csv", rows, COLUMNS)
    return rows


def process_videos(a, pipe, calib, intervals):
    all_rows = []
    inputs = files_under(a.data_dir / "videos", VIDEO_EXT)
    if not inputs:
        print("No video files found under pose_benchmark/videos")
    for vi, path in enumerate(inputs, start=1):
        key = relative_key(path, a.data_dir)
        matches = intervals.get(key, [])
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"SKIP cannot open {key}")
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            print(f"WARNING: {key} has no valid FPS metadata; timestamps and recovery delay unavailable.")
            fps = float("nan")
        tracker = pipe.tracker_class()
        previous_pose = None
        previous_status = None
        previous_phase = None
        frame_idx = 0
        saved = 0
        video_rows = []
        print(f"\n[video {vi}/{len(inputs)}] {key} | FPS={fps} | labelled occlusions={matches}")
        try:
            while True:
                start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx == 1:
                    check_image_geometry(frame, calib, key)
                t = (frame_idx-1)/fps if np.isfinite(fps) else float("nan")
                phase = video_phase(t, matches)
                run_detector = (not tracker.initialised or frame_idx % a.detect_every == 0
                                or not tracker.last_tracking_ok)
                result, det_ok, msg, detect_ms, tracker_ms = detect_or_track(
                    frame, tracker, pipe, run_detector)
                pose, gate, gate_ms = solve_existing_pose(result, calib, pipe, a)
                processing_ms = (time.perf_counter()-start)*1000
                times = dict(detect_ms=detect_ms, tracker_ms=tracker_ms,
                             gate_ms=gate_ms, processing_ms=processing_ms)
                row = make_row(key, "video", phase, frame_idx, t, frame, result,
                               run_detector, det_ok, msg, pose, gate, times,
                               None, calib, a.square_size, previous_pose)
                video_rows.append(row)

                # Only consecutive, accepted poses contribute motion *steps*.
                # These are NOT jitter estimates for a moving/orbiting camera.
                previous_pose = pose if pose.success else None
                wanted = (frame_idx == 1 or
                          (a.sample_every and frame_idx % a.sample_every == 0) or
                          (previous_status is not None and pose.status != previous_status) or
                          (previous_phase is not None and phase != previous_phase))
                if not a.no_figures and wanted and saved < a.max_samples:
                    name = f"{path.stem}__frame{frame_idx:06d}.jpg"
                    # Use path parts in a subdirectory so duplicate stems do not collide.
                    out = a.data_dir / "figures/videos" / path.relative_to(a.data_dir/"videos").with_suffix("") / name
                    save_image(out, render_diagnostic(frame, result, pose,
                             f"{key} | t={t:.2f}s | {phase}"))
                    saved += 1
                previous_status, previous_phase = pose.status, phase
                if frame_idx == 1 or frame_idx % 120 == 0:
                    print(f"  frame {frame_idx:5} {pose.status:16} {phase:9} "
                          f"fit={pose.corner_rmse_px:.2f}px total={processing_ms:.1f}ms")
                if a.max_frames and frame_idx >= a.max_frames:
                    break
        finally:
            cap.release()
        all_rows.extend(video_rows)
        print(f"  Finished {frame_idx} frames; saved {saved} diagnostic frames")
    write_csv(a.data_dir / "results/video_frames.csv", all_rows, COLUMNS)
    return all_rows


def interval_recoveries(rows, intervals):
    """Elapsed time from labelled end of occlusion to next accepted pose."""
    out = []
    for filename, ranges in intervals.items():
        rr = [r for r in rows if r["source"] == filename]
        if not rr:
            continue
        for i, (start, end) in enumerate(ranges, 1):
            after = [r for r in rr if r["time_s"] >= end]
            next_valid = next((r for r in after if r["pose_accepted"]), None)
            # If the clip ends before the labelled end, recovery isn't observable.
            if not after:
                status = "not_observed"
                delay = float("nan")
            elif next_valid is None:
                status = "no_recovery_in_recording"
                delay = float("nan")
            else:
                status = "recovered"
                delay = max(0., float(next_valid["time_s"]-end))
            out.append({"video": filename, "interval": i, "occlusion_start_s": start,
                        "occlusion_end_s": end, "recovery_delay_s": delay,
                        "status": status, "first_valid_frame": next_valid["frame"] if next_valid else ""})
    return out


def summary_rows(image_rows, video_rows, recovery):
    out = []
    def add_group(name, kind, condition, rr):
        if not rr:
            return
        n = len(rr)
        tracked = sum(r["tracking_ok"] for r in rr)
        accepted = sum(r["pose_accepted"] for r in rr)
        labelled = [r for r in rr if r["manual_points"] > 0 and np.isfinite(r["manual_rmse_px"])]
        total_ms = sum(float(r["processing_ms"]) for r in rr)
        out.append({
            "source": name, "source_type": kind, "condition": condition, "frames": n,
            "tracking_ok_pct": 100.*tracked/n, "pose_accepted_pct": 100.*accepted/n,
            "pnp_solved_pct": 100.*sum(r["pnp_solved"] for r in rr)/n,
            "labelled_photos": len(labelled),
            "manual_rmse_median_px": median(r["manual_rmse_px"] for r in labelled),
            "manual_rmse_median_squares": median(r["manual_rmse_squares"] for r in labelled),
            "fit_corner_rmse_median_px": median(r["corner_rmse_px"] for r in rr),
            "max_error_squares_median": median(r["max_error_squares"] for r in rr),
            "detect_ms_median": median(r["detect_ms"] for r in rr),
            "tracker_ms_median": median(r["tracker_ms"] for r in rr),
            "pnp_ms_median": median(r["pnp_ms"] for r in rr if r["pnp_solved"]),
            "processing_ms_median": median(r["processing_ms"] for r in rr),
            "processing_ms_p95": p95(r["processing_ms"] for r in rr),
            # No waitKey/playback, includes image decoding and detector as scheduled.
            "offline_processing_fps": 1000.*n/total_ms if total_ms > 0 else float("nan"),
            "recovery_median_s": median(
                v["recovery_delay_s"] for v in recovery
                if v["video"] == name and v["status"] == "recovered"
            ) if kind == "video" and condition == "all" else float("nan"),
        })

    for condition in ("clear", "occluded", "unlabelled"):
        add_group(f"PHOTOS: {condition}", "image", condition,
                  [r for r in image_rows if r["condition"] == condition])
    for name in sorted({r["source"] for r in video_rows}):
        rr = [r for r in video_rows if r["source"] == name]
        add_group(name, "video", "all", rr)
        if any(r["condition"] != "unlabelled" for r in rr):
            for condition in ("clear", "occluded"):
                add_group(name, "video", condition,
                          [r for r in rr if r["condition"] == condition])
    return out


def plot_figures(a, image_rows, video_rows, intervals):
    if a.no_figures:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One concise photograph figure. Withheld human labels take precedence;
    # otherwise plainly label the chart as in-sample PnP fit, not accuracy.
    if image_rows:
        labelled = [r for r in image_rows if r["manual_points"] and np.isfinite(r["manual_rmse_px"])]
        metric = "manual_rmse_px" if labelled else "corner_rmse_px"
        usable = [r for r in (labelled if labelled else image_rows)
                  if np.isfinite(r[metric])]
        if usable:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            for j, condition in enumerate(("clear", "occluded", "unlabelled")):
                rr = [r for r in usable if r["condition"] == condition]
                if rr:
                    ax.scatter([j]*len(rr), [r[metric] for r in rr], alpha=.85, s=35,
                               label=f"{condition} (n={len(rr)})")
            ax.set_xticks([0, 1, 2], ["Clear", "Occluded", "Unlabelled"])
            ax.set_ylabel("Reprojection RMSE (pixels)")
            ax.set_title("Independent manually labelled grid points" if labelled else
                         "PnP fit to tracker-derived corners (not independent)")
            ax.grid(axis="y", alpha=.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(a.data_dir / "figures/photo_reprojection.png", dpi=170)
            plt.close(fig)

    for key in sorted({r["source"] for r in video_rows}):
        rr = [r for r in video_rows if r["source"] == key]
        t = np.array([r["time_s"] for r in rr], dtype=float)
        if not np.isfinite(t).any():
            t = np.arange(len(rr), dtype=float)
            xlab = "Frame index"
        else:
            xlab = "Time in video (seconds)"
        val = np.array([2 if r["pose_accepted"] else 1 if r["pnp_solved"] else 0 for r in rr])
        fig, ax = plt.subplots(figsize=(9, 3.1))
        ax.step(t, val, where="post", lw=1.25)
        for start, end in intervals.get(key, []):
            ax.axvspan(start, end, facecolor="grey", alpha=.2)
        ax.set_yticks([0, 1, 2], ["Unavailable", "Rejected", "Accepted"])
        ax.set_ylim(-.25, 2.3)
        ax.set_xlabel(xlab)
        ax.set_title(f"Video pose status: {key}")
        ax.grid(axis="x", alpha=.25)
        fig.tight_layout()
        out = a.data_dir / "figures" / (key.replace("/", "__") + "__timeline.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)


def synthetic_study(a, pipe, calib):
    """PnP-only known-pose experiment; deliberately bypasses detector and LK."""
    rng = np.random.default_rng(a.seed)
    rows = []
    obj = pipe.object_outer_corners(a.square_size)
    grid_corners = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    noise_levels = (0., .5, 1., 2., 4.)
    for angle_deg in (0, 20, 40):
        for noise in noise_levels:
            for trial in range(a.synthetic_trials):
                # R0 flips the board y axis to the camera's down-axis while the
                # camera sits on the +Z side of the physical board.
                a0 = np.deg2rad(angle_deg)
                R_tilt, _ = cv2.Rodrigues(np.array([0., a0, 0.], dtype=np.float64))
                R0 = np.diag([1., -1., -1.])
                Rtrue = R_tilt @ R0
                true_rvec = cv2.Rodrigues(Rtrue)[0]
                # Centre projected board in image while varying distance.
                depth = float((.70, 1.00, 1.40)[trial % 3])
                true_tvec = np.array([[-.225], [.225], [depth]], dtype=float)
                ideal = cv2.projectPoints(obj, true_rvec, true_tvec, calib.K, calib.D)[0].reshape(4,2)
                noisy = ideal + rng.normal(0, noise, size=(4,2))
                H = cv2.getPerspectiveTransform(grid_corners, noisy.astype(np.float32))
                fake_result = SimpleNamespace(
                    tracking_ok=True, board_corners=noisy, homography=H,
                    inner=np.full((7,7,2), np.nan, dtype=float), tracked_count=0)
                pose, gate, _ = solve_existing_pose(fake_result, calib, pipe, a)
                actual_C = (-Rtrue.T @ true_tvec).reshape(3)
                if pose.solved and pose.camera_position_board is not None:
                    angular = rotation_difference_deg(pose.R, Rtrue)
                    pos_mm = float(1000*np.linalg.norm(pose.camera_position_board - actual_C))
                    trans_mm = float(1000*np.linalg.norm(pose.tvec - true_tvec))
                else:
                    angular = pos_mm = trans_mm = float("nan")
                rows.append({"view_tilt_deg": angle_deg, "noise_std_px": noise,
                             "depth_m": depth, "trial": trial, "solved": int(pose.solved),
                             "accepted": int(pose.success),
                             "rotation_error_deg": angular,
                             "camera_position_error_mm": pos_mm,
                             "translation_error_mm": trans_mm,
                             "corner_fit_rmse_px": pose.corner_rmse_px,
                             "max_error_squares": gate.max_error_squares if gate is not None else float("nan")})
    cols = list(rows[0])
    write_csv(a.data_dir / "results/synthetic_pose.csv", rows, cols)
    aggregate = []
    for tilt in (0, 20, 40):
        for noise in noise_levels:
            rr = [r for r in rows if r["view_tilt_deg"] == tilt and r["noise_std_px"] == noise]
            aggregate.append({"view_tilt_deg": tilt, "noise_std_px": noise,
                              "rotation_median_deg": median(r["rotation_error_deg"] for r in rr),
                              "camera_position_median_mm": median(r["camera_position_error_mm"] for r in rr),
                              "pose_accepted_pct": 100*np.mean([r["accepted"] for r in rr]),
                              "solved_pct": 100*np.mean([r["solved"] for r in rr])})
    write_csv(a.data_dir / "results/synthetic_summary.csv", aggregate, list(aggregate[0]))
    if not a.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.8, 4.3))
        for tilt in (0, 20, 40):
            rr = [r for r in aggregate if r["view_tilt_deg"] == tilt]
            ax.plot([r["noise_std_px"] for r in rr],
                    [r["rotation_median_deg"] for r in rr], marker="o", label=f"Tilt {tilt}°")
        ax.set_xlabel("Injected corner noise, standard deviation (pixels)")
        ax.set_ylabel("Median rotation error (degrees)")
        ax.set_title("Synthetic known-pose test (PnP only, not detection/tracking)")
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(a.data_dir / "figures/synthetic_rotation_error.png", dpi=170)
        plt.close(fig)
    return rows


def annotate(a):
    """Click selected true intersections on RAW photograph, not estimated overlay.

    grid_row 1 is one row below corner 0; grid_col 1 is one column right
    of corner 0. Rows increase toward corner 3; columns toward corner 1.
    Use a previous diagnostic image to establish tracker corner numbering,
    but click actual physical intersections on the original photograph.
    """
    if a.image is None:
        raise ValueError("--mode annotate requires --image pose_benchmark/images/clear/file.jpg")
    image_path = a.image if a.image.is_absolute() else BASE_DIR / a.image
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    key = relative_key(image_path, a.data_dir)
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot read image {image_path}")
    points = []
    for item in a.grid_points.split(","):
        if not item.strip():
            continue
        try:
            row, col = (int(q) for q in item.strip().split(":"))
        except ValueError as exc:
            raise ValueError("--grid-points format: 1:1,1:4,4:4,...") from exc
        if not (1 <= row <= 7 and 1 <= col <= 7):
            raise ValueError("Grid row/col must be in 1..7")
        if (row, col) in points:
            raise ValueError(f"Duplicate grid index: {item}")
        points.append((row, col))
    if not points:
        raise ValueError("Provide at least one grid point")
    # Downscale only the annotation DISPLAY, not the source pixel coordinates.
    h, w = frame.shape[:2]
    scale = min(1., 1200/max(w, 1), 850/max(h, 1))
    click_records = []
    window = "Manual intersections | left click=add, u=undo, s=skip, q=finish"
    def show_frame():
        shown = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
        for row, col, x, y in click_records:
            xy = (int(round(x*scale)), int(round(y*scale)))
            cv2.circle(shown, xy, 4, (0, 255, 255), -1)
            cv2.putText(shown, f"{row},{col}", (xy[0]+7, xy[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
        idx = len(click_records) + skipped[0]
        instruction = (f"Click row {points[idx][0]}, col {points[idx][1]} "
                       if idx < len(points) else "Finished all requested points")
        cv2.rectangle(shown, (0,0), (shown.shape[1],64), (20, 20, 20), -1)
        cv2.putText(shown, instruction[:70], (10,25), cv2.FONT_HERSHEY_SIMPLEX,
                    .62, (255,255,255), 2)
        cv2.putText(shown, "row 1 near corner 0, col 1 near edge 0->3. s skip, u undo, q save",
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, .48, (230,230,230), 1)
        cv2.imshow(window, shown)

    # Keep click/skip history together so undo properly restores point indices.
    actions = []
    skipped = [0]
    def redraw():
        click_records.clear()
        skipped[0] = 0
        for action in actions:
            idx = len(click_records)+skipped[0]
            if action[0] == "skip":
                skipped[0] += 1
            elif action[0] == "click":
                row, col = points[idx]
                click_records.append((row, col, action[1], action[2]))
        show_frame()
    def mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or len(actions) >= len(points):
            return
        actions.append(("click", float(x)/scale, float(y)/scale))
        redraw()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    try:
        redraw()
        while True:
            keycode = cv2.waitKey(40) & 0xFF
            if keycode in (ord("q"), 13, 10, 27):
                break
            if keycode == ord("s") and len(actions) < len(points):
                actions.append(("skip",))
                redraw()
            if keycode in (ord("u"), 8) and actions:
                actions.pop()
                redraw()
            if len(actions) == len(points) and keycode != 255:
                break
    finally:
        cv2.destroyWindow(window)
    if not click_records:
        print("No annotations collected; nothing saved")
        return
    out_path = a.data_dir / "annotations/image_points.csv"
    existing = read_csv(out_path)
    # Replace only the particular image + grid point entries, preserving labels
    # for other images and manually entered points at different grid positions.
    changed = {(r, c) for r, c, x, y in click_records}
    kept = [r for r in existing if not (r.get("image", "").replace("\\", "/") == key
             and (int(r["grid_row"]), int(r["grid_col"])) in changed)]
    kept.extend({"image": key, "grid_row": row, "grid_col": col, "x_px": f"{x:.2f}",
                 "y_px": f"{y:.2f}"} for row, col, x, y in click_records)
    write_csv(out_path, kept, ["image", "grid_row", "grid_col", "x_px", "y_px"])
    print(f"Saved {len(click_records)} withheld manual points for {key}")


def main(argv=None):
    a = cli(argv)
    folders(a)
    if a.mode == "annotate":
        annotate(a)
        return
    pipe = load_pipeline()
    if not a.calibration.is_absolute():
        a.calibration = BASE_DIR / a.calibration
    if not a.calibration.is_file():
        raise FileNotFoundError(f"Calibration file not found: {a.calibration}")
    calibration = pipe.load_calibration(a.calibration)
    print(f"Calibration: {a.calibration}\nSquare pitch: {a.square_size:.3f} m")
    if calibration.image_size is None:
        print("NOTE: calibration lacks recorded resolution. Webcam/crop/resolution must match; not verifiable from pkl.")
    print(f"Pose gate: {a.pose_gate}, relative limit={a.max_corner_error_squares} squares, "
          f"pixel limit={a.max_corner_error}px")

    if a.mode == "synthetic":
        synthetic_study(a, pipe, calibration)
        return
    photos = []
    videos = []
    if a.mode in ("all", "images"):
        manual = load_manual_points(a.data_dir)
        photos = process_photos(a, pipe, calibration, manual)
    if a.mode in ("all", "videos"):
        intervals = load_occlusion_intervals(a.data_dir)
        videos = process_videos(a, pipe, calibration, intervals)
    else:
        intervals = {}
    recovery = interval_recoveries(videos, intervals)
    recovery_cols = ["video", "interval", "occlusion_start_s", "occlusion_end_s",
                     "recovery_delay_s", "status", "first_valid_frame"]
    if a.mode in ("all", "videos"):
        write_csv(a.data_dir / "results/occlusion_recovery.csv", recovery, recovery_cols)
    summaries = summary_rows(photos, videos, recovery)
    if summaries:
        write_csv(a.data_dir / "results/summary.csv", summaries, list(summaries[0]))
    else:
        print("No inputs processed; put images/videos in the indicated pose_benchmark folders.")
    plot_figures(a, photos, videos, intervals)
    print("\nComplete. Report independent manual image RMSE separately from fitted tracker-corner RMSE.")
    print("Video pose changes during camera motion are not stationary jitter.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
