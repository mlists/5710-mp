"""One-source, two-panel webcam/video runner for RAW get_pose_v6.

The imported detector and tracker are unmodified authoritative originals.
Panels are rendered from the SAME unmodified source frame, after tracking.
No resize, crop, colour correction or overlay is passed to either algorithm.

Examples:
  python find_board_video_v6.py --webcam 0 --calibration cal_data.pkl
  python find_board_video_v6.py --source input.mp4 --calibration cal_data.pkl
  python find_board_video_v6.py --source input.mp4 --no-display --csv result.csv

Run from the directory containing byte-identical find_board.py, board_tracker.py,
get_pose_v6.py, and a calibration matching the original recording.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np

import find_board
from board_tracker import BoardTracker
from get_pose_v6 import PoseResult, get_board_pose, load_calibration


WINDOW = "Chessboard v6 | TRACKING (left) / RAW POSE (right)"
# OpenCV uses BGR.
TRACKER_COLOUR = (0, 220, 255)   # amber
POSE_COLOUR = (255, 0, 255)      # magenta
VALID_COLOUR = (90, 225, 110)
INVALID_COLOUR = (50, 100, 255)
LABEL_COLOURS = ((40, 60, 255), (80, 220, 80), (255, 130, 30), (0, 240, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def source_value(text):
    try:
        return int(text)
    except ValueError:
        return text


def _draw_text(img, text, location, colour=(255, 255, 255), scale=0.65, thickness=1):
    cv2.putText(img, str(text), location, FONT, scale, (20, 20, 20), thickness + 3,
                cv2.LINE_AA)
    cv2.putText(img, str(text), location, FONT, scale, colour, thickness,
                cv2.LINE_AA)


def _valid_corners(result):
    if result is None:
        return None
    corners = np.asarray(result.board_corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    return corners


def _polygon(img, corners, colour, thickness):
    cv2.polylines(img, [np.round(corners).astype(np.int32).reshape(-1, 1, 2)],
                  True, colour, thickness, cv2.LINE_AA)


def _corner_ids(img, corners):
    h, w = img.shape[:2]
    for i, (x, y) in enumerate(corners):
        xy = (int(round(x)), int(round(y)))
        cv2.circle(img, xy, 7, LABEL_COLOURS[i], -1, cv2.LINE_AA)
        label_x = int(np.clip(x + (14 if x < w / 2 else -30), 0, max(0, w - 45)))
        label_y = int(np.clip(y + (20 if y < h / 2 else -15), 20, h - 4))
        _draw_text(img, str(i), (label_x, label_y), LABEL_COLOURS[i], 0.76, 2)


def _with_footer(frame, heading, lines, colour):
    height, width = frame.shape[:2]
    footer_height = 98
    out = np.full((height + footer_height, width, 3), (30, 34, 39), np.uint8)
    out[:height] = frame
    _draw_text(out, heading, (15, height + 27), colour, 0.78, 2)
    _draw_text(out, lines[0], (15, height + 54), (240, 240, 240), 0.57)
    _draw_text(out, lines[1], (15, height + 81), (208, 208, 208), 0.53)
    return out


def render_tracking(frame, result, detector_message=""):
    """Tracking view uses current-frame TrackerResult, not the detector's window."""
    im = frame.copy()
    corners = _valid_corners(result)
    if result is None:
        status = "TRACKER WAITING"
        pts = 0
    else:
        status = "TRACKER OK" if result.tracking_ok else "TRACKER LOST (stale boundary)"
        pts = int(result.tracked_count)
    if corners is not None:
        colour = TRACKER_COLOUR if result.tracking_ok else (120, 120, 120)
        _polygon(im, corners, colour, 2)
        _corner_ids(im, corners)
        if result.tracking_ok:
            # Open circles are homography-PROJECTED centres, not measured corners.
            centres = np.asarray(result.square_centres).reshape(-1, 2)
            for x, y in centres:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(im, (int(round(x)), int(round(y))), 3,
                               (100, 210, 100), 1, cv2.LINE_AA)
            # Solid dots are tracked inner-grid intersections, when retained.
            inner = np.asarray(result.inner).reshape(-1, 2)
            for x, y in inner:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(im, (int(round(x)), int(round(y))), 3,
                               (245, 190, 40), -1, cv2.LINE_AA)
    return _with_footer(
        im, f"TRACKING   {status}",
        (f"Tracked inner: {pts}/49   |   corner IDs: 0,1,2,3   |   origin = 3",
         f"{detector_message[:110]}   |   filled: LK points; rings: projected centres"),
        VALID_COLOUR if result is not None and result.tracking_ok else INVALID_COLOUR,
    )


def _fmt(value, decimals=2):
    try:
        return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def render_pose(frame, result, pose: PoseResult):
    """Show reference/tracked boundary AND independently PnP-projected boundary."""
    im = frame.copy()
    corners = _valid_corners(result)
    if corners is not None and result.tracking_ok:
        _polygon(im, corners, TRACKER_COLOUR, 2)  # tracker reference
        _corner_ids(im, corners)
        # Mark the ACTUAL tracker-defined origin, regardless of pose validity.
        p = tuple(np.round(corners[3]).astype(int))
        cv2.drawMarker(im, p, TRACKER_COLOUR, cv2.MARKER_TILTED_CROSS, 19, 2,
                       cv2.LINE_AA)
        _draw_text(im, "TRK O", (p[0] + 12, p[1] + 31), TRACKER_COLOUR, 0.52)

    if (pose.projected_outer_corners is not None
            and np.isfinite(pose.projected_outer_corners).all()):
        # Retain rejected candidate geometry as diagnostic; draw NO rejected axes.
        projected = pose.projected_outer_corners
        _polygon(im, projected, POSE_COLOUR, 2)
        if corners is not None and result.tracking_ok:
            for i, (src, dst) in enumerate(zip(corners, projected)):
                p = tuple(np.round(src).astype(int))
                q = tuple(np.round(dst).astype(int))
                cv2.line(im, p, q, (245, 245, 245), 1, cv2.LINE_AA)
                if np.isfinite(dst).all():
                    cv2.circle(im, q, 4, POSE_COLOUR, -1, cv2.LINE_AA)
                offset = (11, -12 if i in (0, 1) else 16)
                _draw_text(im, f"{i}:{_fmt(pose.corner_errors_px[i],1)}px",
                           (p[0] + offset[0], p[1] + offset[1]), POSE_COLOUR, 0.46)
        if pose.projected_origin is not None:
            p = tuple(np.round(pose.projected_origin).astype(int))
            cv2.circle(im, p, 10, POSE_COLOUR, 2, cv2.LINE_AA)
            _draw_text(im, "PnP O", (p[0] + 10, p[1] - 18), POSE_COLOUR, 0.53)

    if (pose.success and pose.projected_axes is not None
        and np.isfinite(pose.projected_axes).all()):
        axis = np.round(pose.projected_axes).astype(int)
        root = tuple(axis[0])
        for j, name, colour in ((1, "+X", (30, 30, 255)),
                                (2, "+Y", (45, 240, 45)),
                                (3, "+Z", (255, 80, 35))):
            dest = tuple(axis[j])
            cv2.arrowedLine(im, root, dest, colour, 3, cv2.LINE_AA, tipLength=0.18)
            _draw_text(im, name, (dest[0] + 5, dest[1] - 6), colour, 0.65, 2)

    c = ", ".join(_fmt(v, 1) for v in pose.corner_errors_px)
    heading = f"RAW POSE   {pose.status}"
    # Colour legend is in text, so not dependent on display hue interpretation.
    line1 = (f"C0..C3 [px]: {c}   |   O3: {_fmt(pose.origin_error_px,1)}px"
             f"   |   inner RMSE: {_fmt(pose.inner_reprojection_rmse_px,1)}px")
    line2 = f"Tracker=amber; PnP=magenta   |   {pose.reason[:85]}"
    return _with_footer(im, heading, (line1, line2),
                        VALID_COLOUR if pose.success else INVALID_COLOUR)


def make_two_view(frame, result, pose, detector_message, display_scale):
    left = render_tracking(frame, result, detector_message)
    right = render_pose(frame, result, pose)
    if display_scale != 1.0:
        new_size = (max(1, round(left.shape[1] * display_scale)),
                    max(1, round(left.shape[0] * display_scale)))
        left = cv2.resize(left, new_size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, new_size, interpolation=cv2.INTER_AREA)
    return np.hstack((left, right))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--source", default="0", help="Video filename, or webcam index (default 0)")
    source_group.add_argument("--webcam", type=int, help="Webcam device index, e.g. 0")
    parser.add_argument("--calibration", default="cal_data.pkl", help="Trusted matching .pkl or .npz containing K and D")
    parser.add_argument("--square-size", type=float, default=0.057, help="Physical square pitch in metres (default 0.057)")
    parser.add_argument("--detect-every", type=int, default=60, help="Re-detect every N frames (unchanged default 60)")
    parser.add_argument("--display-scale", type=float, default=0.5, help="Display only, all processing at native pixels")
    parser.add_argument("--max-corner-error", type=float, default=14.0,
                        help="Provisional max pixel error to accept RAW pose; tune only from actual evidence")
    parser.add_argument("--print-every", type=int, default=1, help="Console diagnostics every N frames")
    parser.add_argument("--csv", help="Optional per-frame timings and errors CSV output")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit (useful for reproducible offline tests)")
    parser.add_argument("--no-display", action="store_true", help="Headless mode with console/CSV output")
    parser.add_argument("--save-preview", help="Save a two-panel PNG of first pose-valid frame, else the last frame")
    args = parser.parse_args(argv)
    if args.detect_every < 1 or args.print_every < 1:
        parser.error("--detect-every and --print-every must be >= 1")
    if not (0 < args.display_scale <= 2.0):
        parser.error("--display-scale must be in (0, 2]")
    if args.square_size <= 0 or args.max_corner_error <= 0:
        parser.error("--square-size and --max-corner-error must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames must not be negative")
    return args


CSV_COLUMNS = (
    "frame", "tracking_ok", "tracked_count", "detector_run", "detector_message",
    "pose_status", "reason", "method", "n_pnp_points", "n_inner_tracked",
    "corner0_px", "corner1_px", "corner2_px", "corner3_px", "origin_px",
    "corner_rmse_px", "max_corner_error_px", "inner_rmse_px", "H_consistency_px",
    "capture_ms", "detect_ms", "tracker_ms", "pose_setup_ms", "pnp_ms",
    "pose_projection_ms", "pose_total_ms", "render_ms", "total_ms",
)


def _summarize(records):
    if not records:
        return
    print(f"\nSummary ({len(records)} frames, computation only; excludes waitKey/playback):")
    for key in ("capture_ms", "detect_ms", "tracker_ms", "pose_setup_ms",
                "pnp_ms", "pose_projection_ms", "render_ms", "total_ms"):
        values = np.array([r[key] for r in records if r[key] is not None], dtype=float)
        if len(values):
            print(f"  {key:<22} median {np.median(values):8.2f} ms"
                  f"  p95 {np.percentile(values,95):8.2f} ms (n={len(values)})")
    valid = sum(r["pose_status"] == "POSE VALID" for r in records)
    rejected = sum(r["pose_status"] == "POSE REJECTED" for r in records)
    unavailable = len(records) - valid - rejected
    print(f"  Pose statuses: valid={valid}, rejected={rejected}, unavailable={unavailable}")
    print("  Note: tracker agreement is not independent evidence of metric 3D accuracy.")


def main(argv=None):
    args = parse_args(argv)
    calibration = load_calibration(args.calibration)
    print(f"Calibration: {calibration.path}")
    print(f"K:\n{calibration.K}\nD: {calibration.D.reshape(-1)}")
    if calibration.image_size is None:
        print("WARNING: calibration file does not include image_size; camera/resolution match is UNVERIFIED.")
    source = args.webcam if args.webcam is not None else source_value(args.source)
    is_video_file = isinstance(source, str)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    tracker = BoardTracker()  # unchanged configuration and algorithm
    records = []
    csvfile = None
    writer = None
    last_preview = None
    preview_saved = False
    n = 0
    source_fps = cap.get(cv2.CAP_PROP_FPS) if is_video_file else 0.0
    print(f"Source: {source}; video FPS: {source_fps:.3f}" if is_video_file else f"Webcam: {source}")

    try:
        if args.csv:
            csvfile = open(args.csv, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        while True:
            total_start = time.perf_counter()
            capture_start = total_start
            ok, frame = cap.read()  # the single source frame for BOTH panels
            # if frame_count == 0:
            #     print(f"Capture resolution: {frame.shape[1]} x {frame.shape[0]}")
            capture_ms = (time.perf_counter() - capture_start) * 1000.0
            if not ok:
                break
            n += 1
            width, height = frame.shape[1], frame.shape[0]
            if n == 1:
                print(f"Frame geometry: {width}x{height}; original pixels used for tracking/PnP")
                if calibration.image_size is not None and calibration.image_size != (width, height):
                    raise ValueError(
                        f"Calibration resolution {calibration.image_size} != source {(width,height)}; "
                        "refusing incorrect metric pose (no automatic intrinsics scaling)."
                    )
                # Display immediately, BEFORE potentially slow initial detection.
                if not args.no_display:
                    starter = make_two_view(frame, None, PoseResult(),
                                            "Opening frame; initial detection pending", args.display_scale)
                    cv2.imshow(WINDOW, starter)
                    cv2.waitKey(1)
            detector_message = "Optical flow only"
            result = None
            detect_ms = None
            tracker_ms = None

            run_detector = (not tracker.initialised or n % args.detect_every == 0
                            or not tracker.last_tracking_ok)
            if run_detector:
                detector_start = time.perf_counter()
                try:
                    # Do NOT ask the original detector to open another window.
                    detection = find_board.detect_chessboard(
                        frame, video_mode=False, display_output=False, display_debug=False
                    )
                    detect_ms = (time.perf_counter() - detector_start) * 1000.0
                    update_start = time.perf_counter()
                    result = (tracker.update(frame, detection=detection) if tracker.initialised
                              else tracker.initialize(frame, detection))
                    tracker_ms = (time.perf_counter() - update_start) * 1000.0
                    detector_message = "Detector initialised/corrected tracker"
                except (RuntimeError, ValueError, cv2.error) as exc:
                    detect_ms = (time.perf_counter() - detector_start) * 1000.0
                    detector_message = f"Detector / update failed: {str(exc)[:85]}"
                    print(f"[frame {n}] {detector_message}")

            if result is None and tracker.initialised:
                update_start = time.perf_counter()
                try:
                    result = tracker.update(frame)  # ORIGINAL optical flow
                    tracker_ms = (time.perf_counter() - update_start) * 1000.0
                except (RuntimeError, ValueError, cv2.error) as exc:
                    detector_message = f"Tracker update failed: {str(exc)[:85]}"
                    print(f"[frame {n}] {detector_message}")

            pose = (get_board_pose(result, calibration.K, calibration.D,
                                   square_size=args.square_size,
                                   max_corner_error_px=args.max_corner_error)
                    if result is not None and result.tracking_ok else PoseResult(
                        reason="Tracker not initialised or tracking LOST"))
            render_start = time.perf_counter()
            two_view = make_two_view(frame, result, pose, detector_message, args.display_scale)
            if not args.no_display:
                cv2.imshow(WINDOW, two_view)
            render_ms = (time.perf_counter() - render_start) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0

            record = {
                "frame": n,
                "tracking_ok": bool(result.tracking_ok) if result is not None else False,
                "tracked_count": int(result.tracked_count) if result is not None else 0,
                "detector_run": run_detector,
                "detector_message": detector_message,
                "pose_status": pose.status,
                "reason": pose.reason,
                "method": pose.method,
                "n_pnp_points": pose.num_correspondences,
                "n_inner_tracked": pose.num_tracked_inner,
                "corner0_px": float(pose.corner_errors_px[0]),
                "corner1_px": float(pose.corner_errors_px[1]),
                "corner2_px": float(pose.corner_errors_px[2]),
                "corner3_px": float(pose.corner_errors_px[3]),
                "origin_px": pose.origin_error_px,
                "corner_rmse_px": pose.corner_rmse_px,
                "max_corner_error_px": pose.max_corner_error_px,
                "inner_rmse_px": pose.inner_reprojection_rmse_px,
                "H_consistency_px": pose.homography_corner_max_px,
                "capture_ms": capture_ms,
                "detect_ms": detect_ms,
                "tracker_ms": tracker_ms,
                "pose_setup_ms": pose.setup_ms,
                "pnp_ms": pose.pnp_ms,
                "pose_projection_ms": pose.projection_ms,
                "pose_total_ms": pose.pose_ms,
                "render_ms": render_ms,
                "total_ms": total_ms,
            }
            records.append(record)
            if writer is not None:
                writer.writerow(record)
            if n == 1 or n % args.print_every == 0:
                corners_str = ",".join(_fmt(x, 2) for x in pose.corner_errors_px)
                print(
                    f"[{n:05d}] tracking={'OK' if record['tracking_ok'] else 'LOST'} "
                    f"pts={record['tracked_count']:02d} detection={int(run_detector)} "
                    f"pose={pose.status} "
                    f"c0..3=[{corners_str}] O3={_fmt(pose.origin_error_px)}px "
                    f"inner={_fmt(pose.inner_reprojection_rmse_px)}px "
                    f"ms(capture/detect/track/setup/PnP/project/render/total)="
                    f"{capture_ms:.1f}/{_fmt(detect_ms,1)}/{_fmt(tracker_ms,1)}/"
                    f"{pose.setup_ms:.2f}/{pose.pnp_ms:.2f}/{pose.projection_ms:.2f}/"
                    f"{render_ms:.1f}/{total_ms:.1f}  {pose.reason}"
                )

            if args.save_preview:
                last_preview = two_view  # save failure-state if never valid
                if pose.success and not preview_saved:
                    if not cv2.imwrite(args.save_preview, two_view):
                        raise OSError(f"Could not save preview to {args.save_preview}")
                    preview_saved = True
                    print(f"First valid preview saved: {args.save_preview}")

            # Playback pacing does not contaminate computation timing.
            playback_wait = (max(1, round(1000.0 / source_fps - total_ms))
                             if is_video_file and source_fps > 0 else 1)
            if not args.no_display:
                key = cv2.waitKey(playback_wait) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    tracker.reset()
                    print("Tracker reset; detector will reinitialise on the next frame.")
            if args.max_frames and n >= args.max_frames:
                break
        if args.save_preview and not preview_saved and last_preview is not None:
            if not cv2.imwrite(args.save_preview, last_preview):
                raise OSError(f"Could not save preview to {args.save_preview}")
            print(f"No valid pose; last diagnostic preview saved: {args.save_preview}")
    finally:
        cap.release()
        if csvfile is not None:
            csvfile.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        _summarize(records)


if __name__ == "__main__":
    main()
