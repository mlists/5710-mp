import argparse
import time

import cv2
import find_board
import get_pose
from board_tracker import BoardTracker


def parse_source(source_arg):
    """Interpret an integer source as a camera index; otherwise use a video path."""
    try:
        return int(source_arg)
    except ValueError:
        return source_arg


def main():
    parser = argparse.ArgumentParser(
        description="Track a chessboard and estimate its pose from a camera or video."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or video filename (e.g. input.mp4). Default: 0.",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=60,
        help="Re-run the expensive board detector every N frames. Default: 60.",
    )
    parser.add_argument(
        "--debug-detection",
        action="store_true",
        help="Show the detector's debug output when it is run.",
    )
    parser.add_argument(
        "--tracker-debug",
        action="store_true",
        help="Overlay the tracker's numbered and labeled board corners.",
    )
    args = parser.parse_args()

    if args.detect_every < 1:
        parser.error("--detect-every must be at least 1.")

    source = parse_source(args.source)
    is_video_file = isinstance(source, str)

    K, D = get_pose.load_calibration()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    tracker = BoardTracker()
    frame_count = 0
    previous_frame_time = None
    smoothed_fps = 0.0
    fps_smoothing = 0.1

    # Use the video's frame rate for approximately real-time playback.
    if is_video_file:
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = max(1, round(1000.0 / fps)) if fps and fps > 0 else 1
    else:
        wait_ms = 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1

            # Measure and smooth the actual processing/display rate.
            current_frame_time = time.perf_counter()
            if previous_frame_time is not None:
                frame_delta = current_frame_time - previous_frame_time

                if frame_delta > 0:
                    instantaneous_fps = 1.0 / frame_delta

                    if smoothed_fps == 0.0:
                        smoothed_fps = instantaneous_fps
                    else:
                        smoothed_fps = (
                            fps_smoothing * instantaneous_fps
                            + (1.0 - fps_smoothing) * smoothed_fps
                        )

            display = frame.copy()
            result = None
            detector_message = ""

            # Detect on the first frame, periodically, or whenever tracking
            # failed on the previous frame.
            run_detector = (
                not tracker.initialised
                or frame_count % args.detect_every == 0
                or not tracker.last_tracking_ok
            )

            if run_detector:
                try:
                    detection = find_board.detect_chessboard(
                        frame,
                        video_mode=True,
                        display_output=True,
                        display_debug=args.debug_detection,
                    )

                    if tracker.initialised:
                        result = tracker.update(frame, detection=detection)
                    else:
                        result = tracker.initialize(frame, detection)

                    detector_message = "Detector correction"

                except (RuntimeError, ValueError, cv2.error) as exc:
                    detector_message = f"Detection failed: {str(exc)[:45]}"

            # If no fresh detection was supplied, update from optical flow.
            if result is None and tracker.initialised:
                try:
                    result = tracker.update(frame)

                except (RuntimeError, ValueError, cv2.error) as exc:
                    detector_message = f"Tracking error: {str(exc)[:45]}"

            if result is not None and result.tracking_ok:
                try:
                    pose = get_pose.get_board_pose(
                        result.board_corners,
                        result.square_centres,
                        observed_image_points=result.inner,
                        camera_matrix=K,
                        dist_coeffs=D,
                    )

                    display = get_pose.draw_pose_overlay(frame, pose)

                    label = (
                        f"Tracked: {result.tracked_count} "
                        f"Corners: {pose['num_observations']} "
                        f"Inliers: {pose['num_inliers']} "
                        f"RMSE: {pose['reprojection_rmse']:.1f}px"
                    )

                    cv2.putText(
                        display, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                    )

                    if detector_message:
                        cv2.putText(
                            display, detector_message, (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2,
                        )

                except (RuntimeError, ValueError, cv2.error, KeyError) as exc:
                    cv2.putText(
                        display, f"Pose unavailable: {str(exc)[:55]}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2,
                    )

            else:
                message = (
                    detector_message
                    or "Board unavailable - waiting for detection"
                )

                cv2.putText(
                    display, message[:65], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            # Display FPS in the top-right corner.
            fps_label = (
                f"FPS: {smoothed_fps:.1f}"
                if smoothed_fps > 0
                else "FPS: --"
            )

            (text_width, text_height), _ = cv2.getTextSize(
                fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )

            fps_x = max(20, display.shape[1] - text_width - 250)

            cv2.putText(
                display, fps_label, (fps_x, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                cv2.LINE_AA,
            )

            if args.tracker_debug and result is not None:
                display = tracker.draw_debug(display, result)

            cv2.imshow("Chessboard pose", display)

            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

            previous_frame_time = current_frame_time

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()