import argparse

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

    K, D = get_pose.load_calibration()  # Read calibration once.

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    tracker = BoardTracker()
    frame_count = 0

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

            if args.tracker_debug and result is not None:
                display = tracker.draw_debug(display, result)

            cv2.imshow("Chessboard pose", display)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
