import cv2
import find_board
import get_pose


CAMERA_INDEX = 0


def main():
    K, D = get_pose.load_calibration()  # Read calibration once, outside the loop.
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera index {CAMERA_INDEX}')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            display = frame.copy()
            try:
                centres, corners, inner = find_board.detect_chessboard(frame, video_mode=True, display_output=True, display_debug=True)

                pose = get_pose.get_board_pose(
                    corners, centres,
                    observed_image_points=inner,
                    camera_matrix=K, dist_coeffs=D,
                )
                display = get_pose.draw_pose_overlay(frame, pose)
                label = (
                    f"Corners: {pose['num_observations']} "
                    f"inliers: {pose['num_inliers']} "
                    f"RMSE: {pose['reprojection_rmse']:.1f}px"
                )
                cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
            except (RuntimeError, ValueError) as exc:
                # No temporal pose tracking here: a failed frame has no pose.
                cv2.putText(display, f'Board unavailable: {str(exc)[:50]}',
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

            cv2.imshow('Chessboard pose', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
