"""
The current detector returns 64 centres and 4 boundary corners projected from
its fitted homography

Coordinate convention (right-handed):
    origin: outer bottom-left chessboard corner
    +X:     towards increasing columns
    +Y:     towards decreasing rows (towards the first detector row)
    +Z:     out of the camera-visible playing surface (upright pieces)


All object coordinates are in units of ``square_size`` (default 1.0)

The resulting R and tvec map BOARD coordinates into CAMERA coordinates

This module estimates a pose independently for each frame. 
"""

from pathlib import Path
import pickle

import cv2
import numpy as np


BOARD_SIZE = 8
INTERNAL_SIZE = BOARD_SIZE - 1


def load_calibration(calibration_path=None):
    if calibration_path is None:
        calibration_path = Path(__file__).resolve().with_name("cal_data.pkl")
    path = Path(calibration_path)

    with path.open("rb") as f:
        calibration = pickle.load(f)

    if not isinstance(calibration, dict) or not {"mtx", "dist"}.issubset(calibration):
        raise ValueError("Calibration file needs 'mtx' and 'dist' entries.")

    return check_calibration(calibration["mtx"], calibration["dist"])


def check_calibration(camera_matrix, dist_coeffs):
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(dist_coeffs, dtype=np.float64)

    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("camera_matrix must be a finite 3x3 array.")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive.")
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    if not np.isfinite(D).all() or D.size not in (4, 5, 8, 12, 14):
        raise ValueError("Invalid OpenCV lens distortion coefficients.")
    return K, D


def generate_board_geometry(square_size=1.0):
    """Return physical grid points, square centres, and outer board corners.

    returns:
        internal_grid: (7,7,3), ordered by detected [row, col]
        centres:      (8,8,3), ordered by [row, col]
        outer:        (4,3), in detector's TL, TR, BR, BL order
    """
    s = float(square_size)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("square_size must be positive and finite.")

    internal_grid = np.array(
        [[((c + 1) * s, (BOARD_SIZE - 1 - r) * s, 0.0)
          for c in range(INTERNAL_SIZE)] for r in range(INTERNAL_SIZE)],
        dtype=np.float64,
    )
    centres = np.array(
        [[((c + 0.5) * s, (BOARD_SIZE - r - 0.5) * s, 0.0)
          for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)],
        dtype=np.float64,
    )
    edge = BOARD_SIZE * s
    outer = np.array(
        [[0, edge, 0], [edge, edge, 0], [edge, 0, 0], [0, 0, 0]],
        dtype=np.float64,
    )
    return internal_grid, centres, outer


def prepare_direct_points(observed_grid_points, observed_image_points, square_size):
    """Convert measured grid observations to corresponding 3D object points."""
    pix = np.asarray(observed_image_points, dtype=np.float64)

    if pix.shape == (INTERNAL_SIZE, INTERNAL_SIZE, 2):
        if observed_grid_points is not None:
            raise ValueError("Do not provide grid IDs for a complete 7x7 array.")
        grd = np.array(
            [(c + 1, r + 1)
             for r in range(INTERNAL_SIZE) for c in range(INTERNAL_SIZE)],
            dtype=np.float64,
        )
        pix = pix.reshape(-1, 2)
    elif pix.ndim == 2 and pix.shape[1] == 2:
        if observed_grid_points is None:
            if len(pix) != INTERNAL_SIZE * INTERNAL_SIZE:
                raise ValueError("A partial (N,2) image array needs grid IDs (N,2).")
            grd = np.array(
                [(c + 1, r + 1)
                 for r in range(INTERNAL_SIZE) for c in range(INTERNAL_SIZE)],
                dtype=np.float64,
            )
        else:
            grd = np.asarray(observed_grid_points, dtype=np.float64)
            if grd.shape != pix.shape:
                raise ValueError("observed_grid_points must match image points (N,2).")
    else:
        raise ValueError("observed_image_points must be (N,2) or (7,7,2).")

    good = np.isfinite(grd).all(axis=1) & np.isfinite(pix).all(axis=1)
    grd, pix = grd[good], pix[good]
    if len(grd) < 4:
        raise ValueError("At least four labelled non-collinear observations are needed.")
    if np.any(grd < 0) or np.any(grd > BOARD_SIZE):
        raise ValueError("Grid coordinates must lie between 0 and 8.")
    if not np.allclose(grd, np.round(grd), atol=1e-6):
        raise ValueError("Grid coordinates must be integer intersection indices.")
    if len(np.unique(grd, axis=0)) != len(grd):
        raise ValueError("Each detected grid intersection must have a unique grid ID.")

    s = float(square_size)
    obj = np.column_stack((grd[:, 0] * s,
                           (BOARD_SIZE - grd[:, 1]) * s,
                           np.zeros(len(grd)))).astype(np.float64)
    check_spread(obj)
    return np.ascontiguousarray(obj), np.ascontiguousarray(pix), np.ascontiguousarray(grd)


def check_spread(obj):
    if len(obj) < 4:
        raise ValueError("At least four point correspondences are required.")
    centred = obj[:, :2] - obj[:, :2].mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if len(sv) < 2 or sv[0] <= 1e-12 or sv[1] / sv[0] < 1e-4:
        raise ValueError("Grid observations are collinear or nearly collinear.")


def orient_board_face_towards_camera(R, tvec, object_points, internal_board, centres_board, corners_board, square_size):
    """Give the chessboard a camera-facing +Z axis without changing its pixels.

    A symmetric planar grid can be indexed with either handedness, especially
    by the partial-occlusion fallback. Both can have near-zero reprojection
    error, but one makes +Z point through the board instead of above it.

    We assume the camera sees the PLAYING FACE of the board. When the measured
    indexing gives a back-facing +Z axis, change to an equivalent, right-handed
    board frame by rotating the frame 180 degrees around its X axis and
    moving its origin to the opposite edge.

    For the new board coordinates (x', y', z'):
        (x, y, z)_old = (x', 8*s - y', -z')

    This is a proper rotation, not a reflection of R. The projected locations
    of the SAME detector-indexed points remain unchanged, and chess pieces
    extending in positive Z now stand on the camera-facing side.
    """
    n = R[:, 2]
    side = float(-np.dot(n, np.asarray(tvec).reshape(3)))
    if side >= 0:
        return (R, tvec, object_points, internal_board, centres_board,
                corners_board, False, "bottom_left_y_up")

    edge = float(BOARD_SIZE * square_size)
    Q = np.diag([1.0, -1.0, -1.0])
    offset_old = np.array([[0.0], [edge], [0.0]], dtype=np.float64)

    # A proper rigid change of board coordinates: det(Q) == +1.
    R_corrected = R @ Q
    t_corrected = tvec + R @ offset_old

    # Re-express all ORIGINAL detector-indexed points in the corrected frame.
    # This preserves the meaning of every square_centres[row, col].
    def convert(points):
        points = np.asarray(points, dtype=np.float64)
        converted = points.copy()
        converted[..., 1] = edge - points[..., 1]
        converted[..., 2] = -points[..., 2]
        return converted

    return (R_corrected, t_corrected, convert(object_points),
            convert(internal_board), convert(centres_board),
            convert(corners_board), True, "top_left_y_down")


def get_board_pose(
    board_corners,
    square_centres=None,
    *,
    square_size=1.0,
    calibration_path=None,
    camera_matrix=None,
    dist_coeffs=None,
    observed_grid_points=None,
    observed_image_points=None,
    ransac_threshold_px=3.0,
    ransac_iterations=200,
    enforce_visible_face=True,
):
    """Estimate pose from updated find_board.py outputs.

    Usual usage:
        centres_px, corners_px, inner_px = find_board.detect_chessboard(frame)
        pose = get_board_pose(corners_px, centres_px,
                              observed_image_points=inner_px,
                              camera_matrix=K, dist_coeffs=D)

    Args:
        board_corners: (4,2) projected chessboard outline in TL, TR, BR, BL.
        square_centres: (8,8,2) homography-derived centres, or None to use
            just the four corners. A legacy (7,7,2) array of genuinely detected
            internal corners is also supported for backwards compatibility.
        square_size: a physical square's width in chosen world units.
        calibration_path: trusted file path for mtx/dist, default cal_data.pkl
            beside this script. Not used when K and D are supplied.
        camera_matrix, dist_coeffs: calibrated OpenCV camera parameters.
        observed_grid_points: optional (N,2) integer (x,y) grid coordinates
            from 0 to 8, identifying DIRECT detections.
        observed_image_points: optional (N,2) image pixels corresponding to
            observed_grid_points, or full (7,7,2) internal array (NaNs OK).
        ransac_threshold_px: pixel threshold after lens undistortion, used ONLY
            for directly observed grid points.
        ransac_iterations: RANSAC iterations for direct observations.
        enforce_visible_face: Assumes the camera views the playing side of
            the board and corrects mirrored corner indexing so +Z points
            out of the board towards the camera. Default True.

    Returns:
        Dictionary containing rotation R, rvec, tvec, square-centre geometry,
        board corners, reprojection diagnostics, and provenance of input points.

    Measured 7x7 arrays may contain NaNs where corners are occluded. They are
    excluded automatically. At least four geometrically distributed, labelled
    points are required to estimate a fresh pose.

    CAUTION: When only square_centres/board_corners are available, reprojection
    error is a fit-to-homography diagnostic, NOT an independent corner accuracy
    or occlusion robustness metric.
    """
    if camera_matrix is None and dist_coeffs is None:
        K, D = load_calibration(calibration_path)
    elif camera_matrix is None or dist_coeffs is None:
        raise ValueError("Provide both camera_matrix and dist_coeffs, or neither.")
    else:
        K, D = check_calibration(camera_matrix, dist_coeffs)

    if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0:
        raise ValueError("ransac_threshold_px must be positive.")
    if ransac_iterations < 1:
        raise ValueError("ransac_iterations must be at least one.")

    internal_board, centres_board, corners_board = generate_board_geometry(square_size)
    corners_px = None
    if board_corners is not None:
        corners_px = np.asarray(board_corners, dtype=np.float64)
        if corners_px.shape != (4, 2) or not np.isfinite(corners_px).all():
            raise ValueError("board_corners must be a finite (4,2) array or None.")

    direct = observed_image_points is not None
    if not direct and observed_grid_points is not None:
        raise ValueError("observed_grid_points needs observed_image_points.")

    centres_px = None
    if square_centres is not None:
        centres_px = np.asarray(square_centres, dtype=np.float64)
        if centres_px.shape == (7, 7, 2):
            if direct:
                raise ValueError("Provide either legacy inner points or direct observations.")
            direct = True
            observed_image_points = centres_px
            centres_px = None
        elif centres_px.shape != (8, 8, 2) or not np.isfinite(centres_px).all():
            raise ValueError("square_centres must be (8,8,2) or legacy (7,7,2).")

    if direct:
        obj, img, grid_indices = prepare_direct_points(
            observed_grid_points, observed_image_points, square_size
        )
        input_source = "detected_grid_intersections"
        # Reject incorrect correspondences using a planar homography fitted to
        # undistorted pixels. PnP itself then uses original pixels and D.
        if len(obj) >= 6:
            undistorted_px = cv2.undistortPoints(
                img.reshape(-1, 1, 2), K, D, P=K
            ).reshape(-1, 2)
            H, mask = cv2.findHomography(
                obj[:, :2], undistorted_px, cv2.RANSAC,
                float(ransac_threshold_px),
                maxIters=int(ransac_iterations), confidence=0.99,
            )
            if H is None or mask is None:
                raise RuntimeError("RANSAC homography could not be fitted.")
            inlier_idx = np.flatnonzero(mask.ravel().astype(bool))
            if len(inlier_idx) < 4:
                raise RuntimeError("Too few consistent measured points after RANSAC.")
            method = "planar homography RANSAC + iterative PnP"
        else:
            inlier_idx = np.arange(len(obj))
            method = "iterative PnP (too few points for robust screening)"
        observed_count = len(obj)
    elif centres_px is not None:
        obj = centres_board.reshape(-1, 3)
        img = centres_px.reshape(-1, 2)
        grid_indices = None
        inlier_idx = np.arange(len(obj))
        input_source = "homography_derived_square_centres"
        method = "iterative PnP on homography-derived centres"
        observed_count = 0
    elif corners_px is not None:
        obj = corners_board
        img = corners_px
        grid_indices = None
        inlier_idx = np.arange(4)
        input_source = "homography_derived_outer_corners"
        method = "iterative PnP on homography-derived board corners"
        observed_count = 0
    else:
        raise ValueError("Supply board corners, square centres or measured grid points.")

    obj = np.ascontiguousarray(obj, dtype=np.float64)
    img = np.ascontiguousarray(img, dtype=np.float64)
    check_spread(obj)
    good_obj = np.ascontiguousarray(obj[inlier_idx], dtype=np.float64)
    good_img = np.ascontiguousarray(img[inlier_idx], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        good_obj, good_img, K, D, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("OpenCV solvePnP failed to estimate board pose.")
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(good_obj, good_img, K, D, rvec, tvec)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)

    # Reprojection error cannot distinguish a mirrored grid assignment from
    # a genuine camera-facing board. Choose the playable side explicitly.
    face_corrected = False
    board_frame_convention = "bottom_left_y_up"
    if enforce_visible_face:
        (R, tvec, obj, internal_board, centres_board, corners_board,
         face_corrected, board_frame_convention) = orient_board_face_towards_camera(
             R, tvec, obj, internal_board, centres_board, corners_board, square_size
         )
        if face_corrected:
            rvec, _ = cv2.Rodrigues(R)

    # This is positive when the +Z axis faces the camera from the board.
    # A DOWNWARD SCREEN arrow can still be physically correct if the camera
    # is looking obliquely at the playing surface.
    signed_face_distance = float(-np.dot(R[:, 2], tvec[:, 0]))
    board_centre_cam = (R @ np.array([4.0 * square_size,
                                      4.0 * square_size, 0.0]) + tvec[:, 0])
    camera_facing_cosine = signed_face_distance / max(
        float(np.linalg.norm(board_centre_cam)), 1e-12
    )

    board_xyz_in_camera = (R @ corners_board.T + tvec).T
    if not np.isfinite(board_xyz_in_camera).all() or np.any(board_xyz_in_camera[:, 2] <= 0):
        raise RuntimeError("Estimated chessboard is behind or crosses the camera plane.")

    reprojected_input, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    reprojected_input = reprojected_input.reshape(-1, 2)
    errors_px = np.linalg.norm(img - reprojected_input, axis=1)
    rmse = float(np.sqrt(np.mean(errors_px[inlier_idx] ** 2)))
    rmse_all = float(np.sqrt(np.mean(errors_px ** 2)))

    predicted_corners, _ = cv2.projectPoints(corners_board, rvec, tvec, K, D)
    predicted_centres, _ = cv2.projectPoints(
        centres_board.reshape(-1, 3), rvec, tvec, K, D
    )
    predicted_internal, _ = cv2.projectPoints(
        internal_board.reshape(-1, 3), rvec, tvec, K, D
    )
    centres_camera = (R @ centres_board.reshape(-1, 3).T + tvec).T.reshape(8, 8, 3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec[:, 0]

    projected_board_corners = predicted_corners.reshape(4, 2)
    outer_corner_diff_px = None
    if corners_px is not None:
        outer_corner_diff_px = float(np.mean(np.linalg.norm(
            corners_px - projected_board_corners, axis=1
        )))

    inlier_mask = np.zeros(len(obj), dtype=bool)
    inlier_mask[inlier_idx] = True
    return {
        "success": True,
        "pose_status": "MEASURED" if direct else "HOMOGRAPHY_DERIVED",
        "method": method,
        "input_source": input_source,
        "R": R,
        "rvec": rvec,
        "tvec": tvec,
        "board_normal": R[:, 2].copy(),
        "face_corrected": face_corrected,
        "board_frame_convention": board_frame_convention,
        "camera_facing_cosine": float(camera_facing_cosine),
        "visible_face_enforced": bool(enforce_visible_face),
        "T_board_to_camera": T,
        "T_camera_to_board": np.linalg.inv(T),
        "square_centres_board": centres_board,
        "square_centres_camera": centres_camera,
        "square_centres_pixel": predicted_centres.reshape(8, 8, 2),
        "board_corners_board": corners_board,
        "board_corners_px": projected_board_corners,
        "grid_corners_board": internal_board,
        "grid_corners_px": predicted_internal.reshape(7, 7, 2),
        "detector_board_corners_px": corners_px,
        "detector_square_centres_px": centres_px,
        "fit_object_points": obj,
        "fit_image_points": img,
        "fit_grid_indices": grid_indices,
        "projected_fit_points": reprojected_input,
        "reprojection_errors_px": errors_px,
        "reprojection_rmse": rmse,
        "reprojection_rmse_all": rmse_all,
        # In-sample reprojection is not independent ground truth, even when
        # it is calculated against directly measured image points.
        "reprojection_error_is_independent": False,
        "reprojection_uses_measured_points": direct,
        "num_observations": observed_count,
        "num_fit_points": len(obj),
        "num_inliers": len(inlier_idx),
        "inlier_mask": inlier_mask,
        "outer_corner_difference_px": outer_corner_diff_px,
        "camera_matrix": K,
        "dist_coeffs": D,
        "square_size": float(square_size),
    }


def draw_pose_overlay(image, pose, axis_length=2.0):
    """Draw projected board geometry, optional inliers, and XYZ axes on BGR image.

    Colours (BGR): turquoise outline, green centre dots, red measured inliers,
    standard OpenCV axes red X, green Y, blue Z.
    """
    overlay = np.asarray(image).copy()
    if overlay.ndim != 3 or overlay.shape[2] != 3:
        raise ValueError("Expected BGR image of shape (height,width,3).")

    polygon = np.round(pose["board_corners_px"]).astype(np.int32)
    cv2.polylines(overlay, [polygon], True, (255, 180, 0), 2)
    for point in pose["square_centres_pixel"].reshape(-1, 2):
        cv2.circle(overlay, tuple(np.round(point).astype(int)), 3, (0, 210, 60), -1)

    if pose["input_source"] == "detected_grid_intersections":
        for point in pose["fit_image_points"][pose["inlier_mask"]]:
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 3, (0, 0, 255), -1)

    cv2.drawFrameAxes(
        overlay, pose["camera_matrix"], pose["dist_coeffs"],
        pose["rvec"], pose["tvec"],
        float(axis_length) * float(pose["square_size"]), 2,
    )
    return overlay
