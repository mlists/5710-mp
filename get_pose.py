"""Raw calibrated chessboard pose from the unchanged BoardTracker result.

The sole board coordinate convention (metres) is:
    tracker corner 0 -> (0, 8*s, 0)
    tracker corner 1 -> (8*s, 8*s, 0)
    tracker corner 2 -> (8*s, 0, 0)
    tracker corner 3 -> (0, 0, 0), the PERMANENT axes origin

+X is toward increasing tracker grid columns; +Y toward decreasing rows;
+Z follows the right-hand rule. No smoothing, detector, tracking or hidden
axis flips are performed in this module. R,t map BOARD coordinates to CAMERA:
    P_camera = R @ P_board + t

For the raw baseline only the four tracker boundary corners are passed to PnP.
The existing 7x7 tracked intersections are diagnostic, NOT fitted to in PnP.
The outer corners originate from a homography, not independent detections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pickle
import time

import cv2
import numpy as np


SQUARE_SIZE_M = 0.057
BOARD_CELLS = 8
METHOD = "SOLVEPNP_ITERATIVE (4 tracker boundary corners; RAW)"


@dataclass
class Calibration:
    K: np.ndarray
    D: np.ndarray
    image_size: Optional[tuple[int, int]] = None  # width, height, if provided
    path: str = ""


def load_calibration(path: str | Path) -> Calibration:
    """Load a camera's K,D from .pkl or .npz; never guess a new calibration.

    Supported pickle keys: ('mtx', 'dist'), ('K','D'), or
    ('camera_matrix','dist_coeffs'). Optional image_size=(width,height).
    Only open pickle files you trust: unpickling is capable of executing code.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            record = {key: data[key] for key in data.files}
    else:
        with path.open("rb") as file:
            record = pickle.load(file)  # Only trusted, locally supplied data.
    if not isinstance(record, dict):
        raise ValueError("Calibration must be a dictionary with camera K and distortion D.")

    def first_present(*names):
        return next((record[n] for n in names if n in record), None)

    raw_k = first_present("mtx", "K", "camera_matrix")
    raw_d = first_present("dist", "D", "dist_coeffs")
    if raw_k is None or raw_d is None:
        raise ValueError("Calibration must contain mtx/dist (or K/D).")
    K, D = validate_intrinsics(raw_k, raw_d)

    raw_size = first_present("image_size", "frame_size", "resolution")
    if raw_size is not None:
        size = tuple(int(v) for v in np.asarray(raw_size).ravel())
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("Calibration image_size must be (width, height).")
    elif "image_width" in record and "image_height" in record:
        size = (int(record["image_width"]), int(record["image_height"]))
    else:
        size = None

    return Calibration(K=K, D=D, image_size=size, path=str(path))


def validate_intrinsics(camera_matrix, dist_coeffs):
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Camera K must be a finite 3x3 matrix.")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("Camera K has invalid focal lengths or bottom row.")
    if D.size not in (4, 5, 8, 12, 14) or not np.isfinite(D).all():
        raise ValueError("D must contain 4, 5, 8, 12, or 14 finite distortion values.")
    return K.copy(), D.copy()


def object_outer_corners(square_size: float = SQUARE_SIZE_M) -> np.ndarray:
    """Four physical corners, exactly matching tracker indices 0,1,2,3."""
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("Square size must be finite and positive (metres).")
    edge = BOARD_CELLS * float(square_size)
    return np.array(
        [[0, edge, 0], [edge, edge, 0], [edge, 0, 0], [0, 0, 0]],
        dtype=np.float64,
    )


def object_inner_intersections(square_size: float = SQUARE_SIZE_M) -> np.ndarray:
    """49 points in the precise row-major indexing of TrackerResult.inner."""
    return np.array(
        [[col * square_size, (BOARD_CELLS - row) * square_size, 0.0]
         for row in range(1, BOARD_CELLS)
         for col in range(1, BOARD_CELLS)], dtype=np.float64,
    )


@dataclass
class PoseResult:
    success: bool = False
    solved: bool = False  # True when PnP produced a candidate, even if rejected
    reason: str = "Pose not attempted"
    method: str = METHOD
    R: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    T_board_to_camera: Optional[np.ndarray] = None
    camera_position_board: Optional[np.ndarray] = None
    board_normal_camera: Optional[np.ndarray] = None
    projected_outer_corners: Optional[np.ndarray] = None  # shape (4,2)
    projected_origin: Optional[np.ndarray] = None         # shape (2,)
    projected_axes: Optional[np.ndarray] = None           # shape (4,2), O,X,Y,Z
    corner_errors_px: np.ndarray = field(default_factory=lambda: np.full(4, np.nan))
    origin_error_px: float = float("nan")
    corner_rmse_px: float = float("nan")
    max_corner_error_px: float = float("nan")
    inner_reprojection_rmse_px: float = float("nan")
    num_correspondences: int = 0
    num_tracked_inner: int = 0
    homography_corner_max_px: float = float("nan")
    pose_ms: float = 0.0
    setup_ms: float = 0.0
    pnp_ms: float = 0.0
    projection_ms: float = 0.0

    @property
    def status(self) -> str:
        return "POSE VALID" if self.success else ("POSE REJECTED" if self.solved else "POSE UNAVAILABLE")


def get_board_pose(
    tracker_result,
    camera_matrix,
    dist_coeffs,
    square_size: float = SQUARE_SIZE_M,
    max_corner_error_px: float = 8.0,
    axis_length_squares: float = 3.0,
) -> PoseResult:
    """Compute one unsmoothed distortion-aware planar PnP pose.

    The default 8 px maximum corner-error gate is a *provisional diagnostic*
    starting value. It must be assessed with actual camera footage/calibration;
    it is not a universal accuracy guarantee. Inner tracked points are only
    evaluated for an additional out-of-sample-of-PnP (not independent-of-tracker)
    reprojection error, and do not affect the pose or this validity gate.
    """
    start = time.perf_counter()
    out = PoseResult()

    def finish(message: str, *, valid: bool = False) -> PoseResult:
        out.reason = message
        out.success = valid
        out.pose_ms = (time.perf_counter() - start) * 1000.0
        return out

    setup_start = time.perf_counter()
    try:
        if not bool(tracker_result.tracking_ok):
            return finish("Tracker LOST; no current-frame pose")
        K, D = validate_intrinsics(camera_matrix, dist_coeffs)
        obj = object_outer_corners(square_size)
        if not np.isfinite(max_corner_error_px) or max_corner_error_px <= 0:
            return finish("max_corner_error_px must be positive")
        if not np.isfinite(axis_length_squares) or axis_length_squares <= 0:
            return finish("axis_length_squares must be positive")
        corners = np.asarray(tracker_result.board_corners, dtype=np.float64)
        H = np.asarray(tracker_result.homography, dtype=np.float64)
        inner = np.asarray(tracker_result.inner, dtype=np.float64)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            return finish("Tracker outer corners are missing or invalid")
        if H.shape != (3, 3) or not np.isfinite(H).all():
            return finish("Tracker homography is missing or invalid")
        if inner.shape != (7, 7, 2):
            return finish("Tracker inner array must be (7,7,2)")
        # This audits, but does not replace or refit, the existing homography.
        boundary_grid = np.array([[0, 0], [8, 0], [8, 8], [0, 8]],
                                 dtype=np.float64).reshape(-1, 1, 2)
        from_h = cv2.perspectiveTransform(boundary_grid, H).reshape(4, 2)
        out.homography_corner_max_px = float(
            np.linalg.norm(from_h - corners, axis=1).max())
        if not np.isfinite(out.homography_corner_max_px):
            return finish("Homography projects invalid corners")
        if out.homography_corner_max_px > 1.0:
            return finish("Tracker homography and boundary differ >1 px")
        # Reject truly degenerate geometry, not high-perspective views.
        area = 0.5 * abs(np.dot(corners[:, 0], np.roll(corners[:, 1], -1))
                         - np.dot(corners[:, 1], np.roll(corners[:, 0], -1)))
        if area <= 1.0:
            return finish("Tracker boundary has degenerate area")
        out.num_correspondences = 4
        mask = np.isfinite(inner).all(axis=2).reshape(-1)
        out.num_tracked_inner = int(mask.sum())
        out.setup_ms = (time.perf_counter() - setup_start) * 1000.0

        pnp_start = time.perf_counter()
        ok, rvec, tvec = cv2.solvePnP(
            obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        out.pnp_ms = (time.perf_counter() - pnp_start) * 1000.0
        if not ok:
            return finish("cv2.solvePnP returned false")
        out.solved = True
        R, _ = cv2.Rodrigues(rvec)
        out.R, out.rvec, out.tvec = R, rvec, tvec
        T = np.eye(4, dtype=np.float64)
        T[:3, :3], T[:3, 3] = R, tvec.reshape(3)
        out.T_board_to_camera = T
        out.camera_position_board = (-R.T @ tvec).reshape(3)
        out.board_normal_camera = (R @ [0.0, 0.0, 1.0]).reshape(3)

        project_start = time.perf_counter()
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        out.projected_outer_corners = projected.reshape(4, 2)
        origin = np.zeros((1, 3), dtype=np.float64)
        projected_origin, _ = cv2.projectPoints(origin, rvec, tvec, K, D)
        out.projected_origin = projected_origin.reshape(2)
        out.corner_errors_px = np.linalg.norm(out.projected_outer_corners - corners, axis=1)
        out.origin_error_px = float(np.linalg.norm(out.projected_origin - corners[3]))
        out.corner_rmse_px = float(np.sqrt(np.mean(out.corner_errors_px ** 2)))
        out.max_corner_error_px = float(out.corner_errors_px.max())

        # Independent origin projection MUST equal projection of physical corner 3.
        if not np.allclose(out.projected_origin, out.projected_outer_corners[3], atol=1e-7):
            return finish("Internal origin projection differs from corner 3")

        length = float(axis_length_squares * square_size)
        axes = np.array([[0, 0, 0], [length, 0, 0],
                         [0, length, 0], [0, 0, length]], dtype=np.float64)
        out.projected_axes = cv2.projectPoints(axes, rvec, tvec, K, D)[0].reshape(4, 2)

        if out.num_tracked_inner:
            projected_inner = cv2.projectPoints(
                object_inner_intersections(square_size)[mask], rvec, tvec, K, D
            )[0].reshape(-1, 2)
            inner_observed = inner.reshape(-1, 2)[mask]
            out.inner_reprojection_rmse_px = float(np.sqrt(np.mean(
                np.sum((projected_inner - inner_observed) ** 2, axis=1))))
        out.projection_ms = (time.perf_counter() - project_start) * 1000.0

        # A pose must project *all* board points in front of the pinhole camera.
        camera_xyz = obj @ R.T + tvec.reshape(1, 3)
        if not np.isfinite(camera_xyz).all() or np.any(camera_xyz[:, 2] <= 1e-8):
            return finish("PnP candidate has a corner at/behind the camera")
        if not np.isfinite(out.projected_axes).all():
            return finish("Projected coordinate axes are non-finite")
        if not np.isfinite(out.corner_errors_px).all():
            return finish("PnP reprojection is invalid")
        if out.max_corner_error_px > max_corner_error_px:
            return finish(
                f"Max corner error {out.max_corner_error_px:.2f}px exceeds "
                f"configured {max_corner_error_px:.2f}px gate"
            )
        return finish("4-corner raw calibrated reprojection passes gate", valid=True)
    except (ValueError, TypeError, AttributeError, cv2.error, np.linalg.LinAlgError) as exc:
        return finish(f"Pose computation failed: {str(exc)[:160]}")
