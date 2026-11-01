"""Resolution- and perspective-aware pose acceptance for an unchanged raw PnP solve.

A per-corner reprojection error is divided by the pixel pitch of a single
chessboard square *at that corner*, projected through TrackerResult.homography.
The local pitch is the geometric mean of the two adjacent one-square edges.

This module changes no tracker associations, camera calibration, PnP
correspondences, R/t, or coordinate frames. It only assesses a solved pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# All coordinates use the tracker's existing 8x8 canonical grid.
_CORNER_GRID = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float64)
_ONE_X_INWARD = np.array([[1, 0], [7, 0], [7, 8], [1, 8]], dtype=np.float64)
_ONE_Y_INWARD = np.array([[0, 1], [8, 1], [8, 7], [0, 7]], dtype=np.float64)
_LOCAL_GRID = np.concatenate((_CORNER_GRID, _ONE_X_INWARD, _ONE_Y_INWARD))


@dataclass
class CornerGate:
    accepted: bool
    mode: str
    threshold: float
    reason: str
    square_pitch_px: np.ndarray  # 4 local pixel pitches, one for each tracker corner
    errors_squares: np.ndarray  # error at each corner in fractions of a square
    allowed_errors_px: np.ndarray  # 4 pose thresholds in source-image pixels
    max_error_squares: float


def local_square_pitches(homography) -> np.ndarray:
    """Use the existing homography to measure each corner's 1-square pixel scale.

    Distances are measured along both incident grid directions, then combined
    with a geometric mean. Using local samples respects strong perspective.
    No new homography is fitted and no new image features are detected.
    """
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Tracker homography must be a finite 3x3 matrix")
    projected = cv2.perspectiveTransform(
        _LOCAL_GRID.reshape(-1, 1, 2), H
    ).reshape(3, 4, 2)
    pitch_x = np.linalg.norm(projected[1] - projected[0], axis=1)
    pitch_y = np.linalg.norm(projected[2] - projected[0], axis=1)
    pitch = np.sqrt(pitch_x * pitch_y)
    if not np.isfinite(pitch).all() or np.any(pitch < 2.0):
        raise ValueError("Projected local square pitch is invalid or <2 pixels")
    return pitch


def evaluate_corner_gate(
    homography,
    corner_errors_px,
    *,
    mode: str = "relative",
    max_corner_error_squares: float = 0.25,
    max_corner_error_px: float = 14.0,
) -> CornerGate:
    """Evaluate a PnP result, leaving the pose mathematics entirely unchanged.

    mode='relative': accept iff all corner errors <= a specified FRACTION of
    their respective local square pitch. mode='pixels' reproduces the v6 gate.
    """
    if mode not in ("relative", "pixels"):
        raise ValueError("mode must be 'relative' or 'pixels'")
    if not np.isfinite(max_corner_error_squares) or max_corner_error_squares <= 0:
        raise ValueError("max_corner_error_squares must be positive and finite")
    if not np.isfinite(max_corner_error_px) or max_corner_error_px <= 0:
        raise ValueError("max_corner_error_px must be positive and finite")
    errors = np.asarray(corner_errors_px, dtype=np.float64).reshape(-1)
    if errors.shape != (4,) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Four nonnegative, finite corner reprojection errors are required")
    pitch = local_square_pitches(homography)
    fractions = errors / pitch
    if mode == "relative":
        allowed = pitch * max_corner_error_squares
        accepted = bool(np.all(fractions <= max_corner_error_squares))
        threshold = float(max_corner_error_squares)
        rule = f"{threshold:.3f} squares"
    else:
        allowed = np.full(4, max_corner_error_px, dtype=np.float64)
        accepted = bool(np.all(errors <= max_corner_error_px))
        threshold = float(max_corner_error_px)
        rule = f"{threshold:.2f} px"
    reason = (
        f"{mode} gate {'PASS' if accepted else 'FAIL'}: "
        f"max error {float(np.max(errors)):.2f}px, "
        f"max local fraction {float(np.max(fractions)):.3f} square; "
        f"limit {rule}"
    )
    return CornerGate(
        accepted=accepted, mode=mode, threshold=threshold, reason=reason,
        square_pitch_px=pitch, errors_squares=fractions,
        allowed_errors_px=allowed, max_error_squares=float(np.max(fractions)),
    )
