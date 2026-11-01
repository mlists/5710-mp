"""EXPERIMENTAL calibrated chessboard raw pose fits, without tracker changes.

Keep production get_pose_v6.py unchanged. This experiment evaluates two
additional pose methods against its 4-outer-corner PnP baseline:

* ``inner_pnp``: cv2.solvePnP on the indexed, finite 7x7 tracked intersections.
* ``undistorted_h``: undistort the SAME indexed inner intersections with K,D,
  fit a plane-to-normalized-ray homography with RANSAC, recover a rigid pose by
  orthogonalization (SVD). No PnP refinement and NO temporal smoothing.

Object frame is identical for ALL methods, in metres:
  corner0=(0,8s,0); corner1=(8s,8s,0); corner2=(8s,0,0);
  corner3=(0,0,0), the fixed pose origin; s=0.057 metres.

IMPORTANT: tracker.inner may contain LK-tracked points initially seeded from
homography projections, not independently measured corners. Neither candidate
proves absolute 3D accuracy or resolves the planar front/back ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from get_pose_v6 import (
    PoseResult, SQUARE_SIZE_M, object_outer_corners,
    object_inner_intersections, validate_intrinsics,
)


@dataclass
class ExperimentPose(PoseResult):
    homography_inliers: int = 0
    homography_residual_rmse_px: float = float('nan')
    homography_col_cos: float = float('nan')
    homography_col_norm_ratio: float = float('nan')


def _validate_tracker(result, K, D, square_size, max_corner_error_px):
    if result is None or not bool(result.tracking_ok):
        raise ValueError('Tracker unavailable/LOST for current frame')
    K, D = validate_intrinsics(K, D)
    corners = np.asarray(result.board_corners, np.float64)
    inner = np.asarray(result.inner, np.float64)
    H_tracker = np.asarray(result.homography, np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError('Tracker board_corners are invalid')
    if inner.shape != (7, 7, 2):
        raise ValueError('Tracker inner shape must be (7,7,2)')
    if H_tracker.shape != (3, 3) or not np.isfinite(H_tracker).all():
        raise ValueError('Tracker homography is invalid')
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError('Square size must be positive and finite')
    if not np.isfinite(max_corner_error_px) or max_corner_error_px <= 0:
        raise ValueError('max_corner_error_px must be positive and finite')
    mask = np.isfinite(inner).all(axis=2).reshape(-1)
    return K, D, corners, inner.reshape(-1, 2)[mask], object_inner_intersections(square_size)[mask], mask


def _finish_pose(out, *, reason, valid=False, start):
    out.reason = reason
    out.success = bool(valid)
    out.pose_ms = (time.perf_counter() - start) * 1000
    return out


def _assess_pose(out, tracker_result, K, D, square_size, corners,
                 inner_obj, inner_uv, rvec, tvec, max_corner_error_px,
                 axis_length_squares, start, method_reason):
    """Identical diagnostics and acceptance test for both experimental fits."""
    projection_start = time.perf_counter()
    rvec = np.asarray(rvec, np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    outer = object_outer_corners(square_size)
    camera_points = (outer @ R.T + tvec.reshape(3))
    if (not np.isfinite(R).all() or not np.isfinite(tvec).all()
            or not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 1e-8)):
        return _finish_pose(out, reason='Nonfinite or behind-camera solution', start=start)
    out.solved = True
    out.R = R
    out.rvec = rvec
    out.tvec = tvec
    out.board_normal_camera = R[:, 2].copy()
    out.camera_position_board = -R.T @ tvec.reshape(3)
    out.T_board_to_camera = np.eye(4, dtype=np.float64)
    out.T_board_to_camera[:3, :3] = R
    out.T_board_to_camera[:3, 3] = tvec.ravel()

    out.projected_outer_corners = cv2.projectPoints(outer, rvec, tvec, K, D)[0].reshape(4, 2)
    out.projected_origin = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, D)[0].reshape(2)
    out.corner_errors_px = np.linalg.norm(out.projected_outer_corners - corners, axis=1)
    out.max_corner_error_px = float(np.max(out.corner_errors_px))
    out.corner_rmse_px = float(np.sqrt(np.mean(out.corner_errors_px ** 2)))
    out.origin_error_px = float(np.linalg.norm(out.projected_origin - corners[3]))
    projected_inner = cv2.projectPoints(inner_obj, rvec, tvec, K, D)[0].reshape(-1, 2)
    out.inner_reprojection_rmse_px = float(np.sqrt(np.mean(
        np.sum((projected_inner - inner_uv) ** 2, axis=1))))
    axis_length = axis_length_squares * square_size
    axes_obj = np.array([[0,0,0], [axis_length,0,0], [0,axis_length,0],
                         [0,0,axis_length]], np.float64)
    out.projected_axes = cv2.projectPoints(axes_obj, rvec, tvec, K, D)[0].reshape(4, 2)
    out.projection_ms = (time.perf_counter() - projection_start) * 1000

    if not np.all(np.isfinite(out.projected_axes)) or not np.all(np.isfinite(out.corner_errors_px)):
        return _finish_pose(out, reason='Invalid projected geometry', start=start)
    if not np.allclose(out.projected_origin, out.projected_outer_corners[3], atol=1e-7):
        return _finish_pose(out, reason='Origin is not identical to tracker corner-3 object coordinate', start=start)
    if out.max_corner_error_px > max_corner_error_px:
        return _finish_pose(out, reason=f'Max tracker-boundary error {out.max_corner_error_px:.2f}px > {max_corner_error_px:.2f}px', start=start)
    return _finish_pose(out, valid=True, reason=method_reason, start=start)


def get_inner_pnp_pose(tracker_result, K, D, *, square_size=SQUARE_SIZE_M,
                       max_corner_error_px=14.0, axis_length_squares=3.0):
    """RAW 7x7 indexed-intersection PnP, a diagnostic comparison, not a replacement."""
    start = time.perf_counter()
    out = ExperimentPose(method='v7 INNER-PNP (indexed tracker.inner; RAW)')
    try:
        begin = time.perf_counter()
        K, D, corners, inner_uv, inner_obj, _ = _validate_tracker(
            tracker_result, K, D, square_size, max_corner_error_px)
        out.num_correspondences = out.num_tracked_inner = len(inner_uv)
        if len(inner_uv) < 6:
            return _finish_pose(out, reason='Fewer than 6 tracked intersections', start=start)
        out.setup_ms = (time.perf_counter() - begin)*1000
        begin = time.perf_counter()
        ok, rvec, tvec = cv2.solvePnP(inner_obj, inner_uv, K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        out.pnp_ms = (time.perf_counter()-begin)*1000
        if not ok:
            return _finish_pose(out, reason='OpenCV solvePnP failed', start=start)
        return _assess_pose(out, tracker_result, K, D, square_size, corners, inner_obj, inner_uv,
                            rvec, tvec, max_corner_error_px, axis_length_squares, start,
                            'Inner-grid RAW PnP passes boundary error gate')
    except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError, cv2.error) as exc:
        return _finish_pose(out, reason=f'Inner PnP: {str(exc)[:150]}', start=start)


def get_undistorted_h_pose(tracker_result, K, D, *, square_size=SQUARE_SIZE_M,
                            max_corner_error_px=14.0, axis_length_squares=3.0,
                            ransac_threshold_px=3.0, min_inliers=8, refine_pnp=False):
    """RAW calibrated pose from UNDISTORTED tracked intersections + planar H.

    Notes:
      * cv2.undistortPoints returns NORMALIZED ray coordinates (not pixels).
      * plane-to-ray H estimates [r1,r2,t] up to scale; SVD enforces SO(3).
      * The H fit has NO temporal smoothing and NO iterative PnP refinement.
      * A nonlinear distortion-aware projection of the rigid result is used
        for scoring against the ORIGINAL, DISTORTED tracker pixels.
    """
    start = time.perf_counter()
    out = ExperimentPose(method='v7 UNDISTORTED-H (normalized-ray planar homography; RAW)')
    try:
        begin = time.perf_counter()
        K, D, corners, inner_uv, inner_obj, _ = _validate_tracker(
            tracker_result, K, D, square_size, max_corner_error_px)
        out.num_correspondences = out.num_tracked_inner = len(inner_uv)
        if len(inner_uv) < max(8, min_inliers):
            return _finish_pose(out, reason=f'Fewer than {max(8, min_inliers)} tracked intersections', start=start)
        if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0:
            return _finish_pose(out, reason='Invalid normalized-homography threshold', start=start)
        und = cv2.undistortPoints(inner_uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        if not np.isfinite(und).all():
            return _finish_pose(out, reason='Invalid undistorted intersections', start=start)
        # Coordinates in metres: X,Y of the board plane, where XY is fixed
        # for the entire video and the origin is exactly tracker corner 3.
        XY = inner_obj[:, :2].copy()
        out.setup_ms = (time.perf_counter() - begin)*1000
        begin = time.perf_counter()
        h, inliers = cv2.findHomography(
            XY, und, cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px / ((K[0, 0] + K[1, 1])*0.5),
        )
        if h is None or inliers is None:
            return _finish_pose(out, reason='Undistorted normalized homography failed', start=start)
        keep = inliers.ravel().astype(bool)
        out.homography_inliers = int(keep.sum())
        if out.homography_inliers < min_inliers:
            return _finish_pose(out, reason=f'Only {out.homography_inliers} homography inliers', start=start)
        # Refit precisely on RANSAC inliers. No additional point detector.
        h, _ = cv2.findHomography(XY[keep], und[keep], 0)
        if h is None or not np.isfinite(h).all():
            return _finish_pose(out, reason='Invalid normalized homography refit', start=start)
        h1, h2, h3 = [h[:, i].copy() for i in range(3)]
        n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
        if min(n1, n2) < 1e-12:
            return _finish_pose(out, reason='Degenerate normalized homography columns', start=start)
        out.homography_col_cos = float(np.dot(h1,h2)/(n1*n2))
        out.homography_col_norm_ratio = float(n1/n2)
        # Positive t_z is required for a board in front of the camera.
        scale = (n1+n2)*0.5
        sign = 1.0 if h3[2] >= 0 else -1.0
        raw1, raw2 = sign*h1/scale, sign*h2/scale
        raw3 = np.cross(raw1,raw2)
        R_rough = np.column_stack([raw1,raw2,raw3])
        u, _, vt = np.linalg.svd(R_rough)
        reflect = np.diag([1,1,np.linalg.det(u@vt)])
        R = u @ reflect @ vt
        tvec = (sign*h3/scale).reshape(3,1)
        rvec = cv2.Rodrigues(R)[0]
        out.pnp_ms = (time.perf_counter() - begin)*1000  # setup H/SVD, NOT PnP
        predicted_und = cv2.perspectiveTransform(XY[keep].reshape(-1,1,2), h).reshape(-1,2)
        ray_err = np.linalg.norm(predicted_und-und[keep],axis=1)
        out.homography_residual_rmse_px = float(np.sqrt(np.mean(ray_err**2)) *
                                                 ((K[0,0]+K[1,1])*0.5))
        if refine_pnp:
            # The raw H is an INITIALIZER, not a distortion-aware rigid fit.
            # Refine ONLY over homography RANSAC inliers. This optimizes a
            # physically constrained perspective camera with the original K,D.
            # We report errors on ALL current tracked points, including those
            # withheld from this refinement, to prevent masking poor geometry.
            out.method = 'v7 UNDISTORTED-H + INLIER-PNP (raw per-frame, NO smoothing)'
            begin_refinement = time.perf_counter()
            ok, rvec, tvec = cv2.solvePnP(
                inner_obj[keep], inner_uv[keep], K, D, rvec, tvec,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            out.pnp_ms += (time.perf_counter()-begin_refinement)*1000
            if not ok:
                return _finish_pose(out, reason='Homography-seeded distortion-aware PnP refinement failed', start=start)
            good_reason = 'Undistorted-H initialized + distortion-aware inlier PnP passes gate'
        else:
            good_reason = 'Undistorted H -> rigid pose passes boundary error gate'
        return _assess_pose(out, tracker_result, K, D, square_size, corners, inner_obj, inner_uv,
                            rvec, tvec, max_corner_error_px, axis_length_squares, start,
                            good_reason)
    except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError, cv2.error) as exc:
        return _finish_pose(out, reason=f'Undistorted H: {str(exc)[:150]}', start=start)


def get_undistorted_h_refined_pose(tracker_result, K, D, **kwargs):
    """RANSAC undistorted H initialization followed by raw inlier PnP refinement."""
    return get_undistorted_h_pose(tracker_result, K, D, refine_pnp=True, **kwargs)
