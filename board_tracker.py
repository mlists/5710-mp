"""
Optical-flow tracker for the chessboard detector in find_board.py.

Expected detector API:
    square_centres, board_corners, inner = detect_chessboard(frame, ...)

The detector is deliberately not called by this class. Run it only when you
want to initialise the tracker or periodically correct tracking drift.

Coordinate convention:
    The tracker's canonical grid has its top-left corner at (0, 0), and its
    bottom-right corner at (8, 8). The standard chessboard convention is that
    the top-left corner square (A8) is light. Corner-square brightness is used
    to choose between the two rotation classes. A completely empty,
    unmarked board still leaves a 180-degree ambiguity.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


BOARD_SIZE = 8
INNER_SIZE = BOARD_SIZE - 1

# Internal intersections in the detector's grid coordinate system.
_GRID_COORDS = np.array(
    [(x, y) for y in range(1, BOARD_SIZE) for x in range(1, BOARD_SIZE)],
    dtype=np.float32,
)

# Outer board boundary in TL, TR, BR, BL order.
_BOUNDARY_COORDS = np.array(
    [[0, 0], [BOARD_SIZE, 0], [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE]],
    dtype=np.float32,
)


@dataclass
class TrackerResult:
    """Current board geometry and tracking diagnostics."""

    square_centres: np.ndarray       # (8, 8, 2), image pixels
    board_corners: np.ndarray        # (4, 2), TL, TR, BR, BL
    inner: np.ndarray                # (7, 7, 2), tracked intersections; NaN if lost
    homography: np.ndarray           # canonical board grid -> image pixels
    tracking_ok: bool
    tracked_count: int
    corner_colours: Optional[dict]
    orientation_candidates: tuple
    detector_rotation: int


class BoardTracker:
    """
    Track a previously detected board without running the expensive detector
    on every frame.

    Typical use:
        from find_board import detect_chessboard
        from board_tracker import BoardTracker

        tracker = BoardTracker()
        detection = detect_chessboard(first_frame)
        result = tracker.initialize(first_frame, detection)

        # In the video loop:
        result = tracker.update(frame)
        centres = result.square_centres

        # Occasionally, correct accumulated optical-flow drift:
        detection = detect_chessboard(frame)
        result = tracker.update(frame, detection=detection)
    """

    def __init__(
        self,
        min_tracked_points=10,
        max_lk_error=30.0,
        corner_patch_radius=5,
        corner_contrast_threshold=18.0,
        ransac_threshold=3.0,
        forward_backward_threshold=2.0,
    ):
        self.min_tracked_points = int(min_tracked_points)
        self.max_lk_error = float(max_lk_error)
        self.corner_patch_radius = int(corner_patch_radius)
        self.corner_contrast_threshold = float(corner_contrast_threshold)
        self.ransac_threshold = float(ransac_threshold)
        self.forward_backward_threshold = float(forward_backward_threshold)

        self.previous_gray = None
        self.homography = None
        self.tracked_points = None
        self.tracked_grid_coords = None
        self.corner_colours = None
        self.orientation_candidates = ()
        self.detector_rotation = 0
        self.initialised = False
        self.last_tracking_ok = False

        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

    @staticmethod
    def _as_gray(frame):
        if frame is None:
            raise ValueError("Frame cannot be None.")
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError("Frame must be a grayscale or BGR image.")

    @staticmethod
    def _project_grid(homography, coords):
        points = np.asarray(coords, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, homography).reshape(-1, 2)

    @staticmethod
    def _make_homography(board_corners):
        corners = np.asarray(board_corners, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(corners).all():
            raise ValueError("board_corners contains non-finite coordinates.")
        homography = cv2.getPerspectiveTransform(
            _BOUNDARY_COORDS.astype(np.float32), corners.astype(np.float32)
        )
        if homography is None or not np.isfinite(homography).all():
            raise RuntimeError("Could not calculate a board homography.")
        return homography

    def _sample_corner_colours(self, gray, square_centres):
        """
        Classify the four corner squares using small median-intensity patches.

        This is intended for a normally lit, non-inverted chessboard image.
        It is deliberately conservative: low contrast returns 'unknown'.
        """
        centres = np.asarray(square_centres, dtype=np.float32)
        corner_points = {
            "top_left": centres[0, 0],
            "top_right": centres[0, -1],
            "bottom_right": centres[-1, -1],
            "bottom_left": centres[-1, 0],
        }

        height, width = gray.shape[:2]
        radius = self.corner_patch_radius
        intensities = {}

        for name, point in corner_points.items():
            x, y = np.round(point).astype(int)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            patch = gray[y0:y1, x0:x1]
            intensities[name] = (
                float(np.median(patch)) if patch.size else float("nan")
            )

        values = np.array(list(intensities.values()), dtype=np.float32)
        if not np.isfinite(values).all():
            return intensities, {name: "unknown" for name in intensities}

        contrast = float(values.max() - values.min())
        if contrast < self.corner_contrast_threshold:
            return intensities, {name: "unknown" for name in intensities}

        midpoint = float((values.max() + values.min()) / 2.0)
        labels = {
            name: ("white" if value >= midpoint else "black")
            for name, value in intensities.items()
        }
        return intensities, labels

    @staticmethod
    def _rotate_detection(square_centres, board_corners, inner, k):
        """
        Rotate detector arrays by k quarter-turns counter-clockwise.

        The detector's outer-corner array is TL, TR, BR, BL in its own
        arbitrary grid convention. np.roll(..., -k) keeps it consistent with
        np.rot90 on the square-centre and inner-intersection arrays.
        """
        return (
            np.rot90(square_centres, k=k).copy(),
            np.roll(np.asarray(board_corners), -k, axis=0).copy(),
            np.rot90(inner, k=k).copy(),
        )

    def _select_initial_rotation(self, gray, square_centres, board_corners, inner):
        _, labels = self._sample_corner_colours(gray, square_centres)
        top_left = labels["top_left"]

        # Standard chessboard convention: A8 (top-left) is a light square.
        # Rotating by one quarter-turn swaps the colour at the top-left.
        if top_left == "black":
            first_rotation = 1
        else:
            # If classification is unknown, retain the detector's initial
            # ordering rather than making an unsupported guess.
            first_rotation = 0

        # The opposite 180-degree rotation has the same corner-square colour.
        self.orientation_candidates = (
            first_rotation,
            (first_rotation + 2) % 4,
        )
        self.detector_rotation = first_rotation

        rotated = self._rotate_detection(
            square_centres, board_corners, inner, first_rotation
        )
        self.corner_colours = labels
        return rotated

    def _reset_tracked_points(self, inner):
        """
        Initialise optical-flow points from measured intersections where
        available, and homography projections for any missing intersections.
        """
        projected = self._project_grid(self.homography, _GRID_COORDS)
        measured = np.asarray(inner, dtype=np.float32).reshape(-1, 2)

        valid_measured = np.isfinite(measured).all(axis=1)
        points = projected.copy()
        points[valid_measured] = measured[valid_measured]

        self.tracked_points = points.reshape(-1, 1, 2).astype(np.float32)
        self.tracked_grid_coords = _GRID_COORDS.copy()

    def initialize(self, frame, detection):
        """
        Initialise from a detector result tuple:
            (square_centres, board_corners, inner)
        """
        if detection is None or len(detection) != 3:
            raise ValueError(
                "detection must be (square_centres, board_corners, inner)."
            )

        gray = self._as_gray(frame)
        square_centres, board_corners, inner = detection

        square_centres = np.asarray(square_centres, dtype=np.float32)
        board_corners = np.asarray(board_corners, dtype=np.float32)
        inner = np.asarray(inner, dtype=np.float32)

        if square_centres.shape != (8, 8, 2):
            raise ValueError("square_centres must have shape (8, 8, 2).")
        if board_corners.shape != (4, 2):
            raise ValueError("board_corners must have shape (4, 2).")
        if inner.shape != (7, 7, 2):
            raise ValueError("inner must have shape (7, 7, 2).")

        square_centres, board_corners, inner = self._select_initial_rotation(
            gray, square_centres, board_corners, inner
        )
        self.homography = self._make_homography(board_corners)
        self._reset_tracked_points(inner)
        self.previous_gray = gray.copy()
        self.initialised = True
        self.last_tracking_ok = True

        return self._make_result(True)

    def _align_redetection(self, detection):
        """
        Align a fresh detector result to the tracker's existing grid.

        All four rotations are tested against the board corners predicted by
        the current homography. This avoids trusting the detector's arbitrary
        corner ordering after it has been run again.
        """
        square_centres, board_corners, inner = detection
        square_centres = np.asarray(square_centres, dtype=np.float32)
        board_corners = np.asarray(board_corners, dtype=np.float32)
        inner = np.asarray(inner, dtype=np.float32)

        predicted_corners = self._project_grid(self.homography, _BOUNDARY_COORDS)
        best = None

        for k in range(4):
            rotated = self._rotate_detection(
                square_centres, board_corners, inner, k
            )
            candidate_centres, candidate_corners, candidate_inner = rotated
            error = float(
                np.mean(np.linalg.norm(candidate_corners - predicted_corners, axis=1))
            )
            if best is None or error < best[0]:
                best = (
                    error,
                    k,
                    candidate_centres,
                    candidate_corners,
                    candidate_inner,
                )

        _, k, centres, corners, aligned_inner = best
        self.detector_rotation = k
        self.homography = self._make_homography(corners)
        self._reset_tracked_points(aligned_inner)
        return centres, corners, aligned_inner

    def _track(self, gray):
        if self.tracked_points is None or len(self.tracked_points) < self.min_tracked_points:
            return False

        next_points, status_forward, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.tracked_points,
            None,
            **self._lk_params,
        )
        if next_points is None or status_forward is None:
            return False

        previous_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            next_points,
            None,
            **self._lk_params,
        )
        if previous_back is None or status_backward is None:
            return False

        old_points = self.tracked_points.reshape(-1, 2)
        new_points = next_points.reshape(-1, 2)
        back_points = previous_back.reshape(-1, 2)
        status = status_forward.reshape(-1).astype(bool)
        status &= status_backward.reshape(-1).astype(bool)

        fb_error = np.linalg.norm(old_points - back_points, axis=1)
        status &= fb_error <= self.forward_backward_threshold

        if error is not None:
            status &= error.reshape(-1) <= self.max_lk_error

        status &= np.isfinite(new_points).all(axis=1)
        if int(status.sum()) < self.min_tracked_points:
            return False

        good_new = new_points[status]
        good_grid = self.tracked_grid_coords[status]

        homography, inlier_mask = cv2.findHomography(
            good_grid,
            good_new,
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if homography is None or inlier_mask is None:
            return False

        inliers = inlier_mask.reshape(-1).astype(bool)
        if int(inliers.sum()) < self.min_tracked_points:
            return False

        self.homography = homography
        self.tracked_points = good_new[inliers].reshape(-1, 1, 2).astype(np.float32)
        self.tracked_grid_coords = good_grid[inliers].astype(np.float32)
        return True

    def _make_result(self, tracking_ok):
        square_grid = np.array(
            [(col + 0.5, row + 0.5)
             for row in range(BOARD_SIZE)
             for col in range(BOARD_SIZE)],
            dtype=np.float32,
        )
        square_centres = self._project_grid(self.homography, square_grid).reshape(
            BOARD_SIZE, BOARD_SIZE, 2
        )
        board_corners = self._project_grid(self.homography, _BOUNDARY_COORDS)

        inner = np.full((INNER_SIZE, INNER_SIZE, 2), np.nan, dtype=np.float32)
        if self.tracked_grid_coords is not None:
            for grid_coord, point in zip(
                self.tracked_grid_coords, self.tracked_points.reshape(-1, 2)
            ):
                col = int(round(float(grid_coord[0]))) - 1
                row = int(round(float(grid_coord[1]))) - 1
                if 0 <= row < INNER_SIZE and 0 <= col < INNER_SIZE:
                    inner[row, col] = point

        return TrackerResult(
            square_centres=square_centres,
            board_corners=board_corners,
            inner=inner,
            homography=self.homography.copy(),
            tracking_ok=bool(tracking_ok),
            tracked_count=(
                0 if self.tracked_points is None else len(self.tracked_points)
            ),
            corner_colours=self.corner_colours,
            orientation_candidates=self.orientation_candidates,
            detector_rotation=self.detector_rotation,
        )

    def update(self, frame, detection=None):
        """
        Update the board geometry for a new frame.

        Args:
            frame: Current BGR or grayscale frame.
            detection: Optional fresh detect_chessboard(...) result. Supply
                this occasionally to correct optical-flow drift.

        Returns:
            TrackerResult. If tracking fails, the last valid homography is
            retained and tracking_ok is False. Re-detect and pass detection
            to recover.
        """
        if not self.initialised:
            if detection is None:
                raise RuntimeError(
                    "Tracker is not initialised. Supply a detector result."
                )
            return self.initialize(frame, detection)

        gray = self._as_gray(frame)

        if detection is not None:
            self._align_redetection(detection)
            self.previous_gray = gray.copy()
            self.last_tracking_ok = True
            return self._make_result(True)

        tracking_ok = self._track(gray)
        self.previous_gray = gray.copy()
        self.last_tracking_ok = tracking_ok
        return self._make_result(tracking_ok)

    def reset(self):
        """Clear tracker state so it can be initialised on another board."""
        self.previous_gray = None
        self.homography = None
        self.tracked_points = None
        self.tracked_grid_coords = None
        self.corner_colours = None
        self.orientation_candidates = ()
        self.detector_rotation = 0
        self.initialised = False
        self.last_tracking_ok = False
