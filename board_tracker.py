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

from __future__ import annotations
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


@dataclass
class OrientedTrackerResult(TrackerResult):
    """Tracker result with front-face orientation diagnostics."""
    orientation_verified: bool = False
    orientation_status: str = "UNVERIFIED"
    orientation_reason: str = "Not initialized"
    colour_margin: float = float("nan")
    signed_area_px2: float = float("nan")
    detector_flip: bool = False


class OrientationUnverified(ValueError):
    """A detection cannot be assigned to the front-face grid with confidence."""


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
        self.detector_flip = False
        self.initialised = False
        self.last_tracking_ok = False
        self._orientation_verified = False
        self._orientation_status = "UNVERIFIED"
        self._orientation_reason = "Awaiting front-face board detection"
        self._orientation_margin = float("nan")
        self._orientation_gray = None

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
    def _transform_detection(square_centres, board_corners, inner, k, flip):
        """Apply a k rotations of 90deg and/or flip the axis of the detection."""
        centres = np.rot90(square_centres, k=k)
        intersections = np.rot90(inner, k=k)

        corners = np.roll(board_corners, -k, axis=0)

        if flip:
            centres = np.fliplr(centres)
            intersections = np.fliplr(intersections)
            corners = corners[[1, 0, 3, 2]]

        return centres.copy(), corners.copy(), intersections.copy()

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

        rotated = self._transform_detection(
            square_centres, board_corners, inner, first_rotation, False
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

        try:
            result = self._make_result(True)
        except OrientationUnverified:
            self._orientation_verified = False
            self._orientation_status = "UNVERIFIED"
            self._orientation_reason = "Initialization rejected: cannot verify front-face corner IDs"
            raise
        return self._check_live_geometry(frame, result)

    def _align_redetection(self, detection):
        """
        Align a fresh detector result to the tracker's existing grid.

        All four rotations are tested against the board corners predicted by
        the current homography. This avoids trusting the detector's arbitrary
        corner ordering after it has been run again.
        """
        if self._orientation_gray is None:
                    raise RuntimeError("Current detection frame is not available")
        predicted = self._project_grid(self.homography, np.array(
            [[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32
        ))
        selected = self._choose_orientation(
            self._orientation_gray, detection, predicted_corners=predicted
        )
        centres, corners, inner = selected["centres"], selected["corners"], selected["inner"]
        self.homography = self._make_homography(corners)
        self._reset_tracked_points(inner)
        return centres, corners, inner

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

        unoriented = TrackerResult(
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

        area = self._signed_area(unoriented.board_corners)
        return OrientedTrackerResult(
            **vars(unoriented),
            orientation_verified=bool(self._orientation_verified and tracking_ok),
            orientation_status=self._orientation_status,
            orientation_reason=self._orientation_reason,
            colour_margin=float(self._orientation_margin),
            signed_area_px2=area,
            detector_flip=bool(self.detector_flip),
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

        try:
            unoriented = self._make_result(tracking_ok)
        finally:
            self._orientation_gray = None
        return self._check_live_geometry(frame, unoriented)
                

    def draw_debug(self, frame, result=None, show_grid=True):
        """
        Draw the tracker's board-corner ordering on a copy of the frame.

        Labels describe the tracker's current canonical coordinate convention:
            0 = TL / A8, 1 = TR / H8,
            2 = BR / H1, 3 = BL / A1.

        These chess-square labels are only an assumed orientation. On an
        empty board, a 180-degree ambiguity may remain. The numeric corner
        indices are the most useful way to diagnose ordering mismatches
        between the detector, tracker, and pose-estimation code.
        """
        if result is None:
            if not self.initialised:
                return frame.copy()
            result = self._make_result(self.last_tracking_ok)

        output = frame.copy()
        corners = np.asarray(result.board_corners, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(corners).all():
            return output

        height, width = output.shape[:2]
        names = (
            ("0: TL / A8", "1: TR / H8", "2: BR / H1", "3: BL / A1")
        )

        if show_grid:
            polygon = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [polygon], True, (255, 180, 0), 2, cv2.LINE_AA)

        # Draw the corners with distinct colours so their ordering is obvious.
        marker_colours = (
            (0, 0, 255),       # 0: red
            (0, 255, 0),       # 1: green
            (255, 0, 0),       # 2: blue
            (0, 255, 255),     # 3: yellow
        )

        for index, (point, label, colour) in enumerate(
            zip(corners, names, marker_colours)
        ):
            x, y = np.round(point).astype(int)
            cv2.circle(output, (x, y), 9, colour, -1, cv2.LINE_AA)
            cv2.circle(output, (x, y), 13, (255, 255, 255), 2, cv2.LINE_AA)

            # Offset labels inward from the corresponding image edge.
            text_x = int(np.clip(x + (12 if x < width / 2 else -150), 5, max(5, width - 155)))
            text_y = int(np.clip(y + (25 if y < height / 2 else -12), 22, height - 5))

            cv2.putText(
                output, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                output, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2, cv2.LINE_AA,
            )

            coord_label = f"({x}, {y})"
            coord_y = min(height - 5, text_y + 20)
            cv2.putText(
                output, coord_label, (text_x, coord_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                output, coord_label, (text_x, coord_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
            )

        status = (
            f"Tracker {'OK' if result.tracking_ok else 'LOST'} | "
            f"points: {result.tracked_count} | "
            f"detector rotation: {result.detector_rotation} | "
            f"possible rotations: {result.orientation_candidates}"
        )
        cv2.putText(
            output, status, (15, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            output, status, (15, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return output

    def reset(self):
        """Clear tracker state so it can be initialised on another board."""
        self.previous_gray = None
        self.homography = None
        self.tracked_points = None
        self.tracked_grid_coords = None
        self.corner_colours = None
        self.orientation_candidates = ()
        self.detector_rotation = 0
        self.detector_flip = False
        self.initialised = False
        self.last_tracking_ok = False
        self._orientation_verified = False
        self._orientation_status = "UNVERIFIED"
        self._orientation_reason = "Awaiting front-face board detection"
        self._orientation_margin = float("nan")
        self._orientation_gray = None

    @staticmethod
    def _signed_area(corners) -> float:
        """Signed image polygon area; + is front-facing for this 3D mapping.

        Assumes a camera image NOT horizontally mirrored and the printed front
        of the board visible. The signs are in OpenCV's y-down image system.
        """
        corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(corners).all():
            return float("nan")
        return float(0.5 * np.sum(
            corners[:, 0] * np.roll(corners[:, 1], -1)
            - corners[:, 1] * np.roll(corners[:, 0], -1)
        ))

    def _front_face_margin(self, gray, centres) -> tuple[float, float]:
            """Min white intensity minus max black intensity at four square centres.

            Unlike a global midpoint classification, this requires *both* white
            squares to be brighter than *both* black squares by a margin.
            """
            intensities, _ = self._sample_corner_colours(gray, centres)
            values = list(intensities.values())
            if len(values) != 4 or not np.isfinite(values).all():
                return float("nan"), float("nan")
            # The original _sample_corner_colours orders samples as TL/TR/BR/BL.
            white_min = min(values[0], values[2])
            black_max = max(values[1], values[3])
            black_min = min(values[1], values[3])
            white_max = max(values[0], values[2])
            return float(white_min - black_max), float(black_min - white_max)

    def _verified_candidates(self, gray, detection):
            """Enumerate only assignments with black ID3 and outward +Z.

            Rotation and reflection machinery is unchanged from the base tracker.
            A positive signed polygon is required. Strong four-square colour
            separation resolves the 90-degree parity class, but leaves 180 degrees.
            """
            centres, corners, inner = detection
            valid = []
            for flip in (False, True):
                for k in range(4):
                    transformed = self._transform_detection(centres, corners, inner, k, flip)
                    rotated_centres, rotated_corners, rotated_inner = transformed
                    area = self._signed_area(rotated_corners)
                    if not np.isfinite(area) or area <= 1.0:
                        continue
                    margin, _ = self._front_face_margin(gray, rotated_centres)
                    if not np.isfinite(margin) or margin < self.corner_contrast_threshold:
                        continue
                    valid.append({
                        "k": k, "flip": flip, "centres": rotated_centres,
                        "corners": rotated_corners, "inner": rotated_inner,
                        "area": area, "margin": margin,
                    })
            return valid

    def _choose_orientation(self, gray, detection, predicted_corners=None):
        valid = self._verified_candidates(gray, detection)
        if not valid:
            raise OrientationUnverified(
                "No front-facing black-ID3 assignment verified "
                "(corner colours occluded/low contrast or boundary winding wrong)"
            )
        if predicted_corners is None:
            # Initial fixation is intentionally deterministic. Prefer the
            # original detector's grid when it already satisfies the rules.
            selected = min(valid, key=lambda c: (int(c["flip"]), c["k"]))
        else:
            # Same correspondence/proximity objective as the original tracker,
            # but applied ONLY to front-facing, colour-valid candidates.
            for c in valid:
                c["alignment_error"] = float(np.mean(np.linalg.norm(
                    c["corners"] - predicted_corners, axis=1
                )))
            selected = min(valid, key=lambda c: (
                c["alignment_error"], int(c["flip"]), c["k"]
            ))
        self.orientation_candidates = tuple(c["k"] for c in valid)
        self.detector_rotation = selected["k"]
        self.detector_flip = selected["flip"]
        self._orientation_verified = True
        self._orientation_status = "VERIFIED"
        self._orientation_reason = (
            f"front-face IDs locked; ID3 black; +Z out; "
            f"k={selected['k']} flip={int(selected['flip'])}"
        )
        self._orientation_margin = selected["margin"]
        _, self.corner_colours = self._sample_corner_colours(gray, selected["centres"])
        return selected

    def _select_initial_rotation(self, gray, square_centres, board_corners, inner):
        """Override only the source tracker's initial corner association."""
        selected = self._choose_orientation(
            gray, (square_centres, board_corners, inner)
        )
        return selected["centres"], selected["corners"], selected["inner"]
      
    def _check_live_geometry(self, frame, result):
        """Fail closed for an explicit front-face contradiction after LK.

        No new detector or tracker estimator is called. Uncertain colours under
        hand occlusion retain the established temporal association; a clearly
        inverted black/white pattern or reversed winding invalidates the pose.
        """
        if not result.tracking_ok:
            return result
        area = self._signed_area(result.board_corners)
        if not np.isfinite(area) or area <= 1.0:
            reason = "Boundary winding reversed: +Z would face into board"
        else:
            gray = self._as_gray(frame)
            margin, inverted_margin = self._front_face_margin(gray, result.square_centres)
            if np.isfinite(inverted_margin) and inverted_margin >= self.corner_contrast_threshold:
                reason = "Corner colours inverted: ID3 would be a WHITE square"
            else:
                reason = None
                if np.isfinite(margin):
                    self._orientation_margin = margin
                if np.isfinite(margin) and margin >= self.corner_contrast_threshold:
                    self._orientation_status = "VERIFIED"
                    self._orientation_reason = "front-face winding and corner colours confirmed"
                else:
                    self._orientation_status = "LOCKED_COLOURS_OCCLUDED"
                    self._orientation_reason = "Keeping verified corner IDs through ambiguous corner colours"
        if reason is not None:
            self.last_tracking_ok = False
            self._orientation_verified = False
            self._orientation_status = "REJECTED"
            self._orientation_reason = reason
            result.tracking_ok = False
            result.orientation_verified = False
        result.orientation_status = self._orientation_status
        result.orientation_reason = self._orientation_reason
        result.colour_margin = float(self._orientation_margin)
        result.signed_area_px2 = area
        return result