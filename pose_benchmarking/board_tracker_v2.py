"""Orientation-safe association layer for the ORIGINAL board_tracker.BoardTracker.

Only corner assignment and output validity are changed. The existing detector,
LK optical flow, RANSAC, homography fitting, point seeding, and point thresholds
remain implemented and configured by the original BoardTracker base class.

Physical board convention (viewing the printed front, unmirrored camera):
    corner 0 = A8 or H1 : white
    corner 1 = H8 or A1 : black
    corner 2 = H1 or A8 : white
    corner 3 = A1 or H8 : black, fixed origin
    +X from ID3 to ID2; +Y from ID3 to ID0; +Z out of the printed face.

A1 and H8 are intentionally indistinguishable without an asymmetric marker.
A 180-degree interchange is accepted. A quarter-turn or reflection that maps
ID3 to white, or the camera onto the -Z side, is not accepted.

When a clear observation cannot establish an orientation at acquisition, the
tracker refuses to reassign corners instead of silently using a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from board_tracker import BoardTracker as _OriginalBoardTracker, TrackerResult


@dataclass
class OrientedTrackerResult(TrackerResult):
    """Original fields plus read-only diagnostics for front-face orientation."""
    orientation_verified: bool = False
    orientation_status: str = "UNVERIFIED"
    orientation_reason: str = "Not initialized"
    colour_margin: float = float("nan")
    signed_area_px2: float = float("nan")
    detector_flip: bool = False


class OrientationUnverified(ValueError):
    """A detection cannot be assigned to the front-face grid with confidence."""


class BoardTracker(_OriginalBoardTracker):
    """Thin orientation-validation subclass; no optical-flow modifications."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def _align_redetection(self, detection):
        """Override only redetection correspondence, not tracker algorithms."""
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

    def _make_result(self, tracking_ok):
        original = super()._make_result(tracking_ok)
        area = self._signed_area(original.board_corners)
        return OrientedTrackerResult(
            **vars(original),
            orientation_verified=bool(self._orientation_verified and tracking_ok),
            orientation_status=self._orientation_status,
            orientation_reason=self._orientation_reason,
            colour_margin=float(self._orientation_margin),
            signed_area_px2=area,
            detector_flip=bool(self.detector_flip),
        )

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

    def initialize(self, frame, detection):
        try:
            result = super().initialize(frame, detection)
        except OrientationUnverified:
            self._orientation_verified = False
            self._orientation_status = "UNVERIFIED"
            self._orientation_reason = "Initialization rejected: cannot verify front-face corner IDs"
            raise
        return self._check_live_geometry(frame, result)

    def update(self, frame, detection=None):
        if detection is None:
            result = super().update(frame)
        else:
            self._orientation_gray = self._as_gray(frame)
            try:
                result = super().update(frame, detection=detection)
            finally:
                self._orientation_gray = None
        return self._check_live_geometry(frame, result)

    def reset(self):
        super().reset()
        self._orientation_verified = False
        self._orientation_status = "UNVERIFIED"
        self._orientation_reason = "Awaiting front-face board detection"
        self._orientation_margin = float("nan")
        self._orientation_gray = None
