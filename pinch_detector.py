"""Pinch state detection from MediaPipe hand landmarks."""

import math

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

THUMB = 4
INDEX = 8
WRIST = 0
MIDDLE_BASE = 9
INDEX_BASE = 5
PINKY_BASE = 17


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PinchDetector:
    def __init__(self, pinch=0.25, release=0.30, frames=3, lost_frames=6, contact_fraction=1 / 4):
        if not 0 < pinch < release or frames < 1 or lost_frames < 1:
            raise ValueError("Require 0 < pinch < release and positive frame counts")
        if not 0 <= contact_fraction <= 1:
            raise ValueError("Contact fraction must be between 0 and 1")

        self.pinch = pinch
        self.release = release
        self.frames = frames
        self.lost_frames = lost_frames
        self.contact_fraction = contact_fraction
        self.active = False
        self.count = 0
        self.missing = 0
        self.ratio = None
        self.cursor = None
        self.release_position = None

    def update(self, landmarks, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        if landmarks is None:
            self.missing += 1
            self.count = 0
            self.release_position = None
            event = None

            if self.missing >= self.lost_frames:
                event = "PINCH_CANCEL" if self.active else None
                self.active = False
                self.ratio = None
                self.cursor = None

            return {
                "state": "NO HAND",
                "position": None,
                "ratio": None,
                "event": event,
                "event_position": None,
            }

        points = []
        for landmark in landmarks:
            points.append((landmark.x * width, landmark.y * height))

        palm = max(distance(points[WRIST], points[MIDDLE_BASE]),
                   distance(points[INDEX_BASE], points[PINKY_BASE]), 1.0)
        raw_ratio = distance(points[THUMB], points[INDEX]) / palm

        if self.ratio is None:
            self.ratio = raw_ratio
        else:
            self.ratio = 0.5 * raw_ratio + 0.5 * self.ratio

        # Place the contact point between the thumb and index finger.
        thumb = points[THUMB]
        index = points[INDEX]
        contact = (thumb[0] + self.contact_fraction * (index[0] - thumb[0]),
                   thumb[1] + self.contact_fraction * (index[1] - thumb[1]))

        if self.cursor is None:
            self.cursor = contact
        else:
            self.cursor = (0.55 * contact[0] + 0.45 * self.cursor[0],
                           0.55 * contact[1] + 0.45 * self.cursor[1])

        position = (round(self.cursor[0]), round(self.cursor[1]))
        self.missing = 0

        # Save the contact position when release is first detected.
        if self.active:
            change = self.ratio > self.release
        else:
            change = self.ratio < self.pinch

        if self.active and change and self.count == 0:
            self.release_position = position
        if not change:
            self.release_position = None

        if change:
            self.count += 1
        else:
            self.count = 0

        event = None
        event_position = None
        if self.count >= self.frames:
            self.active = not self.active
            self.count = 0
            event = "PINCH_START" if self.active else "PINCH_RELEASE"
            event_position = position if self.active else self.release_position
            self.release_position = None

        return {
            "state": "PINCHED" if self.active else "OPEN",
            "position": position,
            "ratio": self.ratio,
            "event": event,
            "event_position": event_position,
        }
