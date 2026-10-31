"""Pinch state detection from MediaPipe hand landmarks."""

import math

FRAME_WIDTH, FRAME_HEIGHT = 800, 600
THUMB, INDEX, WRIST, MIDDLE_BASE, INDEX_BASE, PINKY_BASE = 4, 8, 0, 9, 5, 17


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PinchDetector:
    def __init__(self, pinch=0.25, release=0.30, frames=3, lost_frames=30, contact_fraction=1/4):
        if not 0 <= contact_fraction <= 1:
            raise ValueError("Contact fraction must be between 0 and 1")
        self.pinch, self.release, self.frames = pinch, release, frames
        self.lost_frames, self.contact_fraction = lost_frames, contact_fraction
        self.active, self.count, self.missing = False, 0, 0
        self.ratio, self.cursor = None, None
        self.release_position = None

    def update(self, landmarks):
        if landmarks is None:
            self.missing += 1
            self.count, self.release_position = 0, None
            event = None
            if self.missing >= self.lost_frames:
                event = "PINCH_CANCEL" if self.active else None
                self.active = False
                self.ratio, self.cursor = None, None
            return {"state": "NO HAND", "position": None, "ratio": None, "event": event, "event_position": None}

        points = [(p.x * FRAME_WIDTH, p.y * FRAME_HEIGHT) for p in landmarks]
        palm = max(distance(points[WRIST], points[MIDDLE_BASE]), distance(points[INDEX_BASE], points[PINKY_BASE]), 1.0)
        raw_ratio = distance(points[THUMB], points[INDEX]) / palm
        self.ratio = raw_ratio if self.ratio is None else 0.5 * raw_ratio + 0.5 * self.ratio

        # Place the cursor one-third of the way from thumb to index.
        thumb, index = points[THUMB], points[INDEX]
        contact = tuple(t + self.contact_fraction * (i - t) for t, i in zip(thumb, index))
        self.cursor = contact if self.cursor is None else tuple(0.55 * p + 0.45 * old for p, old in zip(contact, self.cursor))
        position = tuple(round(value) for value in self.cursor)
        self.missing = 0

        # Save the first qualifying release position until release is confirmed.
        change = self.ratio > self.release if self.active else self.ratio < self.pinch
        if self.active and change and self.count == 0:
            self.release_position = position
        if not change:
            self.release_position = None
        self.count = self.count + 1 if change else 0
        event, event_position = None, None
        if self.count >= self.frames:
            self.active = not self.active
            self.count = 0
            event = "PINCH_START" if self.active else "PINCH_RELEASE"
            event_position = position if self.active else self.release_position
            self.release_position = None

        return {"state": "PINCHED" if self.active else "OPEN", "position": position, "ratio": self.ratio,
                "event": event, "event_position": event_position}
