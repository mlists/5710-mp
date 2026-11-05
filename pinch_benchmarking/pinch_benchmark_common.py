"""Shared file format, validation and detector adapters for pinch-only benchmarking.

The original pinch_detector.py remains untouched. All real data are stored locally.
Frame indices are zero-based and refer to the exact order of saved video frames.
The authoritative time axis is time_s, measured at the end of each camera read.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

from pinch_detector import (PinchDetector, THUMB, INDEX, WRIST, MIDDLE_BASE,
                            INDEX_BASE, PINKY_BASE, distance)

MANIFEST_FIELDS = [
    "clip_id", "experiment", "condition", "distance_cm", "orientation_deg",
    "participant", "hand", "lighting", "expected_cycles", "notes",
    "video_file", "log_file", "frames_file", "frame_count", "duration_s",
    "capture_fps_measured", "video_fps_nominal", "width", "height",
    "camera_index", "camera_backend", "camera_fourcc", "camera_fps_reported",
    "requested_width", "requested_height", "requested_fps", "mirror",
    "pinch", "release", "confirmation_frames", "lost_frames",
    "min_detection_confidence", "min_presence_confidence",
    "min_tracking_confidence", "model_sha256", "python_version",
    "opencv_version", "mediapipe_version", "os_version",
    "created_utc", "status",
]
GROUND_TRUTH_FIELDS = ["clip_id", "event_id", "event_type", "frame_index", "time_s", "uncertain", "notes"]
VISIBILITY_FIELDS = ["clip_id", "start_frame", "end_frame", "visibility"]
STATUS_FIELDS = ["clip_id", "status", "reviewed_at_utc", "notes"]
EVENT_TYPES = ("PINCH_START", "PINCH_RELEASE")
VISIBILITY_TYPES = ("PRESENT", "ABSENT", "UNKNOWN")
FRAME_CSV_FIELDS = [
    "clip_id", "frame_index", "time_s", "video_time_s", "mp_timestamp_ms",
    "hand_detected", "valid_landmarks", "raw_ratio", "smoothed_ratio",
    "state", "event", "cursor_x", "cursor_y", "event_x", "event_y",
    "capture_ms", "preprocess_ms", "mediapipe_ms", "gesture_ms",
    "frame_to_gesture_ms", "video_write_ms", "full_loop_ms",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def ensure_session(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    for sub in ("recordings", "landmarks", "frames", "results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def clean_clip_id(clip_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", clip_id) or clip_id in (".", ".."):
        raise ValueError("--clip-id must be 1-100 characters, using letters/digits/_.- and starting with a letter/digit")
    return clip_id


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def replace_clip_rows(path: Path, fields: list[str], clip_id: str,
                      rows: list[dict[str, Any]]) -> None:
    old = [x for x in read_csv(path) if x.get("clip_id") != clip_id]
    write_csv(path, fields, old + rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise ValueError(f"No frame records in {path}")
    for idx, row in enumerate(rows):
        if row.get("frame_index") != idx:
            raise ValueError(f"Frame-index gap at row {idx} in {path}")
        if idx and float(row["time_s"]) <= float(rows[idx-1]["time_s"]):
            raise ValueError(f"Non-increasing capture time at frame {idx} in {path}")
        points = row.get("landmarks")
        if points is not None and (len(points) != 21 or any(len(p) < 2 for p in points)):
            raise ValueError(f"Expected 21 landmarks at frame {idx}: {path}")
    return rows


def manifest_by_id(session: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(session / "manifest.csv")
    if len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate clip IDs in manifest.csv")
    return {r["clip_id"]: r for r in rows}


def session_file(session: Path, relative: str) -> Path:
    """Prevent an edited manifest from referencing files outside the session."""
    target = (session / relative).resolve()
    if not target.is_relative_to(session.resolve()):
        raise ValueError(f"File referenced outside session: {relative}")
    return target


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def points_to_landmarks(points: list[list[float]] | None):
    if points is None:
        return None
    return [SimpleNamespace(x=float(p[0]), y=float(p[1]),
                            z=float(p[2]) if len(p) > 2 else 0.0) for p in points]


def raw_measurement(landmarks, width: int, height: int):
    """Return (raw distance/palm-size ratio, palm size) without smoothing.

    A degenerate palm (< 1px) is handled consistently as a missing hand by
    safe_detector_update, while original landmarks remain saved for inspection.
    """
    if landmarks is None or len(landmarks) != 21:
        return None, None
    xy = [(float(p.x) * width, float(p.y) * height) for p in landmarks]
    palm = max(distance(xy[WRIST], xy[MIDDLE_BASE]),
               distance(xy[INDEX_BASE], xy[PINKY_BASE]))
    if not math.isfinite(palm):
        return None, None
    if palm < 1.0:
        return None, palm
    ratio = distance(xy[THUMB], xy[INDEX]) / palm
    if not math.isfinite(ratio):
        return None, palm
    return ratio, palm


def safe_detector_update(detector: PinchDetector, landmarks, width: int, height: int):
    """Adapter only; no changes to the original algorithm on valid landmarks."""
    ratio, palm = raw_measurement(landmarks, width, height)
    valid = ratio is not None
    result = detector.update(landmarks if valid else None, width, height)
    if result is None:
        # Defensive fallback, should never be reached after the precheck.
        result = detector.update(None, width, height)
        valid = False
    return result, ratio, valid, palm


def frame_duration(rows: list[dict[str, Any]], nominal_fps: float = 30.0) -> float:
    if len(rows) < 2:
        return 1.0 / max(nominal_fps, 1.0)
    times = [float(r["time_s"]) for r in rows]
    delta = median(b-a for a, b in zip(times, times[1:]))
    return max(0.0, times[-1] - times[0]) + max(0.0, delta)


def visibility_at_frames(session: Path, clip_id: str, nframes: int) -> list[str]:
    labels = ["UNKNOWN"] * nframes
    segments = [r for r in read_csv(session / "ground_truth_visibility.csv")
                if r.get("clip_id") == clip_id]
    for row in segments:
        lo, hi = int(row["start_frame"]), int(row["end_frame"])
        kind = row["visibility"]
        if not 0 <= lo <= hi < nframes or kind not in VISIBILITY_TYPES:
            raise ValueError(f"Invalid visibility interval for {clip_id}: {row}")
        for index in range(lo, hi + 1):
            if labels[index] != "UNKNOWN" and kind != labels[index]:
                raise ValueError(f"Overlapping visibility labels for {clip_id} at frame {index}")
            labels[index] = kind
    return labels


def write_visibility_breakpoints(session: Path, clip_id: str,
                                 breaks: dict[int, str], nframes: int) -> None:
    if nframes <= 0:
        raise ValueError("Empty clip")
    valid = sorted((int(k), v) for k, v in breaks.items())
    if not valid or valid[0][0] != 0:
        valid.insert(0, (0, "UNKNOWN"))
    compact: list[tuple[int, str]] = []
    for start, kind in valid:
        if not (0 <= start < nframes) or kind not in VISIBILITY_TYPES:
            raise ValueError("Invalid visibility breakpoint")
        if compact and compact[-1][1] == kind:
            continue
        compact.append((start, kind))
    rows = [{"clip_id": clip_id, "start_frame": start,
             "end_frame": (compact[j+1][0]-1 if j+1 < len(compact) else nframes-1),
             "visibility": kind} for j, (start, kind) in enumerate(compact)]
    replace_clip_rows(session / "ground_truth_visibility.csv", VISIBILITY_FIELDS, clip_id, rows)


def visibility_breakpoints(session: Path, clip_id: str) -> dict[int, str]:
    return {int(r["start_frame"]): r["visibility"] for r in
            read_csv(session / "ground_truth_visibility.csv") if r.get("clip_id") == clip_id}


def visibility_for_frame(breaks: dict[int, str], frame: int) -> str:
    eligible = [i for i in breaks if i <= frame]
    return breaks[max(eligible)] if eligible else "UNKNOWN"
