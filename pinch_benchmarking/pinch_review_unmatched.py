"""Interactive, READ-ONLY comparison of human annotations and detector decisions.

Keys (press inside the OpenCV window):
    n/p : next/previous unmatched event (FP/FN)
    t   : toggle navigation between unmatched events and ALL matched/unmatched events
    a/d : step 1 recorded frame back/forward
    j/l : step 10 frames back/forward
    c   : return to the selected event frame
    Space : play / pause    m : toggle MediaPipe landmark overlay
    s   : save an inspection screenshot    q/Esc : quit

Meaning: GT green circle = manually labelled PINCH_START;
         GT blue square = manually labelled PINCH_RELEASE;
         red ring around GT = FN (ground truth not matched);
         prediction yellow = TP (matched prediction);
         prediction red X = FP (unmatched prediction);
         grey = uncertain/ignored prediction.

The benchmark matches START and RELEASE separately within ±0.5 s by default.
A FP and FN near each other can be a *timing mismatch*, not necessarily two
independent gesture failures. Inspect neighbouring frames before interpretation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# OpenCV uses BGR, not RGB.
C = {
    "white": (245, 245, 245),
    "muted": (185, 185, 185),
    "start": (80, 220, 90),       # green: manual START
    "release": (235, 170, 35),    # blue: manual RELEASE
    "tp": (40, 220, 255),        # yellow: matched prediction
    "fp": (55, 60, 245),         # red: false prediction
    "fn": (55, 60, 245),         # red ring: missed manual event
    "uncertain": (175, 155, 175),
    "cursor": (255, 255, 255),
    "background": (27, 30, 34),
    "panel": (33, 37, 43),
}
EVENT_TYPES = ("PINCH_START", "PINCH_RELEASE")
EXPECTED_MANIFEST = ("clip_id", "video_file", "log_file", "frame_count")


def read_csv(path: Path, required: bool = True) -> list[dict[str, str]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing {path}")
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def inside_session(session: Path, relative: str) -> Path:
    """Avoid an accidentally edited manifest pointing outside the session."""
    path = (session / relative).resolve()
    if not path.is_relative_to(session.resolve()):
        raise ValueError(f"Manifest references a file outside the session: {relative}")
    return path


def as_flag(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "y", "yes")


def row_int(row: dict, key: str) -> int | None:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def row_float(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def short_type(event_type: str) -> str:
    return event_type.replace("PINCH_", "")


def get_manifest(session: Path) -> dict[str, dict[str, str]]:
    manifest = read_csv(session / "manifest.csv")
    if not manifest:
        raise ValueError(f"No entries in {session / 'manifest.csv'}")
    ids = [row.get("clip_id", "") for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip IDs found in manifest.csv")
    for row in manifest:
        if any(not row.get(field) for field in EXPECTED_MANIFEST):
            raise ValueError(f"Incomplete manifest row for {row.get('clip_id', '(unknown)')}")
    return {row["clip_id"]: row for row in manifest}


def audit_rows(session: Path, results_dir: Path, clip_id: str, confirmation: int) -> list[dict]:
    all_rows = read_csv(results_dir / "event_matching_audit.csv")
    result = [r for r in all_rows if r.get("clip_id") == clip_id
              and row_int(r, "confirmation_frames") == confirmation]
    if not result:
        print(f"WARNING: No event audit entries for {clip_id} at {confirmation} frames.")
    return result


def annotation_rows(session: Path, clip_id: str) -> tuple[list[dict], str]:
    status_file = session / "annotation_status.csv"
    status_rows = read_csv(status_file, required=False)
    statuses = [r.get("status", "") for r in status_rows if r.get("clip_id") == clip_id]
    status = statuses[-1] if statuses else "MISSING/UNKNOWN"
    gt_file = session / "ground_truth_events.csv"
    records = read_csv(gt_file, required=False)
    result = []
    for row in records:
        if row.get("clip_id") != clip_id:
            continue
        if row.get("event_type") not in EVENT_TYPES:
            raise ValueError(f"Invalid ground truth event type: {row.get('event_type')!r}")
        frame = row_int(row, "frame_index")
        if frame is None:
            raise ValueError("A ground truth row is missing its frame index")
        result.append({"frame": frame,
                       "time_s": row_float(row, "time_s"),
                       "event_type": row["event_type"],
                       "uncertain": as_flag(row.get("uncertain", False)),
                       "event_id": row.get("event_id", "")})
    result.sort(key=lambda e: (e["frame"], e["event_type"]))
    return result, status


def load_predictions(results_dir: Path, clip_id: str, confirmation: int) -> dict[int, dict]:
    entries = read_csv(results_dir / "replay_predictions.csv")
    selected = [r for r in entries if r.get("clip_id") == clip_id
                and row_int(r, "confirmation_frames") == confirmation]
    if not selected:
        raise ValueError(f"No replay predictions for {clip_id}, {confirmation} frames. "
                         "Run pinch_replay_evaluate.py and use the matching results directory.")
    by_frame = {}
    for row in selected:
        frame = row_int(row, "frame_index")
        if frame is None or frame in by_frame:
            raise ValueError("Duplicate/missing prediction frame in replay_predictions.csv")
        by_frame[frame] = {
            "frame": frame,
            "time_s": row_float(row, "time_s"),
            "event": row.get("event", "") or "",
            "state": row.get("state", "") or "UNKNOWN",
            "raw_ratio": row_float(row, "raw_ratio"),
            "smoothed_ratio": row_float(row, "smoothed_ratio"),
            "hand_detected": as_flag(row.get("hand_detected", "")),
            "valid_landmarks": as_flag(row.get("valid_landmarks", "")),
        }
    return by_frame


def classify_events(audit: list[dict]) -> tuple[dict[tuple[int, str], str], dict[tuple[int, str], str]]:
    """Build exact frame/type lookup for audit statuses on both annotation lanes."""
    gt_status = {}
    pred_status = {}
    for row in audit:
        outcome = row.get("outcome", "")
        event_type = row.get("event_type", "")
        if event_type not in EVENT_TYPES:
            continue
        gframe = row_int(row, "gt_frame")
        pframe = row_int(row, "prediction_frame")
        if gframe is not None and outcome in ("TP", "FN"):
            gt_status[(gframe, event_type)] = outcome
        if pframe is not None and outcome in ("TP", "FP", "IGNORED_UNCERTAIN"):
            pred_status[(pframe, event_type)] = outcome
    return gt_status, pred_status


def make_queue(audit: list[dict], gt: list[dict], only_unmatched: bool) -> list[dict]:
    queue = []
    for row in audit:
        outcome = row.get("outcome", "")
        if only_unmatched and outcome not in ("FP", "FN"):
            continue
        if outcome not in ("TP", "FP", "FN", "IGNORED_UNCERTAIN"):
            continue
        key = "prediction_frame" if outcome in ("FP", "IGNORED_UNCERTAIN") else "gt_frame"
        frame = row_int(row, key)
        if frame is None:
            continue
        queue.append({"frame": frame, "event_type": row.get("event_type", ""),
                      "outcome": outcome, "gt_frame": row_int(row, "gt_frame"),
                      "prediction_frame": row_int(row, "prediction_frame"),
                      "delay_ms": row_float(row, "signed_delay_ms")})
    if not only_unmatched:
        covered = {(e["frame"], e["event_type"]) for e in queue}
        for event in gt:
            if event["uncertain"] and (event["frame"], event["event_type"]) not in covered:
                queue.append({"frame": event["frame"], "event_type": event["event_type"],
                              "outcome": "UNCERTAIN_GT", "gt_frame": event["frame"],
                              "prediction_frame": None, "delay_ms": None})
    return sorted(queue, key=lambda e: (e["frame"], e["event_type"], e["outcome"]))


def marker(canvas, x, y, event_type, color, *, outline=False, size=6):
    """START is round; RELEASE is square. All text uses an adjacent legend."""
    x, y = int(x), int(y)
    if event_type == "PINCH_RELEASE":
        cv2.rectangle(canvas, (x-size, y-size), (x+size, y+size), color,
                      1 if outline else -1, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (x, y), size, color, 1 if outline else -1, cv2.LINE_AA)


def text(img, text_value, x, y, size=0.50, color=None, thickness=1):
    cv2.putText(img, str(text_value), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                size, C["white"] if color is None else color, thickness, cv2.LINE_AA)


def fit_text(value, maximum=47):
    value = str(value)
    return value if len(value) <= maximum else value[:maximum-3] + "..."


def display_time(seconds: float | None) -> str:
    return f"{seconds:.3f} s" if seconds is not None else "N/A"


class Viewer:
    def __init__(self, session: Path, info: dict, frames_setting: int,
                 results_dir: Path, audit: list[dict], gt: list[dict],
                 annotation_status: str, preds: dict[int, dict], initial_frame: int | None = None):
        self.session = session
        self.info = info
        self.clip_id = info["clip_id"]
        self.confirmation = frames_setting
        self.results_dir = results_dir
        self.audit = audit
        self.gt = gt
        self.annotation_status = annotation_status
        self.preds = preds
        self.gt_match, self.pred_match = classify_events(audit)
        self.gt_by_frame = {}
        for item in gt:
            self.gt_by_frame.setdefault(item["frame"], []).append(item)
        self.pred_events = sorted([r for r in preds.values() if r["event"] in EVENT_TYPES],
                                  key=lambda e: e["frame"])
        self.pred_by_frame = {}
        for item in self.pred_events:
            self.pred_by_frame.setdefault(item["frame"], []).append(item)
        self.queue_mode = "unmatched"
        self.queue = make_queue(audit, gt, only_unmatched=True)
        if not self.queue:
            self.queue_mode = "all"
            self.queue = make_queue(audit, gt, only_unmatched=False)
        self.selected = 0
        self.show_landmarks = False
        self.playing = False
        self.window_name = f"Pinch annotation + prediction review | {self.clip_id}"
        self.video_path = inside_session(session, info["video_file"])
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video missing: {self.video_path}")
        self.video = cv2.VideoCapture(str(self.video_path))
        if not self.video.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.n = int(info["frame_count"])
        reported = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported > 0 and reported != self.n:
            self.video.release()
            raise ValueError(f"Video/manifest frame mismatch for {self.clip_id}: "
                             f"video {reported}, manifest {self.n}. Check retake metadata.")
        if self.n <= 0 or len(self.preds) != self.n or set(self.preds) != set(range(self.n)):
            self.video.release()
            raise ValueError(f"Prediction/manifest frame mismatch for {self.clip_id}: "
                             f"predictions {len(self.preds)}, manifest {self.n}.")
        for e in gt:
            if not 0 <= e["frame"] < self.n:
                self.video.release()
                raise ValueError(f"GT event outside video: frame {e['frame']}")
        self.landmarks = None
        path = inside_session(session, info["log_file"])
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                self.landmarks = [json.loads(line) for line in handle if line.strip()]
            if len(self.landmarks) != self.n:
                self.video.release()
                raise ValueError(f"Video/landmark frame mismatch for {self.clip_id}: "
                                 f"landmarks {len(self.landmarks)}, expected {self.n}.")
        else:
            print(f"WARNING: Landmark JSONL missing: {path}. Landmark overlay unavailable.")
        if initial_frame is not None and not (0 <= initial_frame < self.n):
            self.video.release()
            raise ValueError(f"--at-frame must be between 0 and {self.n-1}")
        self.current = initial_frame if initial_frame is not None else (
            min(self.n-1, self.queue[0]["frame"]) if self.queue else 0)
        if self.queue:
            self.selected = min(range(len(self.queue)),
                                key=lambda i: abs(self.queue[i]["frame"] - self.current))
        self.cache_idx = None
        self.cache_img = None
        self.width = None
        self.height = None

    def load_frame(self, index: int) -> np.ndarray:
        if index == self.cache_idx and self.cache_img is not None:
            return self.cache_img.copy()
        # Sequential decoding is much faster than a new seek on each play frame.
        if self.cache_idx is None or index != self.cache_idx + 1:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.video.read()
        if not ok or frame is None:
            raise IOError(f"Cannot decode video frame {index}; check the recording")
        self.cache_idx, self.cache_img = index, frame.copy()
        return frame

    def overlay_landmarks(self, image: np.ndarray, frame_index: int):
        if not self.show_landmarks or self.landmarks is None:
            return
        points = self.landmarks[frame_index].get("landmarks")
        if not points or len(points) != 21:
            return
        h, w = image.shape[:2]
        edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                 (13,17),(17,18),(18,19),(19,20),(0,17)]
        xy = [(round(float(p[0])*w), round(float(p[1])*h)) for p in points]
        for a, b in edges:
            cv2.line(image, xy[a], xy[b], (150, 150, 150), 1, cv2.LINE_AA)
        for point_id, (x, y) in enumerate(xy):
            cv2.circle(image, (x, y), 4 if point_id not in (4,8) else 7,
                       (75, 190, 195) if point_id not in (4,8) else (60, 60, 245),
                       -1, cv2.LINE_AA)
            if point_id in (4, 8):
                text(image, f"{point_id}", x+11, y-6, .57, C["white"], 2)

    def _draw_banners(self, image: np.ndarray, frame: int):
        """Show annotation and prediction labels ON TOP OF the current video frame."""
        manual = self.gt_by_frame.get(frame, [])
        predicted = self.pred_by_frame.get(frame, [])
        h, w = image.shape[:2]
        # Small, non-destructive overlay; source file never modified.
        cv2.rectangle(image, (0, 0), (w, 37), (25, 28, 32), -1)
        if manual or predicted:
            y = 28
            if manual:
                segments = []
                for e in manual:
                    status = self.gt_match.get((frame,e["event_type"]),
                                               "UNCERTAIN" if e["uncertain"] else "UNMATCHED")
                    segments.append(f"{short_type(e['event_type'])} [{status}]")
                line = "MANUAL: " + ", ".join(segments)
                if len(manual) == 1:
                    color = C["start"] if manual[0]["event_type"] == "PINCH_START" else C["release"]
                else:
                    color = C["white"]
                text(image, fit_text(line, 75), 10, y, .57, color, 2)
                y += 30
            if predicted:
                segments = []
                for e in predicted:
                    status = self.pred_match.get((frame, e["event"]), "UNMATCHED")
                    segments.append(f"{short_type(e['event'])} [{status}]")
                line = "DETECTOR: " + ", ".join(segments)
                cv2.rectangle(image, (0, y-23), (w, y+8), (25, 28, 32), -1)
                color = C["fp"] if any(self.pred_match.get((frame,e["event"])) == "FP" for e in predicted) else C["tp"]
                text(image, fit_text(line, 75), 10, y, .57, color, 2)
        else:
            text(image, "NO EVENT ON CURRENT FRAME", 10, 27, .48, C["muted"])

    def _timeline(self, width: int, y_offset: int, canvas: np.ndarray, frame: int):
        """Two complete-frame lanes, one for each source of events."""
        left, right = 104, max(120, width-26)
        span = max(1, self.n-1)
        top_y, bottom_y = y_offset + 42, y_offset + 87
        cv2.rectangle(canvas, (0, y_offset), (width, y_offset+121), C["background"], -1)
        for y, label in ((top_y, "MANUAL GT"), (bottom_y, "DETECTOR")):
            text(canvas, label, 10, y+5, .37, C["muted"])
            cv2.line(canvas, (left,y), (right,y), (91, 97, 105), 1, cv2.LINE_AA)
        def position(i):
            return left + round((right-left)*i/span)
        for e in self.gt:
            x = position(e["frame"])
            kind = e["event_type"]
            color = C["uncertain"] if e["uncertain"] else (C["start"] if kind == "PINCH_START" else C["release"])
            marker(canvas, x, top_y, kind, color, size=5)
            if self.gt_match.get((e["frame"],kind)) == "FN":
                cv2.circle(canvas, (x,top_y), 9, C["fn"], 2, cv2.LINE_AA)
        for e in self.pred_events:
            x = position(e["frame"])
            status = self.pred_match.get((e["frame"],e["event"]), "UNKNOWN")
            if status == "FP":
                cv2.line(canvas, (x-6,bottom_y-6), (x+6,bottom_y+6), C["fp"], 2, cv2.LINE_AA)
                cv2.line(canvas, (x-6,bottom_y+6), (x+6,bottom_y-6), C["fp"], 2, cv2.LINE_AA)
            else:
                color = C["tp"] if status == "TP" else C["uncertain"]
                marker(canvas, x, bottom_y, e["event"], color, outline=True, size=6)
        x_cursor = position(frame)
        cv2.line(canvas, (x_cursor, y_offset+17), (x_cursor,y_offset+108), C["cursor"], 1,cv2.LINE_AA)
        cv2.line(canvas, (x_cursor-5, y_offset+17), (x_cursor+5,y_offset+17), C["cursor"],1,cv2.LINE_AA)
        text(canvas, "0 s",left,y_offset+116,.38,C["muted"])
        end_t = self.preds[self.n-1]["time_s"]
        text(canvas, f"{display_time(end_t)}",max(left+20,right-82),y_offset+116,.38,C["muted"])

    def _panel(self, canvas: np.ndarray, x: int, frame: int):
        y = 30
        def line(msg, color=None, size=.49, gap=20):
            nonlocal y
            text(canvas, fit_text(msg, 52), x+15, y, size, color or C["white"])
            y+=gap
        line("GROUND TRUTH + DETECTOR", C["white"], .61, 32)
        line(f"Clip: {fit_text(self.clip_id, 43)}",C["muted"],.47)
        line(f"Annotation status: {self.annotation_status}", C["start"] if self.annotation_status == "COMPLETE" else C["fp"])
        line(f"Manual annotations: {len(self.gt)}", C["start"])
        line(f"Detector events: {len(self.pred_events)}", C["tp"])
        counts = Counter(r["outcome"] for r in self.audit)
        line(f"Matched TP: {counts['TP']}   FP: {counts['FP']}   FN: {counts['FN']}", C["white"])
        y+=10
        line(f"Frame: {frame} / {self.n-1}  (zero-based)")
        cur=self.preds[frame]
        line(f"Capture time: {display_time(cur['time_s'])}")
        line(f"Detector state: {cur['state']}")
        line(f"MediaPipe hand: {'YES' if cur['hand_detected'] else 'NO'}")
        def ratio(r):
            return f"{r:.3f}" if r is not None else "N/A"
        line(f"Raw ratio: {ratio(cur['raw_ratio'])}")
        line(f"Smoothed ratio: {ratio(cur['smoothed_ratio'])}")
        y+=9
        line("CURRENT FRAME",C["white"],.58,23)
        manually = self.gt_by_frame.get(frame, [])
        predicted = self.pred_by_frame.get(frame, [])
        if not manually:
            line("GT: none", C["muted"])
        for e in manually:
            status = self.gt_match.get((e["frame"],e["event_type"]), "UNCERTAIN" if e["uncertain"] else "UNMATCHED")
            line(f"GT {short_type(e['event_type'])} [{status}]", C["start"] if e["event_type"] == "PINCH_START" else C["release"])
        if not predicted:
            line("Predicted: none", C["muted"])
        for e in predicted:
            status = self.pred_match.get((e["frame"],e["event"]), "UNMATCHED")
            line(f"Pred {short_type(e['event'])} [{status}]", C["fp"] if status == "FP" else C["tp"])
        if self.queue:
            y+=10
            line(f"SELECTED ({self.queue_mode.upper()}): {self.selected+1}/{len(self.queue)}", C["white"],.50,23)
            event=self.queue[self.selected]
            outcome=event["outcome"]
            color=C["fp"] if outcome in ("FP","FN") else C["tp"]
            line(f"{outcome} {short_type(event['event_type'])} at frame {event['frame']}",color,.50)
            if outcome == "TP" and event["delay_ms"] is not None:
                line(f"Matched delay: {event['delay_ms']:+.1f} ms", C["muted"])
            elif outcome in ("FP", "FN"):
                # Diagnostic neighbour, NOT a replacement for one-to-one matching.
                kind = event["event_type"]
                if outcome == "FP":
                    same_type = [g for g in self.gt if g["event_type"] == kind]
                    reference = self.preds[event["frame"]]["time_s"]
                    candidates = [(abs(g["time_s"]-reference),g["frame"],g["time_s"])
                                  for g in same_type if g["time_s"] is not None]
                    label = "Nearest GT"
                else:
                    same_type = [p for p in self.pred_events if p["event"] == kind]
                    reference = self.preds[event["frame"]]["time_s"]
                    candidates = [(abs(p["time_s"]-reference),p["frame"],p["time_s"])
                                  for p in same_type if p["time_s"] is not None]
                    label = "Nearest pred"
                if candidates:
                    dt, adjacent_frame, other_t = min(candidates)
                    line(f"{label}: frame {adjacent_frame} ({dt*1000:.0f} ms)", C["muted"],.44,19)
        y+=10
        line("ANNOTATION / PREDICTION KEY",C["white"],.47,23)
        line("GREEN circle = manual START",C["start"],.43,18)
        line("BLUE square = manual RELEASE",C["release"],.43,18)
        line("RED circle around manual = FN",C["fp"],.43,18)
        line("YELLOW hollow = matched prediction",C["tp"],.43,18)
        line("RED X = unmatched prediction (FP)",C["fp"],.43,18)
        line("GREY = uncertain / ignored",C["uncertain"],.43,18)
        y+=8
        line("n/p: next/previous event",C["white"],.44,18)
        line("t: unmatched / all events",C["white"],.44,18)
        line("a/d: +/-1 frame    j/l: +/-10 frames",C["white"],.44,18)
        line("c: event    Space: play/pause",C["white"],.44,18)
        line("m: landmarks    s: save screenshot",C["white"],.44,18)
        line("q/Esc: quit",C["white"],.44,18)
        if self.annotation_status != "COMPLETE":
            line("WARNING: annotations not finalised",C["fp"],.46,18)

    def render(self, frame_number: int) -> np.ndarray:
        frame_number = max(0,min(self.n-1,int(frame_number)))
        image=self.load_frame(frame_number)
        self.overlay_landmarks(image, frame_number)
        self._draw_banners(image,frame_number)
        h,w=image.shape[:2]
        scaled=min(1.0,840/max(1,w),510/max(1,h))
        w2,h2=max(1,round(w*scaled)),max(1,round(h*scaled))
        image=cv2.resize(image,(w2,h2),interpolation=cv2.INTER_AREA)
        panel_width=430
        video_height=max(h2+135,790)
        canvas=np.full((video_height,w2+panel_width,3),C["panel"],dtype=np.uint8)
        canvas[:h2,:w2]=image
        self._timeline(w2,h2,canvas,frame_number)
        self._panel(canvas,w2,frame_number)
        return canvas

    def _switch_queue(self):
        next_mode="all" if self.queue_mode == "unmatched" else "unmatched"
        new=make_queue(self.audit,self.gt,only_unmatched=(next_mode == "unmatched"))
        if not new:
            print(f"No {next_mode} events to navigate")
            return
        self.queue_mode=next_mode
        self.queue=new
        self.selected=min(range(len(new)),key=lambda i:abs(new[i]["frame"]-self.current))
        print(f"Navigation mode: {self.queue_mode} ({len(self.queue)} events)")

    def _snapshot(self, canvas: np.ndarray):
        target=self.results_dir/"inspection_snapshots"
        target.mkdir(parents=True,exist_ok=True)
        safe=re.sub(r"[^A-Za-z0-9_.-]+","_",self.clip_id)
        stem=f"{safe}_{self.confirmation}f_frame_{self.current:05d}"
        if self.queue:
            stem+=f"_{self.queue[self.selected]['outcome']}"
        dest=target/(stem+".png")
        number=2
        while dest.exists():
            dest=target/f"{stem}_{number}.png"
            number+=1
        if not cv2.imwrite(str(dest),canvas):
            raise OSError(f"Failed to save {dest}")
        print(f"Saved inspection screenshot: {dest}")

    def run(self):
        print(f"Viewing {self.clip_id} ({self.n} frames, annotations={len(self.gt)}, "
              f"status={self.annotation_status}; {self.confirmation}-frame replay)")
        if self.annotation_status != "COMPLETE":
            print("WARNING: ground truth not marked COMPLETE. Do not treat missing annotations as negatives.")
        if not self.gt and self.annotation_status == "COMPLETE":
            print("This clip has ZERO ground truth event labels (valid for a reviewed negative-control video).")
        if not self.queue:
            print("No matched or unmatched events in the audit. You can still step through frames.")
        print("Controls: n/p unmatched events, t toggle all events, a/d single frame, "
              "j/l 10 frames, c selected event, Space play, m landmarks, s screenshot, q quit")
        try:
            cv2.namedWindow(self.window_name,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name,1260,740)
            while True:
                canvas=self.render(self.current)
                cv2.imshow(self.window_name,canvas)
                key=cv2.waitKeyEx(35 if self.playing else 0)
                if key == -1 and self.playing:
                    if self.current < self.n-1:
                        self.current+=1
                    else:
                        self.playing=False
                    continue
                if key in (27,ord('q')):
                    break
                if key == ord(' '):
                    self.playing=not self.playing
                elif key == ord('t'):
                    self.playing=False
                    self._switch_queue()
                elif key in (ord('n'),ord('p')):
                    if self.queue:
                        delta=1 if key == ord('n') else -1
                        self.selected=(self.selected+delta)%len(self.queue)
                        self.current=min(self.n-1,self.queue[self.selected]['frame'])
                        self.playing=False
                elif key in (ord('a'),2424832,65361):
                    self.current=max(0,self.current-1)
                    self.playing=False
                elif key in (ord('d'),2555904,65363):
                    self.current=min(self.n-1,self.current+1)
                    self.playing=False
                elif key == ord('j'):
                    self.current=max(0,self.current-10)
                    self.playing=False
                elif key == ord('l'):
                    self.current=min(self.n-1,self.current+10)
                    self.playing=False
                elif key == ord('c'):
                    if self.queue:
                        self.current=min(self.n-1,self.queue[self.selected]['frame'])
                        self.playing=False
                elif key == ord('m'):
                    self.show_landmarks=not self.show_landmarks
                elif key == ord('s'):
                    self._snapshot(canvas)
        finally:
            self.video.release()
            cv2.destroyAllWindows()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session",required=True,help="Benchmark session, e.g. session02")
    p.add_argument("--clip-id",default="",help="Clip ID to open in video viewer")
    p.add_argument("--frames",type=int,default=3,help="Confirmation-window setting (default 3)")
    p.add_argument("--results-dir",default="results",help="Subdirectory containing audit and replay CSVs")
    p.add_argument("--list-only",action="store_true",help="List all unmatched FP/FN without opening video")
    p.add_argument("--at-frame",type=int,default=None,help="Open a specific zero-based video frame")
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.frames<1:
        raise SystemExit("--frames must be >= 1")
    session=Path(args.session).expanduser().resolve()
    if not session.is_dir():
        raise SystemExit(f"Session does not exist: {session}")
    results=inside_session(session,args.results_dir)
    manifest=get_manifest(session)
    if args.list_only and not args.clip_id:
        audits=read_csv(results/"event_matching_audit.csv")
        rows=[r for r in audits if row_int(r,"confirmation_frames")==args.frames
              and r.get("outcome") in ("FP","FN")]
        print("CLIP                                     RESULT  EVENT          FRAME     t(s)")
        print("-"*86)
        for r in sorted(rows,key=lambda v:(v["clip_id"],
                                 row_int(v,"prediction_frame" if v["outcome"]=="FP" else "gt_frame") or 0)):
            fld="prediction_frame" if r["outcome"]=="FP" else "gt_frame"
            tfld="prediction_time_s" if r["outcome"]=="FP" else "gt_time_s"
            print(f"{r['clip_id'][:40]:40} {r['outcome']:6}  {short_type(r['event_type']):13} "
                  f"{r[fld]:>5} {float(r[tfld]):8.3f}")
        print(f"\nUnmatched events at {args.frames} frames: {len(rows)}")
        return
    if not args.clip_id:
        raise SystemExit("Choose --clip-id to open a recording, or use --list-only for all clips")
    if args.clip_id not in manifest:
        raise SystemExit(f"Clip ID {args.clip_id!r} not found in {session/'manifest.csv'}")
    audit=audit_rows(session,results,args.clip_id,args.frames)
    gt,status=annotation_rows(session,args.clip_id)
    preds=load_predictions(results,args.clip_id,args.frames)
    if args.list_only:
        q=make_queue(audit,gt,only_unmatched=True)
        print(f"Clip: {args.clip_id} | Annotation status: {status} | "
              f"Manual events: {len(gt)} | Unmatched: {len(q)}")
        for e in q:
            pred=preds.get(e["frame"],{})
            print(f"  {e['outcome']:2} {e['event_type']:13} "
                  f"frame {e['frame']:5d}   capture time {display_time(pred.get('time_s'))}")
        return
    viewer=Viewer(session,manifest[args.clip_id],args.frames,results,audit,gt,status,preds,args.at_frame)
    viewer.run()


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError,ValueError,OSError,RuntimeError,KeyError,cv2.error) as exc:
        raise SystemExit(f"Pinch inspection failed: {exc}") from exc
