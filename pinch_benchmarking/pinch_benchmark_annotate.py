"""Annotate physical thumb-index contact/separation on ORIGINAL raw video.

GUI controls (shown beside the video, not over the hand):
 a/d = back/forward 1 frame; j/l = back/forward 10; -/= = 30 frames
 s = visible thumb-index contact (PINCH_START)
 r = visible fingertip separation (PINCH_RELEASE)
 u = toggle uncertainty for events at current frame; x = delete events here
 v = hand visibly PRESENT starting here; n = visibly ABSENT; b = UNKNOWN
 Space = play/pause, g = first frame, e = last frame
 f = mark this clip's annotations COMPLETE, q/Esc = save and quit

A clip is NEVER evaluated merely because it has zero event labels: the reviewer
must explicitly press F. Video display does NOT include detector predictions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from pinch_benchmark_common import (
    GROUND_TRUTH_FIELDS, STATUS_FIELDS, clean_clip_id,
    manifest_by_id, read_csv, read_jsonl, replace_clip_rows,
    session_file, utc_now, visibility_breakpoints, visibility_for_frame,
    write_visibility_breakpoints,
)


def events_for_clip(session: Path, clip_id: str):
    events = []
    for item in read_csv(session / "ground_truth_events.csv"):
        if item.get("clip_id") == clip_id:
            events.append({"clip_id": clip_id, "event_id": item.get("event_id", ""),
                           "event_type": item["event_type"], "frame_index": int(item["frame_index"]),
                           "time_s": float(item["time_s"]),
                           "uncertain": item.get("uncertain", "").lower() in ("1", "true", "yes"),
                           "notes": item.get("notes", "")})
    return sorted(events, key=lambda e: (e["frame_index"], e["event_type"]))


def save_events(session: Path, clip_id: str, events):
    ordered = sorted(events, key=lambda e: (e["frame_index"], e["event_type"]))
    for i, item in enumerate(ordered, start=1):
        item["event_id"] = f"{clip_id}_{i:03d}"
    replace_clip_rows(session / "ground_truth_events.csv", GROUND_TRUTH_FIELDS, clip_id, ordered)


def set_status(session: Path, clip_id: str, status: str, notes: str = ""):
    replace_clip_rows(session / "annotation_status.csv", STATUS_FIELDS, clip_id,
                      [{"clip_id": clip_id, "status": status,
                        "reviewed_at_utc": utc_now(), "notes": notes}])


def draw_panel(raw, frame, nframes, time_s, events, visibility, playing,
               status, scale, video_fps, clip_id):
    """The annotation UI uses an EXTERNAL side panel and never modifies raw data."""
    height, width = raw.shape[:2]
    panel_width = 450
    canvas = np.full((height, width + panel_width, 3), (25, 28, 30), np.uint8)
    canvas[:, :width] = raw
    x = width + 16
    y = 32

    def line(text, color=(245,245,245), sz=0.55, step=26):
        nonlocal y
        cv2.putText(canvas, str(text)[:52], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    sz, color, 1, cv2.LINE_AA)
        y += step

    line("GROUND TRUTH ONLY | NO PREDICTIONS", (70,220,250), 0.55, 40)
    line(f"Clip: {clip_id[:37]}")
    line(f"Frame: {frame+1} / {nframes} (index {frame})")
    line(f"Capture time: {time_s:.4f} s")
    line(f"Video nominal FPS: {video_fps:g}")
    line(f"Annotation: {status}", (70,220,250), step=40)
    line(f"Hand visibility: {visibility}")
    at_frame = [e for e in events if e["frame_index"] == frame]
    if not at_frame:
        line("Event: none")
    for e in at_frame:
        line(f"Event: {e['event_type']}", (30,230,150))
        if e["uncertain"]:
            line("        UNCERTAIN (excluded by default)", (40,180,255))
    y += 22
    for text in (
        "a / d             previous / next frame",
        "j / l             -10 / +10 frames",
        "- / =             -30 / +30 frames",
        "g / e             first / last frame",
        "SPACE             play / pause",
        "s                  label PINCH_START",
        "r                  label PINCH_RELEASE",
        "u                  toggle uncertain at frame",
        "x                  delete labels at frame",
        "v / n / b          hand PRESENT/ABSENT/UNKNOWN",
        "f                  FINALISE reviewed clip",
        "q / Esc            save and quit",
    ):
        line(text, step=24, sz=0.50)
    y += 10
    line("Use the first visible CONTACT / SEPARATION", (170,220,210), sz=.45)
    line("Frame index maps to capture time in JSONL", (170,220,210), sz=.45)
    if playing:
        cv2.putText(canvas, "PLAY", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (70,220,70), 2)
    if scale != 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True)
    p.add_argument("--clip-id", required=True)
    p.add_argument("--scale", type=float, default=0.77, help="GUI scale, e.g. 0.65 for smaller screens")
    p.add_argument("--assume-visible", action="store_true", help="Start with all frames labelled HAND PRESENT; REVIEW the full clip before finalising")
    args = p.parse_args()
    if not 0.2 <= args.scale <= 2:
        p.error("--scale must be between 0.2 and 2.0")
    clip = clean_clip_id(args.clip_id)
    session = Path(args.session).expanduser().resolve()
    manifest = manifest_by_id(session)
    if clip not in manifest:
        p.error(f"Clip {clip!r} is not in {session / 'manifest.csv'}")
    info = manifest[clip]
    if info["status"] != "COMPLETE":
        p.error(f"Capture status {info['status']}; inspect/recapture before annotation")
    rows = read_jsonl(session_file(session, info["log_file"]))
    nframes = len(rows)
    if int(info["frame_count"]) != nframes:
        p.error("Manifest frame count does not match JSONL. Cannot align annotations")
    video = cv2.VideoCapture(str(session_file(session, info["video_file"])))
    if not video.isOpened():
        p.error("Cannot open raw video")
    reported = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported > 0 and reported != nframes:
        video.release()
        p.error(f"Video reports {reported} frames; JSONL has {nframes}. Recheck capture")
    events = events_for_clip(session, clip)
    breaks = visibility_breakpoints(session, clip)
    if not breaks:
        breaks = {0: "PRESENT" if args.assume_visible else "UNKNOWN"}
        write_visibility_breakpoints(session, clip, breaks, nframes)
    status_map = {r["clip_id"]: r["status"] for r in read_csv(session / "annotation_status.csv")}
    status = status_map.get(clip, "DRAFT")
    if clip not in status_map:
        set_status(session, clip, status)
    print(f"Annotating {clip} | {nframes} frames | zero-based frame indices")
    print("Predictions are NOT displayed. Press F only after reviewing the entire clip.")

    current = 0
    playing = False
    cached_idx, cached_frame = -1, None

    def show_frame(index):
        nonlocal cached_idx, cached_frame
        if index == cached_idx and cached_frame is not None:
            return cached_frame
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, raw = video.read()
        if not ok or raw is None:
            raise IOError(f"Cannot decode frame {index}; annotate a different/repaired recording")
        cached_idx, cached_frame = index, raw
        return raw

    def save_draft(changed=True):
        nonlocal status
        save_events(session, clip, events)
        write_visibility_breakpoints(session, clip, breaks, nframes)
        if changed:
            status = "DRAFT"
            set_status(session, clip, status)

    try:
        while True:
            raw = show_frame(current)
            vis = visibility_for_frame(breaks, current)
            panel = draw_panel(raw, current, nframes, float(rows[current]["time_s"]),
                               events, vis, playing, status, args.scale,
                               float(info.get("video_fps_nominal") or 30.0), clip)
            cv2.imshow("Pinch ground truth annotation", panel)
            key = cv2.waitKeyEx(40 if playing else 0)
            if playing and key == -1:
                if current < nframes-1:
                    current += 1
                else:
                    playing = False
                continue
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                playing = not playing
                continue
            if key in (ord("a"), 2424832, 65361):
                current = max(0, current-1)
            elif key in (ord("d"), 2555904, 65363):
                current = min(nframes-1, current+1)
            elif key == ord("j"):
                current = max(0, current-10)
            elif key == ord("l"):
                current = min(nframes-1, current+10)
            elif key == ord("-"):
                current = max(0, current-30)
            elif key in (ord("="), ord("+")):
                current = min(nframes-1, current+30)
            elif key == ord("g"):
                current = 0
            elif key == ord("e"):
                current = nframes-1
            elif key in (ord("s"), ord("r")):
                which = "PINCH_START" if key == ord("s") else "PINCH_RELEASE"
                if not any(e["event_type"] == which and e["frame_index"] == current for e in events):
                    events.append({"clip_id": clip, "event_id": "", "event_type": which,
                                   "frame_index": current, "time_s": float(rows[current]["time_s"]),
                                   "uncertain": False, "notes": ""})
                    save_draft()
            elif key == ord("u"):
                found = False
                for event in events:
                    if event["frame_index"] == current:
                        event["uncertain"] = not event["uncertain"]
                        found = True
                if found:
                    save_draft()
            elif key == ord("x"):
                existing = len(events)
                events[:] = [e for e in events if e["frame_index"] != current]
                if len(events) != existing:
                    save_draft()
            elif key in (ord("v"), ord("n"), ord("b")):
                kind = {ord("v"): "PRESENT", ord("n"): "ABSENT", ord("b"): "UNKNOWN"}[key]
                breaks[current] = kind
                save_draft()
            elif key == ord("f"):
                starts = sum(e["event_type"] == "PINCH_START" for e in events)
                releases = sum(e["event_type"] == "PINCH_RELEASE" for e in events)
                target = int(info.get("expected_cycles") or 0)
                if info["experiment"].lower() == "negative" and events:
                    print("WARNING: Negative-control clip contains labelled physical contact/separation; "
                          "it cannot be scored as an uncontaminated negative trial")
                if info["experiment"].lower() != "negative" and target and (starts != target or releases != target):
                    print(f"ANNOTATION CHECK: planned {target} full cycles, "
                          f"but labelled {starts} contacts and {releases} releases. "
                          "Verify the clip was fully annotated; report any truncated trial")
                # Explicit confirmation prevents missing annotations from being
                # silently treated as negatives. A 0-event negative clip is valid.
                save_draft(changed=False)
                status = "COMPLETE"
                set_status(session, clip, status, "Events independently reviewed against raw video")
                print(f"FINALIZED {clip}: {len(events)} physical events. Hand visibility: reviewed labels")
                playing = False
    except (OSError, KeyboardInterrupt) as exc:
        print(f"Annotation interrupted: {exc}", file=sys.stderr)
    finally:
        save_draft(changed=False)
        video.release()
        cv2.destroyAllWindows()
        print(f"Saved {clip} annotations, status={status}. Event count={len(events)}")


if __name__ == "__main__":
    main()
