"""Webcam test for pinch gestures with live processing timings."""

import argparse
import csv
import os
import time
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from pinch_detector import PinchDetector, THUMB, INDEX, FRAME_WIDTH, FRAME_HEIGHT

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_FILE = Path(__file__).resolve().parent / "hand_landmarker.task"
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
         (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
TIMING_KEYS = ("capture", "preprocess", "landmarks", "gesture", "frame_to_gesture", "full_loop")


def average(history, key):
    values = history[key]
    return sum(values) / len(values) if values else 0.0


def ensure_model():
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 100_000:
        return MODEL_FILE
    print("Downloading MediaPipe hand model...")
    try:
        urlretrieve(MODEL_URL, MODEL_FILE)
    except Exception as exc:
        MODEL_FILE.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download model: {MODEL_URL}") from exc
    return MODEL_FILE


def open_camera(index):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not camera.isOpened() and os.name == "nt":
        camera.release()
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera {index}. Try --camera 0 or --camera 1.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return camera


def label(frame, text, x, y, size=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


def draw_hand(frame, landmarks, gesture):
    if landmarks is None:
        return
    height, width = frame.shape[:2]
    points = [(round(p.x * width), round(p.y * height)) for p in landmarks]
    colour = (0, 60, 255) if gesture["state"] == "PINCHED" else (0, 220, 70)
    for start, end in EDGES:
        cv2.line(frame, points[start], points[end], (170, 170, 170), 2)
    for point in points:
        cv2.circle(frame, point, 3, (225, 225, 225), -1)
    cv2.line(frame, points[THUMB], points[INDEX], colour, 3)
    for tip in (THUMB, INDEX):
        cv2.circle(frame, points[tip], 9, (255, 170, 0), -1)


def draw_overlay(frame, gesture, fps, last_event, pinch, release, timing, history, event_point=None):
    height, width = frame.shape[:2]
    state, cursor, ratio = gesture["state"], gesture["position"], gesture["ratio"]
    colour = (0, 60, 255) if state == "PINCHED" else (0, 220, 70)
    cursor_text = f"Pinch point: {cursor[0]}, {cursor[1]}" if cursor else "Pinch point: -"
    ratio_text = f"{ratio:.4f}" if ratio is not None else "-"

    if cursor:
        cv2.circle(frame, cursor, 15, colour, 3)
        cv2.circle(frame, cursor, 3, colour, -1)
    if event_point is not None:
        cv2.circle(frame, event_point, 11, (0, 255, 255), 2)
        cv2.circle(frame, event_point, 3, (0, 255, 255), -1)

    cv2.rectangle(frame, (0, 0), (width, 125), (30, 30, 30), -1)
    label(frame, f"GESTURE: {state}", 15, 31, 0.85, colour, 2)
    label(frame, f"Pinch ratio: {ratio_text}   Close < {pinch:.2f}   Release > {release:.2f}", 15, 64, 0.6)
    label(frame, f"{cursor_text}    FPS: {fps:.0f}    Q / Esc: quit", 15, 92, 0.6)
    if last_event:
        location = f" @ {event_point}" if event_point is not None else ""
        label(frame, f"EVENT: {last_event}{location}", 15, 119, 0.55, (0, 220, 255), 2)

    panel_width, panel_height = min(385, width - 10), 180
    x = width - panel_width - 5
    y = min(136, max(0, height - panel_height - 5))
    layer = frame.copy()
    cv2.rectangle(layer, (x, y), (width - 5, y + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(layer, 0.78, frame, 0.22, 0, frame)
    label(frame, "TIMINGS            Current    Avg 60", x + 10, y + 23, 0.49)

    names = (("capture", "Camera read"), ("preprocess", "Preprocess"),
             ("landmarks", "MediaPipe"), ("gesture", "Gesture decision"),
             ("frame_to_gesture", "FRAME TO GESTURE"))
    for row, (key, name) in enumerate(names):
        color = (0, 235, 235) if key == "frame_to_gesture" else (225, 225, 225)
        unit, scale = ("us", 1000) if key == "gesture" else ("ms", 1)
        label(frame, name, x + 10, y + 49 + row * 25, 0.44, color)
        label(frame, f"{timing[key] * scale:6.1f} {average(history, key) * scale:6.1f} {unit}", width - 155, y + 49 + row * 25, 0.42, color)


def main():
    parser = argparse.ArgumentParser(description="Webcam pinch detection and timings")
    parser.add_argument("--camera", type=int, default=1, help="Webcam index (default: 1)")
    parser.add_argument("--pinch", type=float, default=0.25, help="Pinch threshold")
    parser.add_argument("--release", type=float, default=0.30, help="Release threshold")
    parser.add_argument("--frames", type=int, default=3, help="Frames required for a state change")
    parser.add_argument("--timing-csv", help="Save per-frame timings to a CSV")
    args = parser.parse_args()
    if not 0 < args.pinch < args.release or args.frames < 1:
        parser.error("Require 0 < --pinch < --release and --frames >= 1")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    detector = PinchDetector(args.pinch, args.release, args.frames)
    camera = open_camera(args.camera)
    history = {key: deque(maxlen=60) for key in TIMING_KEYS}
    start = previous_frame = last_summary = time.monotonic()
    last_timestamp, last_event, event_at, fps, frame_number = -1, "", 0.0, 0.0, 0
    last_event_point = None
    csv_file = None

    try:
        if args.timing_csv:
            csv_file = open(args.timing_csv, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(csv_file, fieldnames=["frame", "state", "event", "cursor_x", "cursor_y", "event_x", "event_y"] + [f"{key}_ms" for key in TIMING_KEYS])
            writer.writeheader()

        print("Webcam ready. Pinch point is one-third from thumb to index. Press Q or Esc to quit.")
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as model:
            while True:
                t0 = time.perf_counter()
                ok, frame = camera.read()
                t1 = time.perf_counter()
                if not ok:
                    print("Cannot read a webcam frame.")
                    break

                if frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(last_timestamp + 1, int((time.monotonic() - start) * 1000))
                last_timestamp = timestamp
                t2 = time.perf_counter()

                result = model.detect_for_video(image, timestamp)
                t3 = time.perf_counter()
                landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
                gesture = detector.update(landmarks)
                t4 = time.perf_counter()

                timing = dict(capture=(t1 - t0) * 1000, preprocess=(t2 - t1) * 1000,
                              landmarks=(t3 - t2) * 1000, gesture=(t4 - t3) * 1000,
                              frame_to_gesture=(t4 - t1) * 1000)
                for key, value in timing.items():
                    history[key].append(value)
                frame_number += 1

                now = time.monotonic()
                elapsed = now - previous_frame
                fps = 0.8 * fps + 0.2 / elapsed if elapsed > 0 else fps
                previous_frame = now
                if gesture["event"]:
                    last_event, event_at, last_event_point = gesture["event"], now, gesture["event_position"]
                    ratio_text = f"{gesture['ratio']:.4f}" if gesture["ratio"] is not None else "-"
                    print(f"{last_event}: event_point={last_event_point} cursor={gesture['position']} ratio={ratio_text} latency={timing['frame_to_gesture']:.1f} ms")

                draw_hand(frame, landmarks, gesture)
                recent_event = now - event_at < 1.2
                draw_overlay(frame, gesture, fps, last_event if recent_event else "", args.pinch, args.release, timing, history, last_event_point if recent_event else None)
                cv2.imshow("Pinch gesture test with timing", frame)
                key = cv2.waitKey(1) & 0xFF
                timing["full_loop"] = (time.perf_counter() - t0) * 1000
                history["full_loop"].append(timing["full_loop"])

                if csv_file:
                    cursor, event_point = gesture["position"], gesture["event_position"]
                    row = {"frame": frame_number, "state": gesture["state"], "event": gesture["event"] or "",
                           "cursor_x": cursor[0] if cursor else "", "cursor_y": cursor[1] if cursor else "",
                           "event_x": event_point[0] if event_point else "", "event_y": event_point[1] if event_point else ""}
                    row.update({f"{key}_ms": round(timing[key], 3) for key in TIMING_KEYS})
                    writer.writerow(row)

                if now - last_summary >= 5:
                    print(f"Avg 60: camera={average(history, 'capture'):.1f} ms, "
                          f"MediaPipe={average(history, 'landmarks'):.1f} ms, "
                          f"gesture={average(history, 'gesture') * 1000:.1f} us, "
                          f"frame-to-gesture={average(history, 'frame_to_gesture'):.1f} ms, "
                          f"full-loop={average(history, 'full_loop'):.1f} ms")
                    last_summary = now
                if key in (27, ord("q")):
                    break
    finally:
        camera.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
