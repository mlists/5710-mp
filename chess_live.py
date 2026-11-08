"""Live video chess projection editor with pinch (hand) control.

Pinch over a piece to pick it up, move your hand, release over a square to drop it.
Mouse editing still works as before.

    python chess_live.py
    python chess_live.py --source 1 --frames 2
"""
import argparse
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from urllib.request import urlretrieve

import cv2
import numpy as np
import mediapipe as mp

import real_board_scene as pc
from real_gui import (
    App, BoardPicker, start_position, starting_castling_rights,
)
from pinch_detector import PinchDetector

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_FILE = Path(__file__).resolve().parent / "hand_landmarker.task"

# Pinch detector settings (single source for the CLI and LiveApp)
DEFAULT_PINCH = 0.30      # pinch close threshold
DEFAULT_RELEASE = 0.35    # pinch release threshold
DEFAULT_FRAMES = 3        # frames needed to confirm a change


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


# Hand occlusion tuning, as fractions of palm size (wrist to middle-finger base).
# Make the mask thinner by lowering these; set HAND_ARM_LENGTH = 0 to ignore the forearm.
HAND_FINGER_WIDTH = 0.17
HAND_THUMB_WIDTH = 0.21
HAND_ARM_WIDTH = 0.60
HAND_ARM_LENGTH = 0.45
HAND_FEATHER = 0.05

# Hide the pieces after this many consecutive pose failures of ANY kind
# (RuntimeError "relative gate FAIL", OrientationUnverified, BoardLost, ...).
# 1 = hide on the first failure. Raise it (e.g. 3) to ride out single bad frames.
HIDE_AFTER_FAILURES = 1

FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def hand_mask(landmarks, width, height):
    """Soft (h, w) float32 mask of the hand built from 21 MediaPipe landmarks.

    Palm = filled hull, fingers = thick round-ended lines, plus a short forearm
    extension from the wrist. Edges are feathered. Returns None if degenerate.
    """
    pts = np.array([(x * width, y * height) for x, y in landmarks], dtype=np.float32)
    palm = max(float(np.linalg.norm(pts[0] - pts[9])),
               float(np.linalg.norm(pts[5] - pts[17])), 1.0)
    if not np.isfinite(pts).all() or palm < 8:
        return None

    mask = np.zeros((height, width), np.uint8)
    p = lambda i: (int(round(pts[i][0])), int(round(pts[i][1])))

    hull = cv2.convexHull(np.array([p(i) for i in (0, 1, 2, 5, 9, 13, 17)], np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    for n, chain in enumerate(FINGERS):
        thick = max(3, int(round(palm * (HAND_THUMB_WIDTH if n == 0 else HAND_FINGER_WIDTH))))
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(mask, p(a), p(b), 255, thick, cv2.LINE_AA)
        for i in chain:
            cv2.circle(mask, p(i), thick // 2, 255, -1, cv2.LINE_AA)

    # Forearm: continue from the middle-finger base through the wrist, away from the fingers
    direction = pts[0] - pts[9]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    if HAND_ARM_LENGTH > 0:
        end = pts[0] + direction * palm * HAND_ARM_LENGTH
        cv2.line(mask, p(0), (int(round(end[0])), int(round(end[1]))), 255,
                 max(3, int(round(palm * HAND_ARM_WIDTH))), cv2.LINE_AA)

    k = max(3, int(round(palm * HAND_FEATHER))) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255


class PoseWorker(threading.Thread):
    """Runs board detection + pose solving in the background.

    solve_once() holds the solve/failure logic, so the background loop and the
    in-line (sync) path used when saving a video file behave identically.
    """

    def __init__(self, perf=None):
        super().__init__(daemon=True)
        self.perf = perf
        self.lock = threading.Lock()
        self.pending = None
        self.result = (None, None)        # (pose, picker) from the last good solve
        self.error = ""
        self.solve_ms = 0.0
        self.running = True
        self.failures = 0
        self.tracker = pc.LivePoseTracker()   # detect once, then optical flow

    def submit(self, frame):
        with self.lock:
            self.pending = frame

    def latest(self):
        with self.lock:
            return self.result

    def solve_once(self, frame):
        """Solve one frame and return the current (pose, picker).

        On failure the previous result is kept until HIDE_AFTER_FAILURES
        consecutive failures (or BoardLost), after which it becomes (None, None)
        so no pieces are drawn.
        """
        t = time.perf_counter()
        try:
            pose = self.tracker.solve(frame)
            picker = BoardPicker(pose)
            with self.lock:
                self.result = (pose, picker)
                self.error = ""
                self.failures = 0
        except Exception as exc:
            with self.lock:
                self.failures += 1
                if isinstance(exc, pc.BoardLost) or self.failures >= HIDE_AFTER_FAILURES:
                    self.result = (None, None)   # no valid pose: stop drawing pieces
                self.error = f"{type(exc).__name__}: {str(exc)[:160]}"
        self.solve_ms = (time.perf_counter() - t) * 1000
        if self.perf is not None:
            self.perf.add("pose", self.solve_ms)
        return self.latest()

    def run(self):
        while self.running:
            with self.lock:
                frame, self.pending = self.pending, None
            if frame is None:
                time.sleep(0.005)
                continue
            self.solve_once(frame)


class HandWorker(threading.Thread):
    """Runs MediaPipe + the PinchDetector in the background.

    Every frame is processed exactly once and in order, which the detector's
    debounce counters rely on. Pinch events are queued for the GUI thread.
    """

    def __init__(self, pinch, release, frames, perf=None):
        super().__init__(daemon=True)
        self.perf = perf
        self.lock = threading.Lock()
        self.pending = None
        self.events = []                  # (event, event_position, cursor_position)
        self.gesture = None
        self.landmarks = None             # [(x, y)] normalised, for the hand occlusion mask
        self.lost = 0
        self.submitted = 0
        self.done = 0
        self.error = ""
        self.ms = 0.0
        self.running = True
        self.detector = PinchDetector(pinch, release, frames)

    def submit(self, frame):
        with self.lock:
            self.pending = frame
            self.submitted += 1

    def wait_idle(self, timeout=1.0):
        """Block until the submitted frame is processed (used when saving a video file)"""
        end = time.monotonic() + timeout
        while self.running and self.done < self.submitted and time.monotonic() < end:
            time.sleep(0.002)

    def latest_landmarks(self):
        with self.lock:
            return self.landmarks

    def pop_events(self):
        with self.lock:
            events, self.events = self.events, []
        return events

    def latest_gesture(self):
        with self.lock:
            return self.gesture

    def run(self):
        try:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model())),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            model = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            with self.lock:
                self.error = f"hand model: {type(exc).__name__}: {str(exc)[:60]}"
            return

        start, last_ts = time.monotonic(), -1
        with model:
            while self.running:
                with self.lock:
                    frame, self.pending = self.pending, None
                if frame is None:
                    time.sleep(0.005)
                    continue
                t = time.perf_counter()
                try:
                    # Positions come back in real frame pixels (w, h passed to update below)
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ts = max(last_ts + 1, int((time.monotonic() - start) * 1000))
                    last_ts = ts
                    result = model.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                    landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
                    g = self.detector.update(landmarks, w, h)
                    with self.lock:
                        if landmarks:
                            self.landmarks = [(p.x, p.y) for p in landmarks]
                            self.lost = 0
                        else:
                            self.lost += 1
                            if self.lost > 3:         # hold the mask through brief misses
                                self.landmarks = None
                        self.gesture = g
                        self.error = ""
                        if g["event"]:
                            self.events.append((g["event"], g["event_position"], g["position"]))
                except Exception as exc:
                    with self.lock:
                        self.error = f"hand: {type(exc).__name__}: {str(exc)[:60]}"
                self.ms = (time.perf_counter() - t) * 1000
                if self.perf is not None:
                    self.perf.add("hand", self.ms)
                with self.lock:
                    self.done += 1


class LiveApp(App):
    def __init__(self, root, source, width=None, height=None, pinch=DEFAULT_PINCH,
                 release=DEFAULT_RELEASE, frames=DEFAULT_FRAMES, save=None, save_fps=None,
                 log=None):
        # Not calling App.__init__: it expects a still image
        self.root = root
        root.title("Chess projection editor (live + pinch)")

        # Chess-rule state that App.draw_board / apply / reset expect
        self.state = start_position()
        self.castling_rights = starting_castling_rights()
        self.en_passant_target = None
        self.invalid_square = None
        self.selected = None              # highlight (shared with the mouse Move tool)
        self.held = None                  # square currently held by a pinch
        self.move_notice = ""
        self.move_notice_is_error = False
        self.move_notice_until = 0.0
        self.frame = None
        self.pose = None
        self.picker = None
        self.scale = 1.0
        self.photo = None
        self.out_bgr = None
        self.tool = tk.StringVar(value="None")   # nothing selected: clicks do nothing
        self._last_tool = "None"
        self.frozen = tk.BooleanVar(value=False)
        self.size = (width, height)
        self.fps = 0.0
        self.last_tick = time.perf_counter()

        # Recording
        self.save_path = save
        self.save_fps = save_fps
        self.recording = tk.BooleanVar(value=bool(save))
        self.occlude = tk.BooleanVar(value=True)
        self.writer = None
        self.rec_start = 0.0
        self.rec_written = 0
        self.rec_note = ""

        self.build_ui()
        tk.Checkbutton(self.toolbar, text="Freeze video",
                       variable=self.frozen).pack(side="left", padx=8)
        tk.Checkbutton(self.toolbar, text="Record",
                       variable=self.recording).pack(side="left")
        tk.Checkbutton(self.toolbar, text="Hand occlusion",
                       variable=self.occlude).pack(side="left", padx=8)

        self.perf = None

        self.worker = PoseWorker(self.perf)
        self.worker.start()
        self.hands = HandWorker(pinch, release, frames, self.perf)
        self.hands.start()
        print("Started opening source")
        self.open_source(source)
        print("Finished opening source")
        # Saving a video FILE: solve the pose in-line on every frame so the saved
        # overlay is frame-accurate (slower than real time, but nothing lags).
        self.sync = bool(save) and getattr(self, "is_file", False)
        self.draw_board()
        self.tick()

    # ---------- Video source ----------
    def open_source(self, source):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            self.status.config(text=f"Could not open video source: {source}")
            return
        if hasattr(self, "cap"):
            self.cap.release()
        self.cap = cap
        self.is_file = not str(source).isdigit()
        if not self.is_file:
            if self.size[0]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            if self.size[1]:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

    def open_image(self):
        """The 'Open image' button now picks a video file"""
        path = filedialog.askopenfilename(
            title="Pick a video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.open_source(path)

    # ---------- Recording ----------
    def write_frame(self):
        """Append the current overlay frame (with the pinch cursor) to the video"""
        if not self.recording.get():
            self.close_writer()
            return
        if self.out_bgr is None:
            return
        frame = self.out_bgr.copy()
        g = self.hands.latest_gesture()
        if g and g["position"]:
            colour = (0, 60, 255) if g["state"] == "PINCHED" else (70, 220, 0)
            cv2.circle(frame, tuple(g["position"]), 12, colour, 3)

        if self.writer is None:
            path = self.save_path or time.strftime("recording_%Y%m%d_%H%M%S.mp4")
            h, w = frame.shape[:2]
            fps = 0.0
            if self.is_file:
                fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fps = float(self.save_fps) if self.save_fps else (fps if fps > 0 else 30.0)
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not self.writer.isOpened():
                self.writer = None
                self.recording.set(False)
                self.rec_note = f"could not open {path} for writing"
                return
            self.writer_path, self.writer_fps = path, fps
            self.rec_start, self.rec_written = time.perf_counter(), 0

        if self.is_file:
            self.writer.write(frame)          # one saved frame per source frame
            self.rec_written += 1
        else:
            # Live camera: display rate varies, so repeat frames to keep real-time speed
            due = int((time.perf_counter() - self.rec_start) * self.writer_fps) + 1
            for _ in range(max(1, due - self.rec_written)):
                self.writer.write(frame)
                self.rec_written += 1
        self.rec_note = f"REC {self.writer_path}"

    def close_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.rec_note = f"saved {self.writer_path}"
            print(self.rec_note)

    # ---------- Move feedback ----------
    def announce_move(self, message, is_error=False):
        """Keep chess feedback visible instead of losing it to tick's FPS status."""
        self.move_notice = message
        self.move_notice_is_error = is_error
        self.move_notice_until = time.perf_counter() + 3.0

    def refresh(self):
        """tick() redraws the video every loop, so edits only need the board redrawn."""
        self.draw_board()

    def apply(self, square, button):
        """Mouse editing uses the normal App rules, with persistent feedback."""
        if square is None:
            return
        super().apply(square, button)
        if self.tool.get() != "None":
            self.announce_move(self.status.cget("text"), self.invalid_square is not None)

    def reset(self):
        self.held = None
        super().reset()
        self.announce_move("Board reset to starting position.")

    def clear(self):
        self.held = None
        super().clear()
        self.announce_move("Board cleared.")

    # ---------- Pinch handling ----------
    def handle_events(self):
        for event, event_pos, cursor in self.hands.pop_events():
            pos = event_pos or cursor
            square = self.picker.pick(*pos) if (self.picker is not None and pos) else None

            if event == "PINCH_START":
                self.invalid_square = None
                # Only pick up a piece. No colour/turn restriction.
                self.held = square if square in self.state else None

            elif event == "PINCH_RELEASE":
                if self.held is not None and square is not None and square != self.held:
                    # Same rules and bookkeeping as the mouse Move tool.
                    # An illegal move leaves the board unchanged (snap back).
                    description, reason = self.attempt_move(self.held, square)
                    if description is None:
                        self.announce_move(f"Illegal pinch move: {reason}", is_error=True)
                    else:
                        self.announce_move(f"Pinch: {description}")
                self.held = None

            elif event == "PINCH_CANCEL":
                self.held = None

            self.selected = self.held
            self.draw_board()

    def get_occluder(self):
        """Hand mask so pieces are hidden where your hand is in front of the board"""
        if not self.occlude.get() or self.frame is None:
            return None
        landmarks = self.hands.latest_landmarks()
        if not landmarks:
            return None
        h, w = self.frame.shape[:2]
        t = time.perf_counter()
        mask = hand_mask(landmarks, w, h)
        self._mask_ms = (time.perf_counter() - t) * 1000
        return mask

    # ---------- Performance log ----------
    def log_frame(self, draw_ms):
        """Record this frame's stage timings and whether the pieces are projected."""
        perf, tm, mask_ms = self.perf, self.last_timings, self._mask_ms
        if perf is None:
            return
        ms = lambda key: tm[key] * 1000 if key in tm else None
        display_ms = None
        if tm:
            # draw_image also resizes, PNG-encodes and draws to Tk: whatever is
            # left after the mask and the three render stages
            display_ms = draw_ms - (mask_ms or 0.0) - sum(tm.values()) * 1000
            for stage in ("scene", "render", "composite"):
                perf.add(stage, ms(stage))
            perf.add("display", display_ms)
        if mask_ms is not None:
            perf.add("hand_mask", mask_ms)
        perf.add("draw_image", draw_ms)
        perf.frame(pose_ms=self.worker.solve_ms, hand_ms=self.hands.ms,
                   hand_mask_ms=mask_ms, scene_ms=ms("scene"), render_ms=ms("render"),
                   composite_ms=ms("composite"), display_ms=display_ms, draw_ms=draw_ms)
        perf.track_visibility(self.pose is not None,
                              self.hands.latest_landmarks() is not None)

    def draw_cursor(self):
        g = self.hands.latest_gesture()
        if not g or not g["position"]:
            return
        x, y = g["position"][0] * self.scale, g["position"][1] * self.scale
        colour = "#ff3c00" if g["state"] == "PINCHED" else "#00dc46"
        self.image_cv.create_oval(x - 12, y - 12, x + 12, y + 12, outline=colour, width=3)

    # ---------- Main loop ----------
    def tick(self):
        new_frame = False
        if hasattr(self, "cap") and not self.frozen.get():
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    if self.recording.get():                    # one pass of the file is enough
                        self.recording.set(False)
                        self.close_writer()
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the file
                self.root.after(30, self.tick)
                return
            self.frame = frame                                  # not mirrored
            new_frame = True
            if self.sync:
                self.pose, self.picker = self.worker.solve_once(frame)
            else:
                self.worker.submit(frame)
            self.hands.submit(frame)
            if self.sync:
                self.hands.wait_idle()        # saved video: hand mask matches this exact frame

        if not self.sync:
            self.pose, self.picker = self.worker.latest()
        self.handle_events()
        self._mask_ms = None
        t_draw = time.perf_counter()
        self.draw_image()
        draw_ms = (time.perf_counter() - t_draw) * 1000
        self.draw_cursor()
        if new_frame:
            self.log_frame(draw_ms)
            self.write_frame()

        now = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_tick, 1e-6)
        self.last_tick = now
        if self.frame is not None:
            h, w = self.frame.shape[:2]
            board = "board OK" if not self.worker.error else f"board lost ({self.worker.error})"
            g = self.hands.latest_gesture()
            hand = self.hands.error or (g["state"] if g else "starting")
            show_notice = bool(self.move_notice and now < self.move_notice_until)
            notice = f"{self.move_notice}  |  " if show_notice else ""
            self.status.config(
                text=(notice + f"{w}x{h}   display {self.fps:.0f} fps   "
                      f"pose {self.worker.solve_ms:.0f} ms   {board}   "
                      f"hand {self.hands.ms:.0f} ms: {hand}"
                      + (f"   {self.rec_note}" if self.rec_note else "")),
                fg=("#bb2222" if show_notice and self.move_notice_is_error else "#333"),
            )
        self.root.after(1, self.tick)

    def close(self):
        self.worker.running = False
        self.hands.running = False
        self.close_writer()
        if hasattr(self, "cap"):
            self.cap.release()
        if self.perf is not None:
            self.perf.close()
        self.root.destroy()          # the renderer is closed in the __main__ finally block


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="camera index or video file")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--pinch", type=float, default=DEFAULT_PINCH, help="pinch close threshold")
    parser.add_argument("--release", type=float, default=DEFAULT_RELEASE, help="pinch release threshold")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="frames needed to confirm a change")
    parser.add_argument("--save", default=None, metavar="OUT.mp4",
                        help="record the projected video to this file (a video file source is "
                             "saved in one pass, frame-accurate, then recording stops)")
    parser.add_argument("--save-fps", type=float, default=None,
                        help="fps of the saved video (default: the source file's fps, or 30)")
    parser.add_argument("--log", default=None, metavar="PERF.csv",
                        help="write a per-frame performance CSV (fps, stage times, pieces visible) "
                             "and a summary with occlusion recovery times")
    args = parser.parse_args()

    root = tk.Tk()
    app = LiveApp(root, args.source, args.width, args.height,
                  args.pinch, args.release, args.frames, args.save, args.save_fps, args.log)
    root.protocol("WM_DELETE_WINDOW", app.close)
    try:
        root.mainloop()
    finally:
        pc.close_renderer()