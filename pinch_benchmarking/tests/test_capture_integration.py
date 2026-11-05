"""Hardware-free simulation of webcam/MediaPipe to exercise real video logging."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import pinch_benchmark_capture as capture
from pinch_benchmark_common import read_csv, read_jsonl
from test_benchmark import landmarks_for_ratio


class FakeCamera:
    def __init__(self):
        self.i = 0
        self.closed = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        # A uniform image cannot contain accidental hand/gesture overlays.
        frame = np.full((96, 128, 3), self.i%200, dtype=np.uint8)
        self.i += 1
        return True, frame

    def release(self):
        self.closed = True


class FakeLandmarker:
    def __init__(self):
        self.i = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        return False

    def detect_for_video(self, image, ts):
        ratio = .8 if self.i < 5 or self.i >= 13 else .04
        self.i += 1
        return SimpleNamespace(hand_landmarks=[landmarks_for_ratio(ratio)])


def fake_media_pipe():
    return SimpleNamespace(
        tasks=SimpleNamespace(
            BaseOptions=lambda **kw: None,
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(VIDEO="VIDEO"),
                HandLandmarkerOptions=lambda **kw: None,
                HandLandmarker=SimpleNamespace(create_from_options=lambda opts: FakeLandmarker()),
            ),
        ),
        Image=lambda **kw: None,
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
        __version__="FAKE-FOR-TESTS-NOT-A-REAL-MODEL",
    )


class CaptureIntegrationTests(unittest.TestCase):
    def test_recorded_frame_index_and_raw_video_correspondence(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root)/"fake.task"
            model.write_bytes(b"fake model file for test only")
            session = Path(root)/"session"
            real_capture = cv2.VideoCapture
            camera = FakeCamera()

            def fake_open_camera(src, *args, **kwargs):
                if isinstance(src, int):
                    return camera
                return real_capture(src, *args, **kwargs)

            argv = ["pinch_benchmark_capture.py", "--session", str(session),
                    "--clip-id", "test_clip", "--experiment", "pilot",
                    "--condition", "baseline", "--camera", "0",
                    "--max-frames", "22", "--warmup-frames", "0", "--no-preview",
                    "--video-codec", "MJPG", "--model", str(model)]
            with patch.object(sys,"argv",argv), patch.dict(sys.modules,{"mediapipe":fake_media_pipe()}),\
                 patch.object(capture.cv2,"VideoCapture",side_effect=fake_open_camera):
                capture.main()

            self.assertTrue(camera.closed)
            manifest = read_csv(session/"manifest.csv")
            self.assertEqual(len(manifest),1)
            self.assertEqual(manifest[0]["status"],"COMPLETE")
            self.assertEqual(int(manifest[0]["frame_count"]),22)
            stream = read_jsonl(session/"landmarks"/"test_clip.jsonl")
            self.assertEqual(len(stream),22)
            self.assertEqual([row["frame_index"] for row in stream], list(range(22)))
            self.assertEqual(sum(row["event"] == "PINCH_START" for row in stream),1)
            self.assertEqual(sum(row["event"] == "PINCH_RELEASE" for row in stream),1)
            with (session/"frames"/"test_clip.csv").open(newline="",encoding="utf-8") as stream_file:
                flat = list(csv.DictReader(stream_file))
            self.assertEqual(len(flat),22)
            self.assertEqual([int(r["frame_index"]) for r in flat],list(range(22)))
            self.assertEqual([float(r["time_s"]) for r in flat],
                             [r["time_s"] for r in stream])
            vid = real_capture(str(session/"recordings"/"test_clip.avi"))
            self.assertTrue(vid.isOpened())
            self.assertEqual(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),22)
            ok, first = vid.read()
            vid.release()
            self.assertTrue(ok)
            self.assertEqual(first.shape,(96,128,3))
            self.assertLess(int(first.max()),12, "Raw saved frame should have NO overlay")

    def test_horizontal_mirror_default_and_native_override(self):
        """The saved video must match the preprocessing handed to MediaPipe."""
        class AsymmetricCamera(FakeCamera):
            def read(self):
                # Camera left is RED and camera right is BLUE.
                frame = np.zeros((96, 128, 3), dtype=np.uint8)
                frame[:, :64] = (0, 0, 235)
                frame[:, 64:] = (235, 0, 0)
                self.i += 1
                return True, frame

        real_capture = cv2.VideoCapture
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "fake.task"
            model.write_bytes(b"not a real model")
            for is_mirrored in (True, False):
                session = Path(root) / ("mirrored" if is_mirrored else "native")
                camera = AsymmetricCamera()
                observed_mp_red_left = []

                def fake_open_camera(src, *args, **kwargs):
                    return camera if isinstance(src, int) else real_capture(src, *args, **kwargs)

                # Model input RGB must have the same orientation as the saved video.
                def fake_mp_image(**kwargs):
                    rgb = kwargs["data"]
                    observed_mp_red_left.append(float(rgb[:, :40, 0].mean()))
                    return None

                mp = fake_media_pipe()
                mp.Image = fake_mp_image
                argv = ["pinch_benchmark_capture.py", "--session", str(session),
                        "--clip-id", "flip_test", "--experiment", "pilot",
                        "--condition", "orientation_test", "--camera", "0",
                        "--max-frames", "8", "--warmup-frames", "0", "--no-preview",
                        "--video-codec", "MJPG", "--model", str(model)]
                if not is_mirrored:
                    argv.append("--no-mirror")
                with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"mediapipe": mp}), \
                     patch.object(capture.cv2, "VideoCapture", side_effect=fake_open_camera):
                    capture.main()

                manifest = read_csv(session / "manifest.csv")
                self.assertEqual(manifest[0]["mirror"], str(is_mirrored))
                video = real_capture(str(session / "recordings" / "flip_test.avi"))
                ok, frame = video.read()
                video.release()
                self.assertTrue(ok)
                red_left = float(frame[:, :40, 2].mean())
                red_right = float(frame[:, -40:, 2].mean())
                if is_mirrored:
                    self.assertLess(red_left, 40)
                    self.assertGreater(red_right, 190)
                    self.assertLess(observed_mp_red_left[0], 40)
                else:
                    self.assertGreater(red_left, 190)
                    self.assertLess(red_right, 40)
                    self.assertGreater(observed_mp_red_left[0], 190)


if __name__ == "__main__":
    unittest.main()
