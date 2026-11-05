"""Synthetic software verification only, NOT experiment data or paper results."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pinch_detector import PinchDetector
from pinch_benchmark_common import (
    GROUND_TRUTH_FIELDS, MANIFEST_FIELDS, STATUS_FIELDS, VISIBILITY_FIELDS,
    ensure_session, read_csv, safe_detector_update, visibility_at_frames,
    write_csv, write_visibility_breakpoints,
)
from pinch_benchmark_metrics import f1_metrics, match_one_type, visibility_metrics
from pinch_replay_evaluate import arguments, produce_results, replay_frames


def landmarks_for_ratio(ratio):
    """Make a 21-point synthetic hand with palm diameter 200px in 1000x1000."""
    p = [SimpleNamespace(x=.5,y=.5,z=0.0) for _ in range(21)]
    p[0] = SimpleNamespace(x=.5,y=.5,z=0.0)
    p[9] = SimpleNamespace(x=.5,y=.4,z=0.0)
    p[5] = SimpleNamespace(x=.4,y=.45,z=0.0)
    p[17] = SimpleNamespace(x=.6,y=.45,z=0.0)
    p[8] = SimpleNamespace(x=.5,y=.5,z=0.0)
    p[4] = SimpleNamespace(x=.5-ratio*.2,y=.5,z=0.0)
    return p


def synthetic_frame(clip, idx, ratio, event="", state="OPEN", t=None):
    landmarks = landmarks_for_ratio(ratio) if ratio is not None else None
    return {
        "clip_id": clip, "frame_index": idx, "time_s": idx/30 if t is None else t,
        "video_time_s": idx/30, "mp_timestamp_ms": idx*34, "width":1000, "height":1000,
        "hand_detected": landmarks is not None, "valid_landmarks": landmarks is not None,
        "landmarks": [[p.x,p.y,p.z] for p in landmarks] if landmarks is not None else None,
        "raw_ratio": ratio, "smoothed_ratio": ratio,
        "state": state, "event": event,
        "frame_to_gesture_ms": 22.0, "mediapipe_ms":20.0, "full_loop_ms":26.0,
    }


class DetectorTests(unittest.TestCase):
    def test_baseline_transitions_and_loss_cancel(self):
        d = PinchDetector(pinch=.30, release=.35, frames=3, lost_frames=6)
        ratios = [.80]*5 + [.08]*9 + [.80]*9 + [.08]*9
        events = []
        for v in ratios:
            result, raw, valid, _ = safe_detector_update(d, landmarks_for_ratio(v), 1000,1000)
            self.assertTrue(valid)
            self.assertAlmostEqual(raw, v, places=6)
            if result["event"]:
                events.append(result["event"])
        self.assertEqual(events, ["PINCH_START", "PINCH_RELEASE", "PINCH_START"])
        for i in range(6):
            result, _, valid, _ = safe_detector_update(d, None,1000,1000)
        self.assertEqual(result["event"], "PINCH_CANCEL")
        self.assertFalse(d.active)

    def test_degenerate_palm_does_not_return_none(self):
        d = PinchDetector(.30,.35,3,6)
        p = [SimpleNamespace(x=.5,y=.5,z=0) for _ in range(21)]
        gesture, ratio, valid, palm = safe_detector_update(d,p,1000,1000)
        self.assertEqual(gesture["state"],"NO HAND")
        self.assertFalse(valid)
        self.assertIsNone(ratio)
        self.assertEqual(palm, 0.0)

    def test_three_frames_delays_initial_contact(self):
        frames = [synthetic_frame("x", i, .8 if i<10 else .06) for i in range(20)]
        first = replay_frames(frames,.30,.35,1,6)
        third = replay_frames(frames,.30,.35,3,6)
        start1 = next(r["frame_index"] for r in first if r["event"] == "PINCH_START")
        start3 = next(r["frame_index"] for r in third if r["event"] == "PINCH_START")
        self.assertLess(start1,start3)
        self.assertGreaterEqual(start3-start1,2)


class MetricsTests(unittest.TestCase):
    def test_matching_max_cardinality_not_greedy(self):
        gt = [{"event_type":"PINCH_START","time_s":0.0,"frame_index":0,"uncertain":False},
              {"event_type":"PINCH_START","time_s":0.55,"frame_index":16,"uncertain":False}]
        pred = [{"event":"PINCH_START","time_s":0.49,"frame_index":14},
                {"event":"PINCH_START","time_s":0.90,"frame_index":27}]
        result = match_one_type(gt,pred,"PINCH_START",.50)
        self.assertEqual(result["tp"],2)
        self.assertEqual(result["fp"],0)
        self.assertEqual(result["fn"],0)

    def test_uncertainty_not_counted_as_false_positive(self):
        gt = [{"event_type":"PINCH_START","time_s":1.00,"frame_index":30,"uncertain":True}]
        pred = [{"event":"PINCH_START","time_s":1.16,"frame_index":35}]
        metrics = match_one_type(gt,pred,"PINCH_START",.5)
        self.assertEqual(metrics["fp"],0)
        self.assertEqual(metrics["ignored_predictions"],1)
        self.assertIsNone(metrics["f1"])

    def test_undefined_negative_scores_not_one(self):
        self.assertEqual(f1_metrics(0,0,0),(None,None,None))
        self.assertEqual(f1_metrics(0,1,0),(0.0,None,None))

    def test_visibility_with_unknown(self):
        frames = [{"hand_detected":True},{"hand_detected":False},{"hand_detected":True}]
        result = visibility_metrics(frames,["PRESENT","PRESENT","UNKNOWN"])
        self.assertAlmostEqual(result["hand_detection_rate"],.5)
        self.assertEqual(result["unknown_visibility_frames"],1)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = ensure_session(Path(self.tmp.name)/"session")

    def tearDown(self):
        self.tmp.cleanup()

    def _create_clip(self, clip, experiment, condition, ratios, label_indexes):
        rows = [synthetic_frame(clip,i,ratio) for i,ratio in enumerate(ratios)]
        # Emulate capture logs using the same known-good detector baseline.
        baseline = replay_frames(rows,.30,.35,3,6)
        for row, detected in zip(rows,baseline):
            row["event"] = detected["event"] or None
            row["state"] = detected["state"]
            row["smoothed_ratio"] = detected["smoothed_ratio"]
        path = self.session/"landmarks"/f"{clip}.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        row = {k:"" for k in MANIFEST_FIELDS}
        row.update({"clip_id":clip,"experiment":experiment,"condition":condition,
                    "video_file":f"recordings/{clip}.avi",
                    "log_file":f"landmarks/{clip}.jsonl",
                    "frame_count":len(rows),"video_fps_nominal":30,
                    "pinch":.30,"release":.35,"confirmation_frames":3,
                    "lost_frames":6,"status":"COMPLETE"})
        self.manifest_rows.append(row)
        for idx, event_type in label_indexes:
            self.labels.append({"clip_id":clip,"event_id":f"{clip}_{idx}",
                                "event_type":event_type,"frame_index":idx,
                                "time_s":rows[idx]["time_s"],"uncertain":False,"notes":""})
        self.statuses.append({"clip_id":clip,"status":"COMPLETE","reviewed_at_utc":"", "notes":""})
        write_visibility_breakpoints(self.session,clip,{0:"PRESENT"},len(rows))

    def test_full_offline_pipeline_scores_and_exports(self):
        self.manifest_rows = []
        self.labels = []
        self.statuses = []
        # One positive pinch (contact first closed frame, release first open frame).
        # 10 open + 12 pinched + 12 open; 10 negative control frames.
        self._create_clip("distance_35", "distance", "35cm", [.80]*10+[.05]*12+[.80]*12,
                          [(10,"PINCH_START"),(22,"PINCH_RELEASE")])
        self._create_clip("negative_open", "negative", "open_palm", [.80]*90, [])
        write_csv(self.session/"manifest.csv",MANIFEST_FIELDS,self.manifest_rows)
        write_csv(self.session/"ground_truth_events.csv",GROUND_TRUTH_FIELDS,self.labels)
        write_csv(self.session/"annotation_status.csv",STATUS_FIELDS,self.statuses)
        args = arguments(["--session",str(self.session),"--frames","1","3","5","--no-plots"])
        summary = produce_results(self.session,args)
        output = self.session/"results"
        self.assertEqual(len(summary["evaluated_clips"]),2)
        for name in ("per_clip_metrics.csv","per_condition_metrics.csv","replay_predictions.csv",
                     "confirmation_ablation.csv","summary.json","RESULTS_README.md",
                     "baseline_replay_consistency.csv","event_matching_audit.csv",
                     "table_conditions.tex","table_negative.tex","table_ablation.tex"):
            self.assertTrue((output/name).is_file(),name)
        check = read_csv(output/"baseline_replay_consistency.csv")
        self.assertTrue(all(r["reason"] == "OK" for r in check))
        clip_rows = read_csv(output/"per_clip_metrics.csv")
        self.assertEqual(len(clip_rows),2*3*3)
        pos_three = next(r for r in clip_rows if r["clip_id"]=="distance_35"
                         and r["confirmation_frames"]=="3" and r["event_type"]=="ALL")
        self.assertEqual(pos_three["tp"],"2")
        self.assertEqual(pos_three["fp"],"0")
        negative = next(r for r in clip_rows if r["clip_id"]=="negative_open"
                        and r["confirmation_frames"]=="3" and r["event_type"]=="ALL")
        self.assertEqual(negative["fp"],"0")
        self.assertEqual(negative["f1"],"")
        # Verify real figure-generation code as well as tabular exports.
        args.no_plots = False
        produce_results(self.session, args)
        for name in ("figure_f1_by_condition.png", "figure_negative_false_starts.png",
                     "figure_confirmation_f1.png", "figure_confirmation_delay.png",
                     "figure_confirmation_false_starts.png",
                     "figure_ratio_timeline_distance_35.png", "figure_processing_times.png"):
            self.assertTrue((output/name).is_file(), name)

    def test_incomplete_annotation_is_not_silent_negative(self):
        self.manifest_rows=[]; self.labels=[]; self.statuses=[]
        self._create_clip("unreviewed", "negative", "open_palm", [.8]*4,[])
        write_csv(self.session/"manifest.csv",MANIFEST_FIELDS,self.manifest_rows)
        write_csv(self.session/"annotation_status.csv",STATUS_FIELDS,[])
        args = arguments(["--session",str(self.session),"--no-plots"])
        with self.assertRaisesRegex(ValueError,"No COMPLETE manually annotated clips"):
            produce_results(self.session,args)


if __name__ == "__main__":
    unittest.main()
