"""Replay recorded landmarks through identical baseline/ablation detectors.

No MediaPipe or webcam is needed for this script. Real benchmark values are
computed only from COMPLETE, manually reviewed ground-truth annotations.
Output includes per-clip/per-condition metrics, an event-matching audit,
replayed decisions, baseline parity checks, publication-ready plots and JSON.

Example: python pinch_replay_evaluate.py --session session01 --frames 1 3 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from pinch_detector import PinchDetector
from pinch_benchmark_common import (
    EVENT_TYPES, as_bool, frame_duration, manifest_by_id, points_to_landmarks,
    read_csv, read_jsonl, safe_detector_update, session_file, utc_now,
    visibility_at_frames, write_csv,
)
from pinch_benchmark_metrics import (
    combined_metrics, f1_metrics, match_one_type, percentile,
    visibility_metrics,
)

CLIP_FIELDS = [
    "clip_id", "experiment", "condition", "confirmation_frames", "event_type",
    "annotated_events", "predicted_events", "tp", "fp", "fn",
    "uncertain_gt_excluded", "ignored_predictions", "precision", "recall", "f1",
    "median_signed_delay_ms", "median_absolute_delay_ms", "p90_absolute_delay_ms",
    "duration_s", "false_starts_per_min", "visually_present_frames",
    "visually_absent_frames", "unknown_visibility_frames", "present_hand_detected_frames",
    "absent_hand_detected_frames", "hand_detection_rate", "absent_hand_false_detection_rate",
    "landmark_observation_rate", "pinch_cancel_count", "processing_median_ms",
    "processing_p95_ms", "mediapipe_median_ms", "full_loop_median_ms",
]
CONDITION_FIELDS = [
    "experiment", "condition", "confirmation_frames", "event_type", "clip_count",
    "annotated_events", "predicted_events", "tp", "fp", "fn",
    "uncertain_gt_excluded", "ignored_predictions", "precision", "recall", "f1",
    "median_signed_delay_ms", "median_absolute_delay_ms", "p90_absolute_delay_ms",
    "total_duration_s", "false_starts_per_min", "visually_present_frames",
    "present_hand_detected_frames", "hand_detection_rate", "pinch_cancel_count",
    "processing_median_ms", "processing_p95_ms",
]
AUDIT_FIELDS = ["clip_id", "experiment", "condition", "confirmation_frames", "event_type",
                "outcome", "gt_frame", "prediction_frame", "gt_time_s", "prediction_time_s",
                "signed_delay_ms"]
PRED_FIELDS = ["clip_id", "confirmation_frames", "frame_index", "time_s", "state", "event",
               "raw_ratio", "smoothed_ratio", "hand_detected", "valid_landmarks"]
CONSISTENCY_FIELDS = ["clip_id", "tested_baseline", "compared_frames", "different_event_count",
                      "different_state_count", "different_ratio_count", "reason"]
ABLATION_FIELDS = ["confirmation_frames", "positive_clips", "positive_annotated_events",
                   "positive_tp", "positive_fp", "positive_fn", "positive_precision",
                   "positive_recall", "positive_f1", "median_absolute_delay_ms",
                   "median_signed_delay_ms", "negative_clips", "negative_duration_s",
                   "negative_false_starts", "negative_false_starts_per_min"]


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--session", required=True, help="Directory created by capture")
    p.add_argument("--frames", type=int, nargs="+", default=[1, 3, 5],
                   help="Confirmation frame settings for SAME landmark sequences")
    p.add_argument("--pinch", type=float, default=0.30)
    p.add_argument("--release", type=float, default=0.35)
    p.add_argument("--lost-frames", type=int, default=6)
    p.add_argument("--tolerance", type=float, default=0.50,
                   help="One-to-one ±seconds event-matching window, freeze before scoring")
    p.add_argument("--plot-clip", default="", help="ID for raw/filtered ratio timeline; default first positive clip")
    p.add_argument("--include-pilot", action="store_true", help="Include pilot/calibration clips in primary metrics; normally exclude these")
    p.add_argument("--no-plots", action="store_true", help="Compute CSV/JSON only")
    p.add_argument("--out-dir", type=Path, help="Override output directory (defaults to SESSION/results)")
    args = p.parse_args(argv)
    args.frames = sorted(set(args.frames))
    if not args.frames or any(f < 1 for f in args.frames):
        p.error("--frames must be one or more positive integers")
    if not 0 < args.pinch < args.release or args.lost_frames < 1 or args.tolerance <= 0:
        p.error("Need 0 < pinch < release, lost-frames >= 1 and tolerance > 0")
    return args


def replay_frames(frames, threshold_pinch, threshold_release, confirmation, lost_frames):
    """Replay original 21-point observations, without rerunning MediaPipe."""
    detector = PinchDetector(pinch=threshold_pinch, release=threshold_release,
                             frames=confirmation, lost_frames=lost_frames)
    out = []
    for row in frames:
        landmarks = points_to_landmarks(row.get("landmarks"))
        gesture, raw_ratio, valid, _ = safe_detector_update(
            detector, landmarks, int(row["width"]), int(row["height"]))
        out.append({
            "clip_id": row["clip_id"], "confirmation_frames": confirmation,
            "frame_index": row["frame_index"], "time_s": float(row["time_s"]),
            "state": gesture["state"], "event": gesture["event"] or "",
            "raw_ratio": raw_ratio, "smoothed_ratio": gesture["ratio"],
            "hand_detected": row["hand_detected"], "valid_landmarks": valid,
        })
    return out


def load_annotations(session, clip, frames):
    raw = [r for r in read_csv(session / "ground_truth_events.csv") if r.get("clip_id") == clip]
    result = []
    for entry in raw:
        i = int(entry["frame_index"])
        if not (0 <= i < len(frames)) or entry["event_type"] not in EVENT_TYPES:
            raise ValueError(f"Invalid ground truth event for {clip}: {entry}")
        actual = float(frames[i]["time_s"])
        if abs(actual - float(entry["time_s"])) > 0.00001:
            raise ValueError(f"Ground-truth event time must equal frame {i}'s recorded capture time: {clip}")
        result.append({**entry, "frame_index": i, "time_s": actual,
                       "uncertain": as_bool(entry.get("uncertain", ""))})
    if len({(r["frame_index"],r["event_type"]) for r in result}) != len(result):
        raise ValueError(f"Duplicate ground-truth event type/frame in {clip}")
    return result


def parity_check(clip, logged, replayed, manifest, args):
    compatible = (
        abs(float(manifest["pinch"])-args.pinch) < 1e-9
        and abs(float(manifest["release"])-args.release) < 1e-9
        and int(manifest["confirmation_frames"]) == 3
        and int(manifest["lost_frames"]) == args.lost_frames
    )
    if not compatible:
        return {"clip_id": clip, "tested_baseline": False, "compared_frames": 0,
                "different_event_count": "", "different_state_count": "",
                "different_ratio_count": "", "reason": "Capture settings differ from 3-frame baseline"}
    if len(logged) != len(replayed):
        raise ValueError(f"Replay length mismatch: {clip}")
    event_misses = state_misses = ratio_misses = 0
    for original, now in zip(logged, replayed):
        event_misses += (original.get("event") or "") != now["event"]
        state_misses += original.get("state") != now["state"]
        a, b = original.get("smoothed_ratio"), now["smoothed_ratio"]
        ratio_misses += ((a is None) != (b is None)) or (
            a is not None and b is not None and abs(float(a)-float(b)) > 1e-8)
    return {"clip_id": clip, "tested_baseline": True, "compared_frames": len(logged),
            "different_event_count": event_misses, "different_state_count": state_misses,
            "different_ratio_count": ratio_misses,
            "reason": "OK" if not(event_misses or state_misses or ratio_misses) else "REPLAY_MISMATCH"}


def as_csv_metrics(m, metadata):
    return {**metadata, **{k: v for k, v in m.items() if k not in ("audit", "delays_s")}}


def aggregate_condition(rows):
    """Pool TP/FP/FN and matched-event delays (no averaging per-clip F1)."""
    grouped = defaultdict(list)
    for row in rows:
        key = (row["experiment"], row["condition"], row["confirmation_frames"], row["event_type"])
        grouped[key].append(row)
    result = []
    for (experiment, condition, count, kind), items in sorted(grouped.items()):
        tp = sum(r["tp"] for r in items)
        fp = sum(r["fp"] for r in items)
        fn = sum(r["fn"] for r in items)
        pr, rc, f1 = f1_metrics(tp, fp, fn)
        delays = [v for r in items for v in r["_delays_s"]]
        duration = sum(r["duration_s"] for r in items)
        rate = None
        if experiment.lower() == "negative" and kind in ("PINCH_START", "ALL"):
            # Negative controls deliberately contain no physical pinch contacts.
            # Use predicted START count, never count RELEASE/CANCEL as an activation.
            starts = sum(r["predicted_events"] if kind == "PINCH_START" else r["_predicted_starts"] for r in items)
            rate = 60.0*starts/duration if duration > 0 else None
        visible = sum(r["visually_present_frames"] for r in items)
        detected = sum(r["present_hand_detected_frames"] for r in items)
        timing = [v for r in items for v in r["_processing_ms"]]
        result.append({
            "experiment": experiment, "condition": condition, "confirmation_frames": count,
            "event_type": kind, "clip_count": len(items),
            **{k: sum(r[k] for r in items) for k in
               ("annotated_events", "predicted_events", "tp", "fp", "fn",
                "uncertain_gt_excluded", "ignored_predictions", "pinch_cancel_count")},
            "precision": pr, "recall": rc, "f1": f1,
            "median_signed_delay_ms": median(delays)*1000 if delays else None,
            "median_absolute_delay_ms": median(abs(x) for x in delays)*1000 if delays else None,
            "p90_absolute_delay_ms": percentile([abs(x)*1000 for x in delays],90),
            "total_duration_s": duration, "false_starts_per_min": rate,
            "visually_present_frames": visible, "present_hand_detected_frames": detected,
            "hand_detection_rate": detected/visible if visible else None,
            "processing_median_ms": median(timing) if timing else None,
            "processing_p95_ms": percentile(timing,95),
        })
    return result


def summarize_ablation(metrics):
    grouped = defaultdict(list)
    for r in metrics:
        if r["event_type"] == "ALL":
            grouped[r["confirmation_frames"]].append(r)
    result = []
    for frames_setting, items in sorted(grouped.items()):
        positives = [r for r in items if r["experiment"].lower() != "negative"]
        negatives = [r for r in items if r["experiment"].lower() == "negative"]
        tp = sum(r["tp"] for r in positives)
        fp = sum(r["fp"] for r in positives)
        fn = sum(r["fn"] for r in positives)
        p, rec, f1 = f1_metrics(tp, fp, fn)
        delays = [v for r in positives for v in r["_delays_s"]]
        neg_duration = sum(r["duration_s"] for r in negatives)
        false_starts = sum(r["_predicted_starts"] for r in negatives)
        result.append({
            "confirmation_frames": frames_setting,
            "positive_clips": len(positives), "positive_annotated_events": sum(r["annotated_events"] for r in positives),
            "positive_tp": tp, "positive_fp": fp, "positive_fn": fn,
            "positive_precision": p, "positive_recall": rec, "positive_f1": f1,
            "median_absolute_delay_ms": median(abs(d) for d in delays)*1000 if delays else None,
            "median_signed_delay_ms": median(delays)*1000 if delays else None,
            "negative_clips": len(negatives), "negative_duration_s": neg_duration,
            "negative_false_starts": false_starts,
            "negative_false_starts_per_min": 60*false_starts/neg_duration if neg_duration else None,
        })
    return result


def produce_results(session: Path, args):
    manifest = manifest_by_id(session)
    if not manifest:
        raise ValueError("No clips found in manifest. Record clips with pinch_benchmark_capture.py")
    statuses = {row["clip_id"]: row["status"] for row in read_csv(session / "annotation_status.csv")}
    usable = [x for x in manifest.values()
              if x["status"] == "COMPLETE" and statuses.get(x["clip_id"]) == "COMPLETE"
              and (args.include_pilot or x["experiment"].lower() != "pilot")]
    excluded = [x["clip_id"] for x in manifest.values() if x not in usable]
    for clip in excluded:
        if manifest[clip]["experiment"].lower() == "pilot" and not args.include_pilot:
            print(f"SKIPPING pilot clip {clip} (use --include-pilot to include)")
        else:
            print(f"SKIPPING unreviewed/incomplete clip {clip}. Press F in annotation GUI when finished.")
    if not usable:
        raise ValueError("No COMPLETE manually annotated clips. Run pinch_benchmark_annotate.py, then press F")
    event_rows, pred_rows, internal_rows, consistency = [], [], [], []
    timelines = {}
    for info in sorted(usable, key=lambda r: r["clip_id"]):
        clip = info["clip_id"]
        records = read_jsonl(session_file(session, info["log_file"]))
        if int(info["frame_count"]) != len(records):
            raise ValueError(f"Manifest/landmark count mismatch: {clip}")
        if any(r["clip_id"] != clip for r in records):
            raise ValueError(f"Unexpected clip_id in landmark stream {clip}")
        gt = load_annotations(session, clip, records)
        if info["experiment"].lower() == "negative" and gt:
            raise ValueError(f"Negative control {clip} contains {len(gt)} physical event labels. "
                             "This is not a valid negative trial: recapture or change its experiment group")
        if info["experiment"].lower() != "negative" and not gt:
            print(f"WARNING: {clip} has zero physical event labels; check that it is intentionally empty")
        vis = visibility_metrics(records, visibility_at_frames(session, clip, len(records)))
        duration = frame_duration(records, float(info.get("video_fps_nominal") or 30))
        timings = [float(r["frame_to_gesture_ms"]) for r in records]
        mp_times = [float(r["mediapipe_ms"]) for r in records]
        loop_times = [float(r["full_loop_ms"]) for r in records]
        for f in args.frames:
            replay = replay_frames(records, args.pinch, args.release, f, args.lost_frames)
            pred_rows.extend({k: r.get(k) for k in PRED_FIELDS} for r in replay)
            predicted = [r for r in replay if r["event"]]
            start = match_one_type(gt, predicted, "PINCH_START", args.tolerance)
            release = match_one_type(gt, predicted, "PINCH_RELEASE", args.tolerance)
            all_events = combined_metrics(start, release)
            cancels = sum(r["event"] == "PINCH_CANCEL" for r in predicted)
            starts_predicted = sum(r["event"] == "PINCH_START" for r in predicted)
            metadata = {
                "clip_id": clip, "experiment": info["experiment"],
                "condition": info["condition"], "confirmation_frames": f,
                "duration_s": duration,
                "false_starts_per_min": 60*starts_predicted/duration
                if info["experiment"].lower() == "negative" and duration > 0 else None,
                **vis,
                "pinch_cancel_count": cancels,
                "processing_median_ms": median(timings),
                "processing_p95_ms": percentile(timings,95),
                "mediapipe_median_ms": median(mp_times),
                "full_loop_median_ms": median(loop_times),
            }
            for m in (start, release, all_events):
                internal_rows.append({**as_csv_metrics(m, metadata),
                                      "_delays_s": m["delays_s"],
                                      "_processing_ms": timings,
                                      "_predicted_starts": starts_predicted})
            for m in (start, release):
                for audit in m["audit"]:
                    event_rows.append({"clip_id": clip, "experiment": info["experiment"],
                                       "condition": info["condition"],
                                       "confirmation_frames": f, "event_type": m["event_type"],
                                       **audit})
            # The 3-frame setting should be exactly the same as live logging.
            if f == 3:
                consistency.append(parity_check(clip, records, replay, info, args))
            should_plot_clip = clip == args.plot_clip or (not args.plot_clip and not timelines and info["experiment"].lower() != "negative")
            preferred_frames = 3 if 3 in args.frames else args.frames[0]
            if should_plot_clip and f == preferred_frames:
                timelines[clip] = (records, gt, replay)
    condition_rows = aggregate_condition(internal_rows)
    ablations = summarize_ablation(internal_rows)
    output = args.out_dir.resolve() if args.out_dir else session / "results"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_clip_metrics.csv", CLIP_FIELDS, internal_rows)
    write_csv(output / "per_condition_metrics.csv", CONDITION_FIELDS, condition_rows)
    write_csv(output / "event_matching_audit.csv", AUDIT_FIELDS, event_rows)
    write_csv(output / "replay_predictions.csv", PRED_FIELDS, pred_rows)
    write_csv(output / "baseline_replay_consistency.csv", CONSISTENCY_FIELDS, consistency)
    write_csv(output / "confirmation_ablation.csv", ABLATION_FIELDS, ablations)
    from pinch_benchmark_tables import make_tables
    make_tables(output, condition_rows, ablations)
    summary = {
        "generated_utc": utc_now(),
        "ground_truth_policy": "Only clips explicitly finalised by human reviewer; uncertain labels and predictions near them of same type excluded",
        "timebase": "monotonic acquisition-time-s from first camera frame; not nominal video FPS",
        "event_matching": "max-TP min-absolute-error, one-to-one, same event type, time tolerance ±seconds",
        "tolerance_s": args.tolerance, "pinch_threshold": args.pinch,
        "release_threshold": args.release, "lost_hand_frames": args.lost_frames,
        "confirmation_ablation_frames": args.frames,
        "evaluated_clips": [x["clip_id"] for x in usable], "excluded_clips": excluded,
        "per_condition": condition_rows, "ablation": ablations,
        "note": "Human-to-system delay and frame processing time are different metrics; no results are calculated for unlabelled clips",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (output / "RESULTS_README.md").write_text(make_results_readme(summary, consistency), encoding="utf-8")
    if not args.no_plots:
        try:
            from pinch_benchmark_plots import create_plots
            create_plots(output, condition_rows, ablations, timelines, internal_rows,
                         args.pinch, args.release)
        except ImportError as exc:
            print(f"Plots skipped: {exc}. Install matplotlib/numpy from requirements.txt")
    print(f"Evaluated {len(usable)} reviewed clips; excluded {len(excluded)}. Output: {output}")
    if any(r.get("reason") == "REPLAY_MISMATCH" for r in consistency):
        print("WARNING: Live baseline and offline replay disagree. Inspect baseline_replay_consistency.csv", file=sys.stderr)
    for row in ablations:
        f1 = "N/A" if row["positive_f1"] is None else f"{row['positive_f1']:.3f}"
        neg = "N/A" if row["negative_false_starts_per_min"] is None else f"{row['negative_false_starts_per_min']:.3f}/min"
        print(f"confirmation={row['confirmation_frames']}: pooled positive F1={f1}; negative false starts={neg}")
    return summary


def make_results_readme(summary, parity):
    text = ["# Generated pinch benchmark results", "", f"Generated: {summary['generated_utc']}",
            "", "Only independently hand-reviewed clips marked COMPLETE are included.",
            f"Event tolerance: ±{summary['tolerance_s']:.2f} seconds. Uncertain labels are excluded.",
            "Undefined precision/recall/F1 are BLANK, not perfect scores.",
            "", "**Gesture detection delay** uses actual capture timestamps and human-labelled contact/separation.",
            "**Frame processing time** starts after camera read and ends after gesture decision.",
            "Neither video playback FPS nor camera-read time is substituted for gesture delay.",
            "", "## Confirmation ablation (pooled over trials)", "",
            "| Confirm frames | Positive F1 | Median absolute delay (ms) | Negative false starts/min |",
            "|---:|---:|---:|---:|"]
    def display(value, fmt):
        return format(value, fmt) if value is not None else "N/A"
    for row in summary["ablation"]:
        text.append(f"| {row['confirmation_frames']} | {display(row['positive_f1'], '.3f')} | "
                    f"{display(row['median_absolute_delay_ms'], '.1f')} | "
                    f"{display(row['negative_false_starts_per_min'], '.3f')} |")
    text += ["", "## Data files", "", "- per_clip_metrics.csv: event-type and ALL metrics by clip.",
             "- per_condition_metrics.csv: pooled TP/FP/FN and landmark detection rates by condition.",
             "- event_matching_audit.csv: matched/unmatched individual events, delays, excluded predictions.",
             "- replay_predictions.csv: every frame replayed under each confirmation setting.",
             "- confirmation_ablation.csv: pooled summary of the exact same landmark streams.",
             "- table_conditions.tex / table_negative.tex / table_ablation.tex: editable LaTeX tables.",
             "- baseline_replay_consistency.csv: confirms frame-by-frame identity with live 3-frame decision logging.",
             "- summary.json: machine-readable metrics and protocol parameters.",
             "", "## Limitations", "", "- Event contact timing is limited by frame sampling and manual judgement.",
             "- Capture time stamps are recorded at camera-read return, not at the sensor exposure instant.",
             "- Only the same camera/participant/environment used here are assessed; no generalisation claim is implied.",
             "- Ratios/frequencies and F1 are not physical accuracy in centimetres.", ""]
    mismatches = [p for p in parity if p.get("reason") == "REPLAY_MISMATCH"]
    if mismatches:
        text.extend(["**WARNING:** Recorded and replayed baseline differ. Inspect parity CSV before citing numbers.", ""])
    return "\n".join(text)


def main():
    args = arguments()
    session = Path(args.session).expanduser().resolve()
    try:
        produce_results(session, args)
    except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
        raise SystemExit(f"Benchmark evaluation failed: {exc}") from exc


if __name__ == "__main__":
    main()
