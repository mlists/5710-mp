"""Event matching, visibility and summary metrics for pinch-only recordings.

Ground-truth events must be human-annotated and finalized independently of
model output. Comparison is by capture time_s, NOT by CFR playback time.
"""
from __future__ import annotations

from functools import lru_cache
from statistics import median
from typing import Any

from pinch_benchmark_common import EVENT_TYPES, as_bool


def percentile(values: list[float], percent: float):
    if not values:
        return None
    values = sorted(values)
    location = (len(values)-1)*percent/100.0
    low = int(location)
    high = min(len(values)-1, low+1)
    return float(values[low]*(high-location)+values[high]*(location-low))


def safe_round(v, decimals=5):
    return round(v, decimals) if v is not None else None


def f1_metrics(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = (2*precision*recall/(precision+recall)
          if precision is not None and recall is not None and precision+recall > 0
          else (0.0 if precision is not None and recall is not None else None))
    return precision, recall, f1


def match_one_type(gt: list[dict[str, Any]], pred: list[dict[str, Any]],
                   event_type: str, tolerance_s: float = 0.50):
    """Maximum-cardinality, minimum-absolute-delay monotone one-to-one match.

    The dynamic program avoids greedy matching accidentally losing a possible TP.
    Uncertain ground truths are excluded; predictions near them of the SAME type
    are withheld from FP counting and tabulated as ignored predictions.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(event_type)
    if tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    labels = sorted((e for e in gt if e["event_type"] == event_type
                     and not as_bool(e.get("uncertain", False))), key=lambda e: float(e["time_s"]))
    uncertain = [e for e in gt if e["event_type"] == event_type
                 and as_bool(e.get("uncertain", False))]
    predictions = sorted((e for e in pred if e["event"] == event_type),
                         key=lambda e: float(e["time_s"]))
    n, m = len(labels), len(predictions)

    @lru_cache(None)
    def match(i, j):
        if i == n or j == m:
            return 0, 0.0, ()
        choices = [match(i+1, j), match(i, j+1)]
        difference = float(predictions[j]["time_s"]) - float(labels[i]["time_s"])
        if abs(difference) <= tolerance_s + 1e-12:
            got, cost, links = match(i+1, j+1)
            choices.append((got+1, cost+abs(difference), ((i, j),)+links))
        return max(choices, key=lambda item: (item[0], -item[1]))

    _, _, pairs = match(0, 0)
    matched_labels = {i for i, _ in pairs}
    matched_preds = {j for _, j in pairs}
    delays = [float(predictions[j]["time_s"])-float(labels[i]["time_s"]) for i, j in pairs]
    ignored_preds = set()
    for j, prediction in enumerate(predictions):
        if j not in matched_preds and any(
            abs(float(prediction["time_s"])-float(u["time_s"])) <= tolerance_s
            for u in uncertain
        ):
            ignored_preds.add(j)
    fp_indices = [j for j in range(m) if j not in matched_preds and j not in ignored_preds]
    fn_indices = [i for i in range(n) if i not in matched_labels]
    audit = []
    for i, j in pairs:
        audit.append({"outcome": "TP", "gt_frame": labels[i]["frame_index"],
                      "prediction_frame": predictions[j]["frame_index"],
                      "gt_time_s": labels[i]["time_s"],
                      "prediction_time_s": predictions[j]["time_s"],
                      "signed_delay_ms": (float(predictions[j]["time_s"])-float(labels[i]["time_s"]))*1000})
    for i in fn_indices:
        audit.append({"outcome": "FN", "gt_frame": labels[i]["frame_index"],
                      "prediction_frame": "", "gt_time_s": labels[i]["time_s"],
                      "prediction_time_s": "", "signed_delay_ms": ""})
    for j in fp_indices:
        audit.append({"outcome": "FP", "gt_frame": "",
                      "prediction_frame": predictions[j]["frame_index"],
                      "gt_time_s": "", "prediction_time_s": predictions[j]["time_s"],
                      "signed_delay_ms": ""})
    for j in sorted(ignored_preds):
        audit.append({"outcome": "IGNORED_UNCERTAIN", "gt_frame": "",
                      "prediction_frame": predictions[j]["frame_index"],
                      "gt_time_s": "", "prediction_time_s": predictions[j]["time_s"],
                      "signed_delay_ms": ""})
    tp, fp, fn = len(pairs), len(fp_indices), len(fn_indices)
    precision, recall, f1 = f1_metrics(tp, fp, fn)
    return {
        "event_type": event_type, "annotated_events": n, "predicted_events": m,
        "tp": tp, "fp": fp, "fn": fn,
        "uncertain_gt_excluded": len(uncertain), "ignored_predictions": len(ignored_preds),
        "precision": precision, "recall": recall, "f1": f1,
        "median_signed_delay_ms": safe_round(median(delays)*1000) if delays else None,
        "median_absolute_delay_ms": safe_round(median(abs(x) for x in delays)*1000) if delays else None,
        "p90_absolute_delay_ms": safe_round(percentile([abs(x)*1000 for x in delays],90)) if delays else None,
        "delays_s": delays, "audit": audit,
    }


def combined_metrics(start, release):
    tp, fp, fn = (start[k]+release[k] for k in ("tp", "fp", "fn"))
    precision, recall, f1 = f1_metrics(tp, fp, fn)
    delays = start["delays_s"] + release["delays_s"]
    return {
        "event_type": "ALL", "annotated_events": start["annotated_events"]+release["annotated_events"],
        "predicted_events": start["predicted_events"]+release["predicted_events"],
        "tp": tp, "fp": fp, "fn": fn,
        "uncertain_gt_excluded": start["uncertain_gt_excluded"]+release["uncertain_gt_excluded"],
        "ignored_predictions": start["ignored_predictions"]+release["ignored_predictions"],
        "precision": precision, "recall": recall, "f1": f1,
        "median_signed_delay_ms": safe_round(median(delays)*1000) if delays else None,
        "median_absolute_delay_ms": safe_round(median(abs(x) for x in delays)*1000) if delays else None,
        "p90_absolute_delay_ms": safe_round(percentile([abs(x)*1000 for x in delays],90)) if delays else None,
        "delays_s": delays, "audit": start["audit"]+release["audit"],
    }


def visibility_metrics(frames: list[dict[str, Any]], visibility: list[str]):
    if len(frames) != len(visibility):
        raise ValueError("Visibility must have exactly one state per frame")
    present = [f for f, state in zip(frames, visibility) if state == "PRESENT"]
    absent = [f for f, state in zip(frames, visibility) if state == "ABSENT"]
    visible_detected = sum(bool(f["hand_detected"]) for f in present)
    falsely_detected = sum(bool(f["hand_detected"]) for f in absent)
    return {
        "visually_present_frames": len(present),
        "visually_absent_frames": len(absent),
        "unknown_visibility_frames": len(frames)-len(present)-len(absent),
        "present_hand_detected_frames": visible_detected,
        "absent_hand_detected_frames": falsely_detected,
        "hand_detection_rate": visible_detected/len(present) if present else None,
        "absent_hand_false_detection_rate": falsely_detected/len(absent) if absent else None,
        "landmark_observation_rate": sum(bool(f["hand_detected"]) for f in frames)/len(frames) if frames else None,
    }
