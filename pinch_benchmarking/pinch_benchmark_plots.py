"""Publication-ready plots derived only from real annotated benchmark recordings."""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe on headless machines; plots saved to PNG/PDF.
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path):
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _pick_baseline(available):
    return 3 if 3 in available else (available[0] if available else None)


def plot_condition_f1(output, conditions, baseline):
    src = [r for r in conditions if r["confirmation_frames"] == baseline
           and r["experiment"].lower() != "negative" and r["event_type"] in ("PINCH_START", "PINCH_RELEASE")]
    conditions_names = sorted(set((r["experiment"], r["condition"]) for r in src))
    if not conditions_names:
        return
    values = {(r["experiment"],r["condition"],r["event_type"]): r for r in src}
    x = np.arange(len(conditions_names))
    fig, ax = plt.subplots(figsize=(max(8, 1.40*len(x)),4.5))
    for offset, kind, label in ((-.19, "PINCH_START", "Contact (START)"),
                                (.19, "PINCH_RELEASE", "Separation (RELEASE)")):
        y = [values.get((e,c,kind), {}).get("f1") for e,c in conditions_names]
        y_plot = [v if v is not None else np.nan for v in y]
        bars = ax.bar(x+offset, y_plot, width=.36, label=label)
        for bar, value in zip(bars, y):
            if value is not None:
                ax.annotate(f"{value:.2f}", (bar.get_x()+bar.get_width()/2, value),
                            xytext=(0,3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylim(0,1.13)
    ax.set_ylabel("Event F1 (one-to-one match)")
    ax.set_title(f"Pinch events across conditions, {baseline}-frame confirmation")
    ax.set_xticks(x, [f"{e}\n{c}" for e,c in conditions_names], rotation=20, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=.2)
    _save(fig, output/"figure_f1_by_condition")


def plot_negative_false_starts(output, conditions, baseline):
    rows = [r for r in conditions if r["confirmation_frames"] == baseline
            and r["experiment"].lower() == "negative" and r["event_type"] == "ALL"]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["condition"])
    fig, ax = plt.subplots(figsize=(max(7,1.4*len(rows)),4))
    x = np.arange(len(rows))
    counts = [r["false_starts_per_min"] for r in rows]
    ax.bar(x, [v if v is not None else 0 for v in counts])
    ax.set_xticks(x, [r["condition"] for r in rows], rotation=15, ha="right")
    ax.set_ylabel("False PINCH_START activations per minute")
    ax.set_title(f"Negative-control false activation rate, {baseline}-frame confirmation")
    ax.grid(axis="y", alpha=.2)
    _save(fig, output/"figure_negative_false_starts")


def plot_ablation(output, ablation):
    if len(ablation) < 2:
        return
    frames = [r["confirmation_frames"] for r in ablation]
    series = [
        ("positive_f1", "Pooled positive event F1", "figure_confirmation_f1"),
        ("median_absolute_delay_ms", "Matched event median absolute delay (ms)", "figure_confirmation_delay"),
        ("negative_false_starts_per_min", "Negative false START events/min", "figure_confirmation_false_starts"),
    ]
    for key, ylabel, filename in series:
        y = [r[key] if r[key] is not None else np.nan for r in ablation]
        if all(math.isnan(float(v)) for v in y):
            continue
        fig, ax = plt.subplots(figsize=(6.8,4.0))
        ax.plot(frames, y, marker="o", lw=1.8)
        for x, v in zip(frames, y):
            if not math.isnan(float(v)):
                ax.annotate(f"{v:.3f}" if key != "median_absolute_delay_ms" else f"{v:.0f}",
                            (x,v), xytext=(0,8), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(frames)
        ax.set_xlabel("Consecutive processed frames for confirmation")
        ax.set_ylabel(ylabel)
        ax.set_title("Temporal-confirmation ablation on identical landmarks")
        if key == "positive_f1":
            ax.set_ylim(0,1.12)
        ax.grid(alpha=.24)
        _save(fig, output/filename)


def plot_timeline(output, clip, records, gt, replay, pinch, release):
    t = np.array([float(r["time_s"]) for r in records])
    ratio = np.array([np.nan if r.get("raw_ratio") is None else float(r["raw_ratio"]) for r in records])
    smoothed = np.array([np.nan if r.get("smoothed_ratio") is None else float(r["smoothed_ratio"]) for r in replay])
    fig, ax = plt.subplots(figsize=(13.5,4.7))
    ax.plot(t, ratio, lw=1.0, alpha=.55, label="Raw palm-normalised ratio")
    ax.plot(t, smoothed, lw=1.7, label="Smoothed ratio (0.5 current + 0.5 previous)")
    ax.axhline(pinch, ls="--", lw=1.3, label=f"Pinch threshold = {pinch:g}")
    ax.axhline(release, ls=":", lw=1.3, label=f"Release threshold = {release:g}")
    for k, marker, lbl in (("PINCH_START", "o", "START"), ("PINCH_RELEASE", "s", "RELEASE")):
        physical = [float(e["time_s"]) for e in gt if e["event_type"] == k and not e["uncertain"]]
        predicted = [float(r["time_s"]) for r in replay if r["event"] == k]
        if physical:
            ax.scatter(physical, [0.13 if k == "PINCH_START" else 0.70]*len(physical),
                       marker=marker, s=35, label=f"Hand-labelled {lbl}", zorder=6)
        if predicted:
            ax.scatter(predicted, [0.20 if k == "PINCH_START" else 0.77]*len(predicted),
                       marker=marker, facecolors="none", edgecolors="black",
                       s=55, label=f"Detector {lbl}", zorder=6)
    ax.set_xlim(t[0], t[-1] if len(t)>1 else t[0]+1)
    ax.set_ylim(-.05, min(max(np.nanmax(smoothed) if np.isfinite(smoothed).any() else 0.8,
                              np.nanmax(ratio) if np.isfinite(ratio).any() else 0.8)+.18,3.0))
    ax.set_xlabel("Camera-capture timeline (s), NOT encoded video playback time")
    ax.set_ylabel("Dimensionless fingertip distance / palm size")
    ax.set_title(f"Pinch recognition: raw/filtered ratio and physical/detected events ({clip})")
    ax.grid(alpha=.18)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", clip)
    _save(fig, output/f"figure_ratio_timeline_{safe}")


def plot_processing(output, clip_rows, baseline):
    selected = [r for r in clip_rows if r["confirmation_frames"] == baseline
                and r["event_type"] == "ALL"]
    if not selected:
        return
    names = [r["clip_id"] for r in selected]
    med = [r["processing_median_ms"] for r in selected]
    p95 = [r["processing_p95_ms"] for r in selected]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8,1.25*len(names)),4.3))
    ax.bar(x-.19, med, width=.38, label="Median")
    ax.bar(x+.19, p95, width=.38, label="95th percentile")
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("Frame-to-gesture processing time (ms)")
    ax.set_title("Per-frame computation, excluding waiting for gesture confirmation")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    _save(fig, output/"figure_processing_times")


def create_plots(output, conditions, ablation, timelines, internal_rows, pinch, release):
    available = sorted(set(r["confirmation_frames"] for r in internal_rows))
    baseline = _pick_baseline(available)
    if baseline is None:
        return
    plot_condition_f1(output, conditions, baseline)
    plot_negative_false_starts(output, conditions, baseline)
    plot_ablation(output, ablation)
    for clip, (records, gt, replay) in timelines.items():
        plot_timeline(output, clip, records, gt, replay, pinch, release)
    plot_processing(output, internal_rows, baseline)
