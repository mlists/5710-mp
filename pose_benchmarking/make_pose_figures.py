from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import benchmark_pose as bm
import benchmark_pose_methods as methods


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# These colours match the pinch-gesture plotting style, but the runtime panel
# intentionally uses a different pair from the held-out image-point panel.
# This keeps the two panels visually distinct while staying in the same report
# palette family.
#
# Held-out image points (paired accuracy):
#   clear      -> navy
#   occluded   -> orange
#
# Solver runtime:
#   median     -> teal
#   95th pct   -> blue
#
# OpenCV pose-overlay colours are geometric/semantic and remain unchanged.
PLOT_NAVY = "#315b7c"       # pinch 'Viewpoint'
PLOT_BLUE = "#4e83a9"       # pinch 'Working area'
PLOT_TEAL = "#4b958e"       # pinch 'Interaction'
PLOT_ORANGE = "#bf7a40"     # pinch 'Back-of-hand yaw'
PLOT_DARK = "#25313d"
PLOT_MID = "#1A5907"
PLOT_GRID = "#e1e6e9"
PLOT_SPINE = "#a5b0b9"

POINT_COLOURS = {
    "clear": PLOT_NAVY,
    "occluded": PLOT_ORANGE,
}

RUNTIME_COLOURS = {
    "median": PLOT_TEAL,
    "p95": PLOT_MID,
}


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def numeric(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def med(rows, key):
    vals = np.array([numeric(r, key) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def p95(rows, key):
    vals = np.array([numeric(r, key) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 95)) if vals.size else float("nan")


def validate_pairs(pairs):
    seen = set()
    for r in pairs:
        key = r["image"]
        if key in seen:
            raise ValueError(f"Duplicate paired image: {key}")
        seen.add(key)
        for col in ("four_rmse_px", "inner_rmse_px", "four_rmse_squares",
                    "inner_rmse_squares"):
            val = numeric(r, col)
            if not np.isfinite(val) or val < 0:
                raise ValueError(f"Invalid {col} in {key}")
        if numeric(r, "manual_points") <= 0:
            raise ValueError(f"No manual reference points in {key}")


def plot_accuracy(ax, pairs):
    condition_styles = (
        ("clear", "o", POINT_COLOURS["clear"]),
        ("occluded", "s", POINT_COLOURS["occluded"]),
    )
    for condition, marker, colour in condition_styles:
        rr = [r for r in pairs if r["condition"] == condition]
        if rr:
            ax.scatter([numeric(r, "four_rmse_px") for r in rr],
                       [numeric(r, "inner_rmse_px") for r in rr],
                       marker=marker, s=49, color=colour, edgecolor="white",
                       linewidth=0.6, label=f"{condition.capitalize()} (n={len(rr)})",
                       alpha=0.95, zorder=3)
    upper = max(max(numeric(r, "four_rmse_px"), numeric(r, "inner_rmse_px"))
                for r in pairs) * 1.1
    ax.plot([0, upper], [0, upper], ":", color=PLOT_MID,
            linewidth=1.25, label="Equal error", zorder=2)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Four-corner RMSE (px)")
    ax.set_ylabel("Internal-grid RMSE (px)")
    ax.grid(True, color=PLOT_GRID, linewidth=0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def summary_row(summary, kind, group, method):
    matching = [r for r in summary if r.get("source_type") == kind
                and r.get("group") == group and r.get("method") == method]
    if len(matching) != 1:
        raise ValueError(
            f"Expected one {kind}/{group}/{method} row in summary; got {len(matching)}"
        )
    return matching[0]


def plot_runtime(ax, summary):
    # Latest summary CSV is authoritative for timing and acceptance statistics.
    four = summary_row(summary, "video", "ALL", "four")
    inner = summary_row(summary, "video", "ALL", "inner")
    medians = [numeric(four, "pnp_median_ms"), numeric(inner, "pnp_median_ms")]
    p95s = [numeric(four, "pnp_p95_ms"), numeric(inner, "pnp_p95_ms")]
    x = np.arange(2)
    width = 0.33
    ax.bar(x - width / 2, medians, width, color=RUNTIME_COLOURS["median"],
           edgecolor="white", linewidth=0.3, label="Median")
    ax.bar(x + width / 2, p95s, width, color=RUNTIME_COLOURS["p95"],
           edgecolor="white", linewidth=0.3, label="95th percentile")
    ax.set_xticks(x, ["Four-corner", "Internal-grid"])
    ax.set_ylabel("PnP solver time (ms)")
    ax.set_ylim(0, max(p95s) * 1.2)
    ax.grid(axis="y", color=PLOT_GRID, linewidth=0.55)
    ax.legend(frameon=False, fontsize=8)


def create_charts(pairs, summary, destination):

    # Same neutral plot styling as the gesture results.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.labelcolor": PLOT_DARK, "text.color": PLOT_DARK,
        "xtick.color": PLOT_DARK, "ytick.color": PLOT_DARK,
        "axes.edgecolor": PLOT_SPINE, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 240,
    })

    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    plot_accuracy(ax, pairs)
    fig.tight_layout()
    fig.savefig(destination / "pose_accuracy_paired.png", dpi=240, bbox_inches="tight")
    fig.savefig(destination / "pose_accuracy_paired.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.45, 4.1))
    plot_runtime(ax, summary)
    fig.tight_layout()
    fig.savefig(destination / "pose_solver_runtime.png", dpi=240, bbox_inches="tight")
    fig.savefig(destination / "pose_solver_runtime.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.15), gridspec_kw={"wspace": 0.36})
    plot_accuracy(axes[0], pairs)
    plot_runtime(axes[1], summary)
    axes[0].set_title("(a) Held-out image points")
    axes[1].set_title("(b) Solver runtime")
    fig.savefig(destination / "pose_results_combined.png", dpi=240, bbox_inches="tight")
    fig.savefig(destination / "pose_results_combined.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_pose_panel(frame, tracker, pose, title):
    """Draw measured tracker outline, fitted pose outline and 3D axes on RAW pixels."""

    image = frame.copy()
    outer = np.asarray(tracker.board_corners, dtype=float).reshape(4, 2)
    projected = np.asarray(pose.projected_outer_corners, dtype=float).reshape(4, 2)
    axes = np.asarray(pose.projected_axes, dtype=float).reshape(4, 2)

    def polygon(points, color, width):
        cv2.polylines(image, [np.rint(points).astype(np.int32).reshape(-1, 1, 2)],
                      True, color, width, cv2.LINE_AA)

    polygon(outer, (0, 205, 255), 3)      # tracker: amber BGR
    polygon(projected, (240, 30, 220), 3) # pose projection: magenta BGR
    origin = tuple(np.rint(axes[0]).astype(int))
    for end, color, text_label in zip(
        axes[1:],
        [(10, 60, 255), (35, 225, 50), (255, 80, 25)],
        ["X", "Y", "Z"],
    ):
        target = tuple(np.rint(end).astype(int))
        cv2.arrowedLine(image, origin, target, color, 4, cv2.LINE_AA, tipLength=0.14)
        cv2.putText(image, text_label, target, cv2.FONT_HERSHEY_SIMPLEX,
                    0.95, color, 2, cv2.LINE_AA)
    cv2.circle(image, origin, 6, (240, 240, 240), -1, cv2.LINE_AA)
    header_h = 93
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (min(image.shape[1], 940), header_h), (27, 27, 30), -1)
    cv2.addWeighted(panel, .86, image, .14, 0, image)
    cv2.putText(image, title, (17, 34), cv2.FONT_HERSHEY_SIMPLEX, .81,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Amber: tracked boundary   Magenta: PnP projection   Red/Green/Blue: axes",
                (17, 72), cv2.FONT_HERSHEY_SIMPLEX, .53, (235, 235, 235), 1, cv2.LINE_AA)
    return image


def create_overlay(project_root, data_dir, calibration_file, example, outdir,
                   square_size, gate_squares):

    path = Path(example)
    if not path.is_absolute():
        path = data_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Original photo not found: {path}")
    key = bm.relative_key(path, data_dir)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read original photo: {path}")
    pipeline = bm.load_pipeline()
    calib = pipeline.load_calibration(calibration_file)
    bm.check_image_geometry(frame, calib, key)
    tracker = pipeline.tracker_class()
    tr, _, msg, _, _ = bm.detect_or_track(frame, tracker, pipeline, True)
    if tr is None or not tr.tracking_ok:
        raise ValueError(f"Board not tracked in example photo: {key}. Details: {msg}")
    labels = bm.load_manual_points(data_dir).get(key, {})
    cfg = SimpleNamespace(methods=("four", "inner"), pose_gate="relative",
                          max_corner_error_squares=gate_squares,
                          max_corner_error=8.0, ransac_px=3.0,
                          square_size=square_size)
    outs = methods.solve_methods(tr, calib, pipeline, cfg, manual=labels)
    four = outs["four"][0]
    inner = outs["inner"][0]
    if not four.success:
        raise ValueError(f"Four-corner pose not accepted: {four.reason}")
    final = draw_pose_panel(frame, tr, four, "Final method: 4-corner PnP")
    cv2.imwrite(str(outdir / "pose_example_four_corner.png"), final)
    if inner.success:
        alt = draw_pose_panel(frame, tr, inner, "Alternative: internal-grid PnP")
        joined = np.concatenate((final, alt), axis=1)
        if joined.shape[1] > 2700:
            width = 2700
            height = round(joined.shape[0] * width / joined.shape[1])
            joined = cv2.resize(joined, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(outdir / "pose_example_methods.png"), joined)
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                    formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=home / "pose_benchmark")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--example", type=str, default=None,
                        help="Original image relative to pose_benchmark/, e.g. images/occluded/4.jpg")
    parser.add_argument("--calibration", type=Path, default=home / "cal_data.pkl")
    parser.add_argument("--square-size", type=float, default=0.057)
    parser.add_argument("--gate-squares", type=float, default=0.60)
    parser.add_argument("--total-photos", type=int, default=18)
    parser.add_argument("--total-video-frames", type=int, default=1759)
    args = parser.parse_args()

    data = args.data_dir.resolve()
    outdir = args.output_dir.resolve() if args.output_dir else data / "figures/report"
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = read_rows(args.pairs or data / "results/pose_method_pairs.csv")
    summary = read_rows(args.summary or data / "results/pose_method_summary.csv")
    validate_pairs(pairs)
    create_charts(pairs, summary, outdir)
    example_ready = (outdir / "pose_example_four_corner.png").is_file()
    if args.example:
        key = create_overlay(home, data, args.calibration, args.example,
                             outdir, args.square_size, args.gate_squares)
        print("Saved fresh pose-overlay example:", key)
        example_ready = True
    print("Saved report files to:", outdir)


if __name__ == "__main__":
    main()
