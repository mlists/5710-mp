from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
    for condition, marker in (("clear", "o"), ("occluded", "s")):
        rr = [r for r in pairs if r["condition"] == condition]
        if rr:
            ax.scatter([numeric(r, "four_rmse_px") for r in rr],
                       [numeric(r, "inner_rmse_px") for r in rr],
                       marker=marker, s=49, label=f"{condition.capitalize()} (n={len(rr)})",
                       alpha=0.9)
    upper = max(max(numeric(r, "four_rmse_px"), numeric(r, "inner_rmse_px"))
                for r in pairs) * 1.1
    ax.plot([0, upper], [0, upper], ":", linewidth=1.25, label="Equal error")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Four-corner RMSE (px)")
    ax.set_ylabel("Internal-grid RMSE (px)")
    ax.grid(True, alpha=0.17)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def summary_row(summary, kind, group, method):
    matching = [r for r in summary if r.get("source_type") == kind
                and r.get("group") == group and r.get("method") == method]
    if len(matching) != 1:
        raise ValueError(f"Expected one {kind}/{group}/{method} row in summary; got {len(matching)}")
    return matching[0]


def plot_runtime(ax, summary):
    # Latest summary CSV is authoritative for timing and acceptance statistics.
    four = summary_row(summary, "video", "ALL", "four")
    inner = summary_row(summary, "video", "ALL", "inner")
    medians = [numeric(four, "pnp_median_ms"), numeric(inner, "pnp_median_ms")]
    p95s = [numeric(four, "pnp_p95_ms"), numeric(inner, "pnp_p95_ms")]
    x = np.arange(2)
    width = 0.33
    ax.bar(x - width / 2, medians, width, label="Median")
    ax.bar(x + width / 2, p95s, width, label="95th percentile")
    ax.set_xticks(x, ["Four-corner", "Internal-grid"])
    ax.set_ylabel("PnP solver time (ms)")
    ax.set_ylim(0, max(p95s) * 1.2)
    ax.grid(axis="y", alpha=0.17)
    ax.legend(frameon=False, fontsize=8)


def create_charts(pairs, summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 240})
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
    import cv2
    image = frame.copy()
    outer = np.asarray(tracker.board_corners, dtype=float).reshape(4, 2)
    projected = np.asarray(pose.projected_outer_corners, dtype=float).reshape(4, 2)
    axes = np.asarray(pose.projected_axes, dtype=float).reshape(4, 2)

    def polygon(points, color, width):
        cv2.polylines(image, [np.rint(points).astype(np.int32).reshape(-1, 1, 2)],
                      True, color, width, cv2.LINE_AA)

    polygon(outer, (0, 205, 255), 3)       # tracker: amber BGR
    polygon(projected, (240, 30, 220), 3) # pose projection: magenta BGR
    origin = tuple(np.rint(axes[0]).astype(int))
    for end, color, text_label in zip(axes[1:],
                                       [(10, 60, 255), (35, 225, 50), (255, 80, 25)],
                                       ["X", "Y", "Z"]):
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
    import cv2
    # Load project code only for this image. No model fitting is invented.
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import benchmark_pose as bm
    import benchmark_pose_methods as methods

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
        # side-by-side for optional appendix; preserve all native source pixels
        joined = np.concatenate((final, alt), axis=1)
        if joined.shape[1] > 2700:
            width = 2700
            height = round(joined.shape[0] * width / joined.shape[1])
            joined = cv2.resize(joined, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(outdir / "pose_example_methods.png"), joined)
    return key


def generate_tex(pairs, summary, outdir, example_ready, photo_total, video_total):
    count_points = int(sum(numeric(r, "manual_points") for r in pairs))
    n_photos = len(pairs)
    clear = sum(r["condition"] == "clear" for r in pairs)
    occ = sum(r["condition"] == "occluded" for r in pairs)
    med_four = med(pairs, "four_rmse_px")
    med_inner = med(pairs, "inner_rmse_px")
    p95_four = p95(pairs, "four_rmse_px")
    p95_inner = p95(pairs, "inner_rmse_px")
    square_four = med(pairs, "four_rmse_squares")
    square_inner = med(pairs, "inner_rmse_squares")
    wins = sum(numeric(r, "inner_rmse_px") < numeric(r, "four_rmse_px") for r in pairs)
    four = summary_row(summary, "video", "ALL", "four")
    inner = summary_row(summary, "video", "ALL", "inner")
    frame_n = int(numeric(four, "eligible_frames"))
    if frame_n != int(numeric(inner, "eligible_frames")):
        raise ValueError("Both methods must have identical eligible video frames")
    a_four = int(numeric(four, "accepted"))
    a_inner = int(numeric(inner, "accepted"))
    four_pnp = numeric(four, "pnp_median_ms")
    inner_pnp = numeric(inner, "pnp_median_ms")
    four_pose = numeric(four, "pose_total_median_ms")
    inner_pose = numeric(inner, "pose_total_median_ms")

    table = rf"""% Requires \usepackage{{booktabs}}
\begin{{table}}[t]
\centering
\caption{{Pose estimation results on tracker-valid observations. Photograph errors use manually annotated reference intersections withheld from internal-grid PnP fitting.}}
\label{{tab:pose_results}}
\small
\begin{{tabular}}{{lcc}}
\toprule
Metric & Four-corner & Internal-grid \\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{Annotated photos ({n_photos}, {count_points} points)}}}} \\
Median reprojection RMSE (px) & {med_four:.2f} & {med_inner:.2f} \\
95th-percentile RMSE (px) & {p95_four:.2f} & {p95_inner:.2f} \\
Median RMSE (square units) & {square_four:.3f} & {square_inner:.3f} \\
\midrule
\multicolumn{{3}}{{l}}{{\emph{{Tracked video frames ({frame_n:,})}}}} \\
Accepted poses & {a_four:,}/{frame_n:,} & {a_inner:,}/{frame_n:,} \\
Median PnP time (ms) & {four_pnp:.3f} & {inner_pnp:.3f} \\
Median pose-call time (ms) & {four_pose:.3f} & {inner_pose:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (outdir / "pose_results_table.tex").write_text(table, encoding="utf-8")

    section = rf"""% Results only. Keep causal interpretation in Discussion.
\subsection{{Pose Estimation}}
Of the {photo_total} photographs, valid tracker geometry was obtained for {n_photos}
({clear} unobstructed and {occ} partially occluded), providing {count_points} manually annotated
reference intersections. Both PnP methods returned accepted poses for all
{n_photos} photographs. The internal-grid method achieved lower reprojection RMSE
on {wins} of {n_photos} paired photographs, with a median of {med_inner:.2f} pixels
({square_inner:.3f} squares), compared with {med_four:.2f} pixels ({square_four:.3f} squares)
for the four-corner method. The corresponding 95th-percentile errors were
{p95_inner:.2f} and {p95_four:.2f} pixels, respectively (Table~\ref{{tab:pose_results}};
Fig.~\ref{{fig:pose_comparison}}).

Across the three video sequences, {frame_n:,} of {video_total:,} frames provided valid
tracker geometry. Four-corner PnP produced accepted poses in
{a_four:,}/{frame_n:,} eligible frames, compared with {a_inner:,}/{frame_n:,}
for internal-grid PnP. Median PnP solver times were {four_pnp:.3f} and
{inner_pnp:.3f}~ms, respectively, while the median pose-call times were
{four_pose:.3f} and {inner_pose:.3f}~ms.
"""
    if example_ready:
        section += r"Figure~\ref{fig:pose_example} shows the four-corner pose projected onto a representative webcam image." + "\n"
    (outdir / "pose_results_section.tex").write_text(section, encoding="utf-8")

    fig = r"""% For a two-column report, figure* spans both columns.
% Requires \usepackage{graphicx}.
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.88\textwidth]{pose_benchmark/figures/report/pose_results_combined.pdf}
    \caption{(a) Paired manual reprojection RMSE for photographs with valid tracked geometry; the dotted line indicates equal error. (b) Median and 95th-percentile PnP solver times across eligible video frames.}
    \label{fig:pose_comparison}
\end{figure*}
"""
    if example_ready:
        fig += r"""
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{pose_benchmark/figures/report/pose_example_four_corner.png}
    \caption{Four-corner PnP pose overlaid on a representative webcam photograph. Amber shows the tracked board boundary, magenta the pose-projected board boundary, and red, green and blue the projected axes at the fixed board origin.}
    \label{fig:pose_example}
\end{figure}
"""
    else:
        fig += r"""
% Once you run with --example, add:
% \begin{figure}[t]
%   \centering
%   \includegraphics[width=\columnwidth]{pose_benchmark/figures/report/pose_example_four_corner.png}
%   \caption{Four-corner PnP pose projected onto a representative webcam image.}
%   \label{fig:pose_example}
% \end{figure}
"""
    (outdir / "pose_figures.tex").write_text(fig, encoding="utf-8")
    print(f"Annotated photos={n_photos}/{photo_total}, points={count_points}, clear={clear}, occluded={occ}")
    print(f"Manual RMSE median: four={med_four:.3f}, inner={med_inner:.3f} px; wins={wins}/{n_photos}")
    print(f"Video eligible={frame_n}/{video_total}; accepted: four={a_four}, inner={a_inner}")
    print(f"Video median PnP times: four={four_pnp:.4f}ms, inner={inner_pnp:.4f}ms")


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
    generate_tex(pairs, summary, outdir, example_ready,
                 args.total_photos, args.total_video_frames)
    print("Saved report files to:", outdir)


if __name__ == "__main__":
    main()
