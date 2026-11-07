"""Paired, independent-image-space benchmark for chessboard pose methods.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import cv2
import numpy as np

# Shares ONLY the input pipeline, reference annotation metric and I/O helpers.
import benchmark_pose as bm

METHODS = ("four", "inner", "h", "h_refined")
FIELDS = (
    "source", "source_type", "condition", "frame", "time_s", "method",
    "tracking_ok", "orientation_status", "tracked_count", "available_inner_points",
    "fitted_points", "held_out_points", "pnp_solved", "pose_accepted",
    "pose_status", "pose_reason", "fit_corner_rmse_px", "fit_max_corner_px",
    "fit_inner_rmse_px", "fit_max_corner_squares", "manual_points",
    "manual_rmse_px", "manual_rmse_squares", "manual_median_px",
    "pnp_ms", "pose_total_ms", "gate_ms", "detect_ms", "tracker_ms",
)
SUMMARY_FIELDS = (
    "source_type", "group", "method", "eligible_frames", "solved", "accepted",
    "conditional_acceptance_pct", "annotated_photos", "manual_points_total",
    "manual_rmse_median_px", "manual_rmse_p95_px", "manual_rmse_median_squares",
    "corner_fit_median_px", "pnp_median_ms", "pnp_p95_ms", "pose_total_median_ms",
    "fitted_points_median", "held_out_points_median",
)
PAIR_FIELDS = (
    "image", "condition", "manual_points", "four_rmse_px", "inner_rmse_px",
    "inner_minus_four_px", "four_rmse_squares", "inner_rmse_squares",
    "four_pnp_ms", "inner_pnp_ms", "four_pose_total_ms", "inner_pose_total_ms",
    "four_accepted", "inner_accepted", "inner_fitted_points",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("all", "images", "videos"), default="all")
    p.add_argument("--data-dir", type=Path, default=bm.DATA_DIR)
    p.add_argument("--calibration", type=Path, default=bm.BASE_DIR / "cal_data.pkl")
    p.add_argument("--methods", default="four,inner",
                   help="Comma-separated: four,inner,h,h_refined (default: four,inner)")
    p.add_argument("--square-size", type=float, default=0.057)
    p.add_argument("--detect-every", type=int, default=60)
    p.add_argument("--max-frames", type=int, default=0, help="Per video; 0=all")
    p.add_argument("--pose-gate", choices=("relative", "pixels"), default="relative")
    p.add_argument("--max-corner-error-squares", type=float, default=0.60)
    p.add_argument("--max-corner-error", type=float, default=8.0)
    p.add_argument("--ransac-px", type=float, default=3.0,
                   help="RANSAC pixel tolerance for optional h/h_refined estimators")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--max-photo-overlays", type=int, default=12,
                   help="Maximum labelled-photo side-by-side figures (default 12)")
    args = p.parse_args(argv)
    args.methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    if not args.methods or len(args.methods) != len(set(args.methods)) or any(x not in METHODS for x in args.methods):
        p.error("--methods should contain unique values chosen from four,inner,h,h_refined")
    if args.detect_every < 1 or args.max_frames < 0 or args.square_size <= 0 or args.max_photo_overlays < 0:
        p.error("Invalid --detect-every, --max-frames or --square-size")
    if args.max_corner_error_squares <= 0 or args.max_corner_error <= 0 or args.ransac_px <= 0:
        p.error("Pose gate thresholds and RANSAC threshold must be positive")
    return args


def raw_inner_without_labels(tracker, points):
    """Copy tracked intersection array only; never mutate production tracker.

    labels: {(row,col): manually observed uv}; grid coordinates are 1..7.
    Pose method must not fit any grid INDEX used for manual validation.
    """
    arr = np.asarray(tracker.inner, dtype=np.float64)
    if arr.shape != (7, 7, 2):
        raise ValueError("Tracker.inner must have shape (7,7,2)")
    copy = arr.copy()
    for row, col in points:
        if not (1 <= row <= 7 and 1 <= col <= 7):
            raise ValueError("Manual grid point IDs must be in 1..7")
        copy[row-1, col-1] = np.nan
    # SimpleNamespace preserves all other TrackerResult attributes without
    # modifying the original, including .homography and .board_corners.
    clone = SimpleNamespace(**vars(tracker))
    clone.inner = copy
    return clone


def solve_methods(tracker, calibration, pipeline, args, manual=None, order_index=0):
    """Run same raw tracker result through all methods; same acceptance gate.

    Rotate the evaluation order across samples to reduce systematic
    first-call/warm-cache bias in per-method solver timing.
    """
    from get_pose_v7 import (get_inner_pnp_pose, get_undistorted_h_pose,
                             get_undistorted_h_refined_pose)
    if tracker is None or not bool(tracker.tracking_ok):
        return {m: (pipeline.pose_result_class(reason="Tracker unavailable"), None, float("nan"))
                for m in args.methods}
    holdout = manual or {}
    inner_source = raw_inner_without_labels(tracker, holdout) if holdout else tracker
    # The original get_pose_v6 and v7 methods use the same 3D board convention.
    # In relative mode, bypass their legacy pixel gate before applying the
    # same perspective-aware fraction-of-square acceptance to ALL methods.
    solver_px_gate = (args.max_corner_error if args.pose_gate == "pixels" else 1_000_000.0)
    out = {}
    # Keep the methods unchanged, but rotate their measurement order across
    # successive samples. In particular, the four-corner method must not
    # always be timed first. Returned records retain the original order.
    shift = int(order_index) % len(args.methods)
    evaluation_order = args.methods[shift:] + args.methods[:shift]
    for method in evaluation_order:
        if method == "four":
            pose = pipeline.get_pose(tracker, calibration.K, calibration.D,
                                    square_size=args.square_size,
                                    max_corner_error_px=solver_px_gate)
        elif method == "inner":
            pose = get_inner_pnp_pose(inner_source, calibration.K, calibration.D,
                        square_size=args.square_size, max_corner_error_px=solver_px_gate)
        elif method == "h":
            pose = get_undistorted_h_pose(inner_source, calibration.K, calibration.D,
                        square_size=args.square_size, max_corner_error_px=solver_px_gate,
                        ransac_threshold_px=args.ransac_px)
        elif method == "h_refined":
            pose = get_undistorted_h_refined_pose(inner_source, calibration.K, calibration.D,
                        square_size=args.square_size, max_corner_error_px=solver_px_gate,
                        ransac_threshold_px=args.ransac_px)
        gate = None
        gate_ms = float("nan")
        if pose.solved:
            start = time.perf_counter()
            try:
                gate = pipeline.evaluate_gate(
                    tracker.homography, pose.corner_errors_px,
                    mode=args.pose_gate,
                    max_corner_error_squares=args.max_corner_error_squares,
                    max_corner_error_px=args.max_corner_error)
                if pose.success:
                    pose.success = bool(gate.accepted)
                    pose.reason = gate.reason if pose.success else f"Boundary gate failed: {gate.reason}"
                cam = pose.camera_position_board
                if cam is not None and (not np.isfinite(cam).all() or float(cam[2]) <= 0.0):
                    pose.success = False
                    pose.reason = "Front-face guard rejected camera at/below board Z=0"
            except (RuntimeError, ValueError, TypeError, cv2.error) as ex:
                pose.success = False
                pose.reason = f"Pose gate invalid: {ex}"
            gate_ms = (time.perf_counter()-start)*1000.0
        out[method] = pose, gate, gate_ms
    return {method: out[method] for method in args.methods}


def make_records(key, kind, condition, frame_idx, seconds, result, outputs,
                 calib, square_size, manual=None, detect_ms=float("nan"),
                 tracker_ms=float("nan")):
    labelled = manual or {}
    valid = result is not None and bool(result.tracking_ok)
    avail_inner = (int(np.isfinite(np.asarray(result.inner)).all(axis=2).sum())
                   if valid else 0)
    actual_holdout = (sum(bool(np.isfinite(np.asarray(result.inner)[row-1,col-1]).all())
                          for row, col in labelled)
                      if valid else 0)
    records = []
    for method, (pose, gate, gate_ms) in outputs.items():
        n, rmse, med, fraction = bm.manual_reprojection(
            pose, labelled, calib.K, calib.D, square_size)
        records.append(dict(
            source=key, source_type=kind, condition=condition,
            frame=frame_idx, time_s=seconds, method=method,
            tracking_ok=int(valid),
            orientation_status=getattr(result, "orientation_status", "UNAVAILABLE"),
            tracked_count=int(getattr(result, "tracked_count", 0)) if valid else 0,
            available_inner_points=avail_inner,
            fitted_points=int(getattr(pose, "num_correspondences", 0)),
            held_out_points=(actual_holdout if method != "four" else 0),
            pnp_solved=int(pose.solved), pose_accepted=int(pose.success),
            pose_status=pose.status, pose_reason=pose.reason,
            fit_corner_rmse_px=pose.corner_rmse_px,
            fit_max_corner_px=pose.max_corner_error_px,
            fit_inner_rmse_px=pose.inner_reprojection_rmse_px,
            fit_max_corner_squares=(gate.max_error_squares if gate else float("nan")),
            manual_points=n, manual_rmse_px=rmse,
            manual_rmse_squares=fraction, manual_median_px=med,
            pnp_ms=pose.pnp_ms, pose_total_ms=pose.pose_ms,
            gate_ms=gate_ms, detect_ms=detect_ms, tracker_ms=tracker_ms,
        ))
    return records


def render_paired_overlay(frame, source, outputs, manual, calib, square_size):
    """Two raw-photo panels; manual clicks vs projected physical grid points.

    This is a VISUAL diagnostic only, not used in either PnP solve or scoring.
    """
    panels = []
    locations = sorted(manual)
    obj = np.array([[col*square_size, (8-row)*square_size, 0.]
                    for row, col in locations], dtype=np.float64)
    for method in ("four", "inner"):
        pose = outputs[method][0]
        canvas = frame.copy()
        if pose.solved and pose.rvec is not None:
            try:
                proj = cv2.projectPoints(obj, pose.rvec, pose.tvec,
                                         calib.K, calib.D)[0].reshape(-1, 2)
                for (row, col), uv in zip(locations, proj):
                    ref = np.asarray(manual[(row, col)], float)
                    a, b = tuple(np.rint(ref).astype(int)), tuple(np.rint(uv).astype(int))
                    cv2.line(canvas, a, b, (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.circle(canvas, a, 6, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.drawMarker(canvas, b, (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
                corners = np.asarray(getattr(pose, "projected_outer_corners", None))
                if corners.shape == (4,2) and np.isfinite(corners).all():
                    cv2.polylines(canvas, [np.rint(corners).astype(np.int32).reshape(-1,1,2)],
                                  True, (255, 0, 255), 2, cv2.LINE_AA)
            except cv2.error:
                pass
        cv2.rectangle(canvas, (0,0), (canvas.shape[1],100), (28,28,28), -1)
        n, rmse, _, _ = bm.manual_reprojection(pose, manual, calib.K, calib.D, square_size)
        cv2.putText(canvas, f"{method.upper()} | {pose.status} | RMSE {rmse:.2f}px",
                    (24,38), cv2.FONT_HERSHEY_SIMPLEX, .9, (245,245,245), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Yellow dot = manual | magenta cross = projected | orange = error",
                    (24,76), cv2.FONT_HERSHEY_SIMPLEX, .55, (220,220,220), 1, cv2.LINE_AA)
        panels.append(canvas)
    # Resize display only AFTER all native-pixel geometry computations.
    scale = min(1., 1050 / frame.shape[1])
    display = [cv2.resize(x, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA) if scale < 1 else x for x in panels]
    return np.concatenate(display, axis=1)


def process_images(args, pipe, calib, labels):
    records = []
    saved_overlays = 0
    inputs = bm.files_under(args.data_dir / "images", bm.IMAGES_EXT)
    if not inputs:
        print("No photos under images/clear or images/occluded")
    for i, path in enumerate(inputs, 1):
        key = bm.relative_key(path, args.data_dir)
        condition = "occluded" if "occluded" in path.parts else "clear" if "clear" in path.parts else "unlabelled"
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print("Skipping unreadable image", path)
            continue
        bm.check_image_geometry(frame, calib, key)
        tracker = pipe.tracker_class()
        result, _, message, detect_ms, tracker_ms = bm.detect_or_track(frame, tracker, pipe, True)
        # Include detection failures in raw records for traceability, but ALL
        # numeric method summaries condition on a valid tracker result.
        outputs = solve_methods(result, calib, pipe, args, labels.get(key, {}), order_index=i-1)
        records.extend(make_records(key,"image",condition,1,0.0,result,outputs,
                                     calib,args.square_size,labels.get(key,{}),detect_ms,tracker_ms))
        if (not args.no_figures and saved_overlays < args.max_photo_overlays and
                all(m in outputs for m in ("four", "inner")) and
                result is not None and result.tracking_ok and labels.get(key)):
            overlay = render_paired_overlay(frame, key, outputs, labels[key], calib, args.square_size)
            destination = args.data_dir / "figures/methods/paired_photos" / path.relative_to(args.data_dir/"images")
            bm.save_image(destination.with_suffix(".png"), overlay)
            saved_overlays += 1
        if i % 5 == 0 or i == len(inputs):
            names = ", ".join(m+":"+outputs[m][0].status.replace("POSE ","") for m in args.methods)
            print(f"[photos {i}/{len(inputs)}] {key}: {names}", flush=True)
    return records


def process_videos(args, pipe, calib, intervals):
    records=[]
    videos=bm.files_under(args.data_dir / "videos", bm.VIDEO_EXT)
    for i, path in enumerate(videos, 1):
        key=bm.relative_key(path,args.data_dir)
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print("Skipping unreadable video",path)
            continue
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0: fps=float("nan")
        tracker=pipe.tracker_class()
        n=0
        print(f"[video {i}/{len(videos)}] {key} frames starting",flush=True)
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                n+=1
                if n==1: bm.check_image_geometry(frame,calib,key)
                t=(n-1)/fps if np.isfinite(fps) else float("nan")
                phase=bm.video_phase(t,intervals.get(key,[]))
                detector_needed=(not tracker.initialised or n%args.detect_every==0
                                 or not tracker.last_tracking_ok)
                result,_,msg,detect_ms,tracker_ms=bm.detect_or_track(frame,tracker,pipe,detector_needed)
                outputs=solve_methods(result,calib,pipe,args,order_index=n-1)
                records.extend(make_records(key,"video",phase,n,t,result,outputs,
                                            calib,args.square_size,None,detect_ms,tracker_ms))
                if n%120==0:
                    print(f"  {n} frames",flush=True)
                if args.max_frames and n>=args.max_frames:break
        finally:
            cap.release()
        print(f"  completed {n} frames",flush=True)
    return records


def summaries(records):
    """Pose success per tracked frame, NOT raw-frame detection availability."""
    groups={}
    for r in records:
        if r["tracking_ok"] != 1:continue
        categories=[(r["source_type"],"ALL")]
        if r["source_type"] == "image": categories.append(("image",r["condition"]))
        if r["source_type"] == "video": categories.append(("video",r["source"]))
        for src_group in categories:
            groups.setdefault((*src_group,r["method"]),[]).append(r)
    out=[]
    for (kind,group,method), rr in sorted(groups.items()):
        solved=[r for r in rr if r["pnp_solved"]]
        labelled=[r for r in rr if r["manual_points"] > 0 and
                  np.isfinite(float(r["manual_rmse_px"]))]
        out.append(dict(source_type=kind,group=group,method=method,
                        eligible_frames=len(rr),solved=len(solved),
                        accepted=sum(r["pose_accepted"] for r in rr),
                        conditional_acceptance_pct=100*sum(r["pose_accepted"] for r in rr)/len(rr),
                        annotated_photos=len(labelled),
                        manual_points_total=sum(r["manual_points"] for r in labelled),
                        manual_rmse_median_px=bm.median(r["manual_rmse_px"] for r in labelled),
                        manual_rmse_p95_px=bm.p95(r["manual_rmse_px"] for r in labelled),
                        manual_rmse_median_squares=bm.median(r["manual_rmse_squares"] for r in labelled),
                        corner_fit_median_px=bm.median(r["fit_corner_rmse_px"] for r in solved),
                        pnp_median_ms=bm.median(r["pnp_ms"] for r in solved),
                        pnp_p95_ms=bm.p95(r["pnp_ms"] for r in solved),
                        pose_total_median_ms=bm.median(r["pose_total_ms"] for r in rr),
                        fitted_points_median=bm.median(r["fitted_points"] for r in solved),
                        held_out_points_median=bm.median(r["held_out_points"] for r in rr),
                        ))
    return out


def paired_photos(rows):
    """Only within-photo, independently labelled, two-method comparisons."""
    per={}
    for r in rows:
        if r["tracking_ok"] and r["manual_points"]:
            per.setdefault(r["source"],{})[r["method"]]=r
    out=[]
    for image, r in sorted(per.items()):
        if not all(k in r for k in ("four","inner")):continue
        f,i=r["four"],r["inner"]
        if not (np.isfinite(f["manual_rmse_px"]) and np.isfinite(i["manual_rmse_px"])):
            continue
        out.append(dict(image=image,condition=f["condition"],manual_points=f["manual_points"],
                        four_rmse_px=f["manual_rmse_px"],inner_rmse_px=i["manual_rmse_px"],
                        inner_minus_four_px=i["manual_rmse_px"]-f["manual_rmse_px"],
                        four_rmse_squares=f["manual_rmse_squares"],
                        inner_rmse_squares=i["manual_rmse_squares"],
                        four_pnp_ms=f["pnp_ms"],inner_pnp_ms=i["pnp_ms"],
                        four_pose_total_ms=f["pose_total_ms"],inner_pose_total_ms=i["pose_total_ms"],
                        four_accepted=f["pose_accepted"],inner_accepted=i["pose_accepted"],
                        inner_fitted_points=i["fitted_points"]))
    return out


def plot_results(args, images, videos, pairs):
    if args.no_figures:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target=args.data_dir/"figures/methods"
    target.mkdir(parents=True,exist_ok=True)
    if pairs:
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        for condition, marker in (("clear","o"),("occluded","s"),("unlabelled","^")):
            rr=[r for r in pairs if r["condition"]==condition]
            if rr:
                ax.scatter([r["four_rmse_px"] for r in rr],
                           [r["inner_rmse_px"] for r in rr],label=condition,marker=marker,s=44)
        upper=max([0.1]+[max(r["four_rmse_px"],r["inner_rmse_px"]) for r in pairs])*1.07
        ax.plot([0,upper],[0,upper],linestyle="--",linewidth=1)
        ax.set(xlabel="Four-corner PnP: manual RMSE (px)",
               ylabel="Internal-grid PnP: manual RMSE (px)",
               title="Paired image-space accuracy (same manual references)")
        ax.set_xlim(0,upper); ax.set_ylim(0,upper)
        ax.grid(alpha=.25); ax.legend()
        fig.tight_layout()
        fig.savefig(target/"pose_methods_photo_comparison.png",dpi=180)
        plt.close(fig)
    all_rows=images+videos
    usable=[r for r in all_rows if r["tracking_ok"] and r["pnp_solved"]]
    if usable:
        fig,ax=plt.subplots(figsize=(7.5,4.3))
        keys=[m for m in args.methods if any(r["method"]==m for r in usable)]
        data=[[r["pnp_ms"] for r in usable if r["method"]==m and
               np.isfinite(float(r["pnp_ms"]))] for m in keys]
        data=[d for d in data if d]
        plotkeys=[m for m in keys if any(r["method"]==m and
                      np.isfinite(float(r["pnp_ms"])) for r in usable)]
        if data:
            ax.boxplot(data,tick_labels=plotkeys,showfliers=False)
            ax.set(ylabel="PnP / estimation kernel time (ms)",
                   title="Per-frame solver time, eligible tracker frames")
            ax.grid(axis="y",alpha=.25)
            fig.tight_layout()
            fig.savefig(target/"pose_methods_runtime.png",dpi=180)
        plt.close(fig)


def main(argv=None):
    args=parse_args(argv)
    args.data_dir=args.data_dir.resolve()
    bm.folders(args)
    pipe=bm.load_pipeline()
    calib=pipe.load_calibration(args.calibration)
    # Make all experiment methods available in main project (legacy v7 module).
    try:
        import get_pose_v7
    except ModuleNotFoundError as ex:
        raise RuntimeError("Place the existing get_pose_v7.py beside benchmark_pose_methods.py "
                           "to benchmark the original internal-grid PnP implementation.") from ex
    labels=bm.load_manual_points(args.data_dir)
    intervals=bm.load_occlusion_intervals(args.data_dir) if args.mode in ("all","videos") else {}
    print("Methods:",", ".join(args.methods),"; calibration:",args.calibration)
    print("Internal-grid photos: all manually labelled indices WITHHELD from fit.")
    images=process_images(args,pipe,calib,labels) if args.mode in ("all","images") else []
    videos=process_videos(args,pipe,calib,intervals) if args.mode in ("all","videos") else []
    out=args.data_dir/"results"
    bm.write_csv(out/"pose_method_images.csv",images,FIELDS)
    bm.write_csv(out/"pose_method_video_frames.csv",videos,FIELDS)
    sm=summaries(images+videos)
    bm.write_csv(out/"pose_method_summary.csv",sm,SUMMARY_FIELDS)
    pp=paired_photos(images)
    bm.write_csv(out/"pose_method_pairs.csv",pp,PAIR_FIELDS)
    plot_results(args,images,videos,pp)
    for row in sm:
        if row["group"]=="ALL":
            print(f"{row['source_type']} {row['method']:<10} "
                  f"eligible={row['eligible_frames']:4} accepted={row['accepted']:4} "
                  f"manual={row['annotated_photos']:3} "
                  f"manual-RMSE={row['manual_rmse_median_px']:.2f}px "
                  f"PnP={row['pnp_median_ms']:.3f}ms")
    if not pp:
        print("No paired manual photo accuracy yet. Run annotate_pose_images.py "
              "and ensure at least six *non-annotated* tracked intersections remain.")
    print("Completed. Existing benchmark_pose.py outputs remain untouched.")

if __name__=="__main__":
    main()
