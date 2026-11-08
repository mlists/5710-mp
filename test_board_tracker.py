"""
Benchmark BoardTracker against find_board.detect_chessboard as ground truth.

How it works
------------
1. Ground truth: the detector is run on every processed frame of each video.
   Results are cached on disk, so sweeping tracker settings is cheap.
2. Tracker run: the tracker is initialised on the first frame where the
   detector succeeds. After that it only sees frames, except that every
   --redetect-every N frames it is handed the detector's result (this is the
   "how often we find the board" knob; 0 = never after init).
3. Comparison: on every frame the tracker's features are compared with the
   detector's features for that same frame:
       inner   - the 7x7 internal intersections (measured vs measured)
       centres - the 8x8 square centres
       corners - the 4 outer board corners
   The detector's corner ordering is arbitrary, so the best of the 8
   rotation/flip assignments is chosen per frame before comparing.

Examples
--------
    python test_board_tracker.py videos\5.mp4
    python test_board_tracker.py videos\5.mp4 --redetect-every 0 15 30 60 --plot
    python test_board_tracker.py "videos\*.mp4" --redetect-every 30 --redetect-on-loss
    python test_board_tracker.py "videos\*.mp4" --tracker-params ransac_threshold=2 max_lk_error=20

Outputs (in --out)
------------------
    summary.json                         all statistics
    <video>_every<N>_frames.csv          one row per frame
    <video>_every<N>_features.npz        per-feature errors for every frame
    <video>_every<N>.png                 plots (with --plot)
    gt_cache/                            cached detector output
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import find_board
from board_tracker import BoardTracker
from find_board import detect_chessboard

# Frames-since-detection bins used for the drift-vs-age table (log2 spaced).
AGE_BIN_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, np.inf]


# --------------------------------------------------------------------------- #
# Video / ground truth
# --------------------------------------------------------------------------- #
def iter_frames(path, start, stride, max_frames, scale):
    """Yield (original_frame_index, BGR frame) honouring start/stride/limit."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    idx = -1
    yielded = 0
    try:
        while True:
            if not cap.grab():
                break
            idx += 1
            if idx < start or (idx - start) % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            yield idx, frame
            yielded += 1
            if max_frames and yielded >= max_frames:
                break
    finally:
        cap.release()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, count


@dataclass
class GroundTruth:
    frame_idx: np.ndarray   # (N,) original video frame numbers
    found: np.ndarray       # (N,) bool, detector succeeded
    centres: np.ndarray     # (N, 8, 8, 2)
    corners: np.ndarray     # (N, 4, 2)
    inner: np.ndarray       # (N, 7, 7, 2), NaN where occluded/unmatched
    detect_ms: np.ndarray   # (N,)
    fps: float


def gt_cache_path(video: Path, args):
    st = video.stat()
    key = "|".join(str(v) for v in (
        video.resolve(), st.st_size, st.st_mtime_ns, args.start, args.stride,
        args.max_frames, args.scale, args.inlier_threshold,
        args.standard_detector,
    ))
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return Path(args.out) / "gt_cache" / f"{video.stem}_{digest}.npz"


def compute_ground_truth(video: Path, args) -> GroundTruth:
    cache = gt_cache_path(video, args)
    if cache.exists() and not args.no_cache and not args.refresh_gt:
        with np.load(cache) as z:
            print(f"  ground truth: loaded cache {cache.name}")
            return GroundTruth(z["frame_idx"], z["found"], z["centres"],
                               z["corners"], z["inner"], z["detect_ms"],
                               float(z["fps"]))

    fps, total = video_info(video)
    total = total if not args.max_frames else min(total or args.max_frames,
                                                  args.max_frames * args.stride)
    idxs, found, centres, corners, inner, ms = [], [], [], [], [], []
    nan = lambda *s: np.full(s, np.nan, np.float32)  # noqa: E731
    t_start = time.perf_counter()
    for idx, frame in iter_frames(video, args.start, args.stride,
                                  args.max_frames, args.scale):
        t0 = time.perf_counter()
        try:
            c, k, i = detect_chessboard(
                frame,
                inlier_threshold=args.inlier_threshold,
                use_standard_detector=args.standard_detector,
            )
            ok = bool(np.isfinite(c).all() and np.isfinite(k).all())
        except Exception:  # detector raises RuntimeError when nothing found
            ok = False
        ms.append((time.perf_counter() - t0) * 1000.0)
        idxs.append(idx)
        found.append(ok)
        centres.append(np.asarray(c, np.float32) if ok else nan(8, 8, 2))
        corners.append(np.asarray(k, np.float32) if ok else nan(4, 2))
        inner.append(np.asarray(i, np.float32) if ok else nan(7, 7, 2))
        if len(idxs) % 50 == 0 and not args.quiet:
            done = len(idxs)
            pct = f" / ~{total // args.stride}" if total else ""
            print(f"\r  ground truth: {done}{pct} frames "
                  f"({done / (time.perf_counter() - t_start):.1f} fps)",
                  end="", file=sys.stderr, flush=True)
    if not args.quiet:
        print(file=sys.stderr)
    if not idxs:
        raise RuntimeError(f"No frames read from {video}")

    gt = GroundTruth(np.array(idxs), np.array(found, bool), np.stack(centres),
                     np.stack(corners), np.stack(inner), np.array(ms), fps)
    print(f"  ground truth: detector found the board in "
          f"{gt.found.sum()}/{len(gt.found)} frames "
          f"(mean {gt.detect_ms.mean():.1f} ms/frame)")
    if not args.no_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, frame_idx=gt.frame_idx, found=gt.found,
                            centres=gt.centres, corners=gt.corners,
                            inner=gt.inner, detect_ms=gt.detect_ms,
                            fps=np.float64(gt.fps))
    return gt


# --------------------------------------------------------------------------- #
# Tracker run
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    video: str
    interval: int
    frame_idx: np.ndarray
    time_s: np.ndarray
    is_detect: np.ndarray       # tracker was (re)initialised from the detector
    age: np.ndarray             # frames since the tracker last saw a detection
    tracking_ok: np.ndarray
    tracked_count: np.ndarray
    status: list
    gt_found: np.ndarray
    align_idx: np.ndarray       # which of the 8 GT orderings matched (-1: none)
    update_ms: np.ndarray
    square_px: np.ndarray       # GT mean square width, for normalising errors
    inner_err: np.ndarray       # (M, 7, 7)  px, NaN where either side missing
    centre_err: np.ndarray      # (M, 8, 8)
    corner_err: np.ndarray      # (M, 4)
    gt_detect_ms: np.ndarray    # detector time on the same frames
    redetect_rejected: int
    init_failures: int


def align_and_compare(result, gt_c, gt_k, gt_i):
    """Pick the GT ordering closest to the tracker and return per-feature errors."""
    if not np.isfinite(result.square_centres).all():
        return -1, None
    best, best_err = -1, np.inf
    best_t = None
    j = 0
    for flip in (False, True):
        for k in range(4):
            t = BoardTracker._transform_detection(gt_c, gt_k, gt_i, k, flip)
            err = float(np.linalg.norm(t[0] - result.square_centres, axis=2).mean())
            if err < best_err:
                best, best_err, best_t = j, err, t
            j += 1
    c, k, i = best_t
    centre_err = np.linalg.norm(c - result.square_centres, axis=2)
    corner_err = np.linalg.norm(k - result.board_corners, axis=1)
    inner_err = np.linalg.norm(i - result.inner, axis=2)  # NaN propagates
    return best, (inner_err, centre_err, corner_err)


def _gt_square_px(centres):
    d_cols = np.linalg.norm(np.diff(centres, axis=1), axis=-1).mean()
    d_rows = np.linalg.norm(np.diff(centres, axis=0), axis=-1).mean()
    return float(0.5 * (d_cols + d_rows))


def run_tracker(video: Path, gt: GroundTruth, interval: int, args) -> RunResult:
    tracker = BoardTracker(**args.tracker_params)
    nan = np.nan
    rows = {k: [] for k in (
        "frame_idx", "time_s", "is_detect", "age", "tracking_ok",
        "tracked_count", "status", "gt_found", "align_idx", "update_ms",
        "square_px", "gt_detect_ms")}
    inner_e, centre_e, corner_e = [], [], []
    last_detect_n = None
    prev_ok = False
    rejected = 0
    init_failures = 0

    for n, (idx, frame) in enumerate(iter_frames(
            video, args.start, args.stride, args.max_frames, args.scale)):
        if n >= len(gt.frame_idx):
            break
        if idx != gt.frame_idx[n]:
            print(f"  warning: frame index mismatch at {n} "
                  f"({idx} vs {gt.frame_idx[n]}); stopping this run early")
            break

        has_gt = bool(gt.found[n])
        detection = (gt.centres[n], gt.corners[n], gt.inner[n]) if has_gt else None
        used_detection = False

        if not tracker.initialised:
            if detection is None:
                continue
            t0 = time.perf_counter()
            try:
                result = tracker.initialize(frame, detection)
            except (ValueError, RuntimeError):
                # initialize() sets initialised=True before it can raise
                # OrientationUnverified, so the tracker must be reset.
                tracker.reset()
                init_failures += 1
                continue
            update_ms = (time.perf_counter() - t0) * 1000.0
            used_detection = True
        else:
            age = n - last_detect_n
            due = interval > 0 and age >= interval
            if args.redetect_on_loss and not prev_ok:
                due = True
            gray = None
            if due and detection is not None:
                # NOTE: BoardTracker._align_redetection() needs
                # _orientation_gray, which board_tracker.py never assigns, so
                # update(frame, detection) would always raise RuntimeError.
                gray = BoardTracker._as_gray(frame)
            t0 = time.perf_counter()
            if gray is not None:
                tracker._orientation_gray = gray
                try:
                    result = tracker.update(frame, detection=detection)
                    used_detection = True
                except (ValueError, RuntimeError):
                    # Redetection rejected (e.g. orientation unverified); the
                    # tracker state is untouched, so carry on tracking.
                    rejected += 1
                    result = tracker.update(frame)
                finally:
                    tracker._orientation_gray = None
            else:
                result = tracker.update(frame)
            update_ms = (time.perf_counter() - t0) * 1000.0

        if used_detection:
            last_detect_n = n
        prev_ok = bool(result.tracking_ok)

        align = -1
        errs = None
        if has_gt:
            align, errs = align_and_compare(
                result, gt.centres[n], gt.corners[n], gt.inner[n])
        if errs is None:
            inner_e.append(np.full((7, 7), nan, np.float32))
            centre_e.append(np.full((8, 8), nan, np.float32))
            corner_e.append(np.full(4, nan, np.float32))
        else:
            inner_e.append(errs[0].astype(np.float32))
            centre_e.append(errs[1].astype(np.float32))
            corner_e.append(errs[2].astype(np.float32))

        rows["frame_idx"].append(idx)
        rows["time_s"].append(idx / gt.fps if gt.fps > 0 else nan)
        rows["is_detect"].append(used_detection)
        rows["age"].append(n - last_detect_n)
        rows["tracking_ok"].append(bool(result.tracking_ok))
        rows["tracked_count"].append(int(result.tracked_count))
        rows["status"].append(str(getattr(result, "orientation_status", "")))
        rows["gt_found"].append(has_gt)
        rows["align_idx"].append(align)
        rows["update_ms"].append(update_ms)
        rows["square_px"].append(_gt_square_px(gt.centres[n]) if has_gt else nan)
        rows["gt_detect_ms"].append(float(gt.detect_ms[n]))

    if not rows["frame_idx"]:
        raise RuntimeError(
            f"{video.name}: tracker never initialised (no usable detection)")

    return RunResult(
        video=video.name, interval=interval,
        frame_idx=np.array(rows["frame_idx"]),
        time_s=np.array(rows["time_s"], float),
        is_detect=np.array(rows["is_detect"], bool),
        age=np.array(rows["age"]),
        tracking_ok=np.array(rows["tracking_ok"], bool),
        tracked_count=np.array(rows["tracked_count"]),
        status=rows["status"],
        gt_found=np.array(rows["gt_found"], bool),
        align_idx=np.array(rows["align_idx"]),
        update_ms=np.array(rows["update_ms"], float),
        square_px=np.array(rows["square_px"], float),
        inner_err=np.stack(inner_e), centre_err=np.stack(centre_e),
        corner_err=np.stack(corner_e),
        gt_detect_ms=np.array(rows["gt_detect_ms"], float),
        redetect_rejected=rejected, init_failures=init_failures,
    )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _stats(a):
    a = np.asarray(a, float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def _per_frame(err):
    """Per-frame mean/max over the feature axes, NaN-safe."""
    flat = err.reshape(len(err), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(flat, axis=1)
        mx = np.nanmax(flat, axis=1)
    return mean, mx


def summarise(runs):
    """Aggregate one or more RunResults (pooled) into a statistics dict."""
    cat = lambda name: np.concatenate([getattr(r, name) for r in runs])  # noqa: E731
    is_detect, ok, gt_found = cat("is_detect"), cat("tracking_ok"), cat("gt_found")
    age = cat("age")
    inner, centre, corner = cat("inner_err"), cat("centre_err"), cat("corner_err")
    sq = cat("square_px")
    upd = cat("update_ms")
    det_ms = cat("gt_detect_ms")

    # Frames where the tracker is on its own and there is something to compare.
    evalm = gt_found & ~is_detect
    sq_b = sq[evalm]

    out = {
        "frames": int(len(is_detect)),
        "frames_with_gt": int(gt_found.sum()),
        "frames_evaluated": int(evalm.sum()),
        "detections_used": int(is_detect.sum()),
        "redetect_rejected": int(sum(r.redetect_rejected for r in runs)),
        "init_failures": int(sum(r.init_failures for r in runs)),
        "tracking_lost_frames": int((~ok).sum()),
        "tracking_lost_pct": float(100.0 * (~ok).mean()),
    }
    # Loss events = ok -> not ok transitions within each run
    events = 0
    for r in runs:
        events += int(np.sum(r.tracking_ok[:-1] & ~r.tracking_ok[1:]))
    out["tracking_lost_events"] = events

    for name, err in (("centres", centre), ("inner", inner), ("corners", corner)):
        e = err[evalm]
        out[f"{name}_px"] = _stats(e)
        shape = (-1,) + (1,) * (e.ndim - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"{name}_squares"] = _stats(e / sq_b.reshape(shape))
        pf_mean, pf_max = _per_frame(e)
        out[f"{name}_frame_mean_px"] = _stats(pf_mean)
        out[f"{name}_frame_max_px"] = _stats(pf_max)

    # Also with tracker-reported-OK frames only (so "lost" frames don't dominate)
    ok_eval = evalm & ok
    out["centres_px_tracking_ok"] = _stats(centre[ok_eval])
    out["inner_px_tracking_ok"] = _stats(inner[ok_eval])

    # Drift vs frames since last detection
    by_age = []
    lo = AGE_BIN_EDGES[0]
    for hi in AGE_BIN_EDGES[1:]:
        m = evalm & (age >= lo) & (age < hi)
        if m.any():
            s = _stats(centre[m])
            by_age.append({
                "age_from": int(lo),
                "age_to": None if np.isinf(hi) else int(hi - 1),
                "frames": int(m.sum()),
                "centre_mean_px": s.get("mean"),
                "centre_p95_px": s.get("p95"),
                "lost_pct": float(100.0 * (~ok[m]).mean()),
            })
        lo = hi
    out["error_by_frames_since_detection"] = by_age

    # Timing
    track_only = ~is_detect
    out["tracker_ms"] = _stats(upd[track_only])
    out["detector_ms"] = _stats(det_ms[gt_found])
    if out["tracker_ms"].get("mean") and out["detector_ms"].get("mean"):
        out["speedup_vs_detector"] = (
            out["detector_ms"]["mean"] / out["tracker_ms"]["mean"])
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def run_tag(r: RunResult):
    return f"{Path(r.video).stem}_every{r.interval}"


def write_frames_csv(r: RunResult, path: Path):
    c_mean, c_max = _per_frame(r.centre_err)
    i_mean, i_max = _per_frame(r.inner_err)
    k_mean, k_max = _per_frame(r.corner_err)
    i_n = np.isfinite(r.inner_err.reshape(len(r.inner_err), -1)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c_mean_sq = c_mean / r.square_px
    cols = [
        ("frame", r.frame_idx), ("time_s", r.time_s),
        ("is_detect", r.is_detect.astype(int)), ("frames_since_detect", r.age),
        ("tracking_ok", r.tracking_ok.astype(int)),
        ("tracked_count", r.tracked_count), ("orientation_status", r.status),
        ("gt_found", r.gt_found.astype(int)), ("gt_alignment", r.align_idx),
        ("centre_mean_px", c_mean), ("centre_max_px", c_max),
        ("centre_mean_squares", c_mean_sq),
        ("inner_mean_px", i_mean), ("inner_max_px", i_max),
        ("inner_compared", i_n),
        ("corner_mean_px", k_mean), ("corner_max_px", k_max),
        ("square_px", r.square_px), ("update_ms", r.update_ms),
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols])
        for row in zip(*[c[1] for c in cols]):
            w.writerow([
                "" if (isinstance(v, (float, np.floating)) and not np.isfinite(v))
                else (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v)
                for v in row])


def write_features_npz(r: RunResult, path: Path):
    np.savez_compressed(
        path, frame_idx=r.frame_idx, is_detect=r.is_detect, age=r.age,
        tracking_ok=r.tracking_ok, inner_err=r.inner_err,
        centre_err=r.centre_err, corner_err=r.corner_err,
        square_px=r.square_px)


def plot_run(r: RunResult, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_mean, c_max = _per_frame(r.centre_err)
    x = np.arange(len(r.frame_idx))
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(r.frame_idx, c_max, color="tab:orange", lw=0.8, label="max over squares")
    ax.plot(r.frame_idx, c_mean, color="tab:blue", lw=1.2, label="mean over squares")
    lost = ~r.tracking_ok
    if lost.any():
        ymax = np.nanmax(c_max) if np.isfinite(c_max).any() else 1
        ax.fill_between(r.frame_idx, 0, ymax, where=lost, color="red",
                        alpha=0.12, label="tracker reports lost")
    for f in r.frame_idx[r.is_detect]:
        ax.axvline(f, color="green", alpha=0.35, lw=0.7)
    ax.plot([], [], color="green", alpha=0.5, label="detector used")
    ax.set_xlabel("video frame")
    ax.set_ylabel("square-centre error (px)")
    ax.set_title(f"{r.video} - redetect every {r.interval or 'never'}")
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    m = ~r.is_detect & r.gt_found
    ages = np.unique(r.age[m])
    if len(ages):
        means = [np.nanmean(c_mean[m & (r.age == a)]) for a in ages]
        ax.plot(ages, means, ".-", ms=3)
    ax.set_xlabel("frames since last detection")
    ax.set_ylabel("mean centre error (px)")
    ax.set_title("Drift vs time since detection")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        heat = np.nanmean(r.inner_err[m], axis=0) if m.any() else np.full((7, 7), np.nan)
    im = ax.imshow(heat, cmap="viridis")
    ax.set_title("Mean error per internal intersection (px)")
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    fig.colorbar(im, ax=ax)
    for (row, col), v in np.ndenumerate(heat):
        if np.isfinite(v):
            ax.text(col, row, f"{v:.1f}", ha="center", va="center",
                    color="w", fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_sweep(rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["interval"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [r["centres_px"].get("mean", np.nan) for r in rows], "o-", label="mean")
    ax.plot(xs, [r["centres_px"].get("p95", np.nan) for r in rows], "s--", label="p95")
    ax.set_xlabel("redetect interval (frames, 0 = never)")
    ax.set_ylabel("square-centre error (px)")
    ax.set_title("Accuracy vs detection frequency (all videos pooled)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def print_table(rows):
    hdr = (f"{'video':<22} {'every':>6} {'frames':>7} {'det':>5} {'lost%':>6} "
           f"{'ctr mean':>9} {'ctr med':>8} {'ctr p95':>8} {'ctr max':>8} "
           f"{'inner mean':>10} {'corner mean':>11} {'trk ms':>7} {'speedup':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    f = lambda d, k: f"{d[k]:.2f}" if d.get(k) is not None else "-"  # noqa: E731
    for r in rows:
        s = r["stats"]
        sp = s.get("speedup_vs_detector")
        print(f"{r['video'][:22]:<22} {r['interval']:>6} {s['frames']:>7} "
              f"{s['detections_used']:>5} {s['tracking_lost_pct']:>6.1f} "
              f"{f(s['centres_px'], 'mean'):>9} {f(s['centres_px'], 'median'):>8} "
              f"{f(s['centres_px'], 'p95'):>8} {f(s['centres_px'], 'max'):>8} "
              f"{f(s['inner_px'], 'mean'):>10} {f(s['corners_px'], 'mean'):>11} "
              f"{f(s['tracker_ms'], 'mean'):>7} "
              f"{(f'{sp:.0f}x' if sp else '-'):>7}")
    print("\nErrors are in pixels, over frames where the tracker ran without "
          "being handed a detection.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_tracker_params(items):
    params = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected key=value, got '{item}'")
        k, v = item.split("=", 1)
        params[k] = float(v)
    return params


def build_parser():
    p = argparse.ArgumentParser(
        description="Benchmark BoardTracker against detect_chessboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+",
                   help="video files (globs allowed, e.g. 'clips/*.mp4')")
    p.add_argument("--redetect-every", type=int, nargs="+", default=[0],
                   metavar="N",
                   help="hand the tracker a detection every N processed frames. "
                        "0 = only initialise. Give several values to sweep. "
                        "(default: 0)")
    p.add_argument("--redetect-on-loss", action="store_true",
                   help="also redetect as soon as the tracker reports it is lost")
    p.add_argument("--start", type=int, default=0, help="first frame to use")
    p.add_argument("--stride", type=int, default=1,
                   help="use every Nth video frame (simulates a lower frame rate)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after this many processed frames (0 = all)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="resize frames before detection/tracking (errors are then "
                        "in scaled pixels)")
    p.add_argument("--inlier-threshold", type=float,
                   default=find_board.DEFAULT_INLIER_THRESHOLD,
                   help="detector inlier threshold in px")
    p.add_argument("--standard-detector", action="store_true",
                   help="let the detector try cv2.findChessboardCorners first")
    p.add_argument("--tracker-params", nargs="*", metavar="KEY=VALUE",
                   help="BoardTracker constructor overrides, e.g. "
                        "ransac_threshold=2 max_lk_error=20")
    p.add_argument("--out", default="benchmark_out", help="output directory")
    p.add_argument("--no-cache", action="store_true",
                   help="neither read nor write the ground-truth cache")
    p.add_argument("--refresh-gt", action="store_true",
                   help="recompute ground truth and overwrite the cache")
    p.add_argument("--plot", action="store_true", help="write PNG plots")
    p.add_argument("--no-csv", action="store_true",
                   help="skip the per-frame CSV and NPZ files")
    p.add_argument("--quiet", action="store_true", help="less progress output")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stride < 1:
        sys.exit("--stride must be >= 1")
    if any(n < 0 for n in args.redetect_every):
        sys.exit("--redetect-every values must be >= 0")
    args.tracker_params = parse_tracker_params(args.tracker_params)

    videos = []
    for pat in args.videos:
        hits = sorted(glob.glob(pat)) or [pat]
        videos.extend(Path(h) for h in hits)
    missing = [v for v in videos if not v.exists()]
    if missing:
        sys.exit("Not found: " + ", ".join(map(str, missing)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    intervals = list(dict.fromkeys(args.redetect_every))

    all_runs = {n: [] for n in intervals}
    rows = []
    for video in videos:
        print(f"\n{video}")
        gt = compute_ground_truth(video, args)
        for n in intervals:
            label = "never" if n == 0 else f"every {n}"
            print(f"  tracker run: redetect {label}"
                  f"{' + on loss' if args.redetect_on_loss else ''}")
            try:
                r = run_tracker(video, gt, n, args)
            except RuntimeError as e:
                print(f"  skipped: {e}")
                continue
            all_runs[n].append(r)
            rows.append({"video": r.video, "interval": n,
                         "stats": summarise([r])})
            if not args.no_csv:
                write_frames_csv(r, out / f"{run_tag(r)}_frames.csv")
                write_features_npz(r, out / f"{run_tag(r)}_features.npz")
            if args.plot:
                plot_run(r, out / f"{run_tag(r)}.png")

    if not rows:
        sys.exit("No runs completed.")

    pooled = []
    if len(videos) > 1:
        for n in intervals:
            if all_runs[n]:
                pooled.append({"video": "ALL (pooled)", "interval": n,
                               "stats": summarise(all_runs[n])})
    print_table(rows + pooled)

    if args.plot and len(intervals) > 1:
        sweep = [{"interval": p["interval"], **p["stats"]} for p in (pooled or rows)
                 if p["video"].startswith("ALL") or len(videos) == 1]
        if len(sweep) > 1:
            plot_sweep(sweep, out / "sweep.png")

    summary = {
        "config": {k: v for k, v in vars(args).items()},
        "runs": rows,
        "pooled": pooled,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if o is None else str(o))
    print(f"\nResults written to {out.resolve()}")


if __name__ == "__main__":
    main()