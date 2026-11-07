"""Manual, withheld chessboard-intersection annotations for benchmark_pose.py.

Examples (PowerShell, run from the project directory):
    python annotate_pose_images.py
    python annotate_pose_images.py --image "pose_benchmark/images/occluded/4.jpg"
    python annotate_pose_images.py --grid-points "1:1,1:4,1:7,4:1,4:4,4:7,7:1,7:4,7:7"
    python annotate_pose_images.py --include-failed
    python benchmark_pose.py --mode images

Click true 7x7 *internal* grid-line intersections, NOT square centres. The
(row,col) IDs follow the CURRENT tracker's fixed convention:
    corner 0 = canonical grid (0,0) upper left
    corner 1 = canonical grid (8,0) upper right
    corner 2 = canonical grid (8,8) lower right
    corner 3 = canonical grid (0,8) lower left, board pose origin
    internal row grows from 0->3, col grows from 0->1.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLUMNS = ("image", "grid_row", "grid_col", "x_px", "y_px")
DEFAULT_GRID = "1:1,1:4,1:7,4:1,4:4,4:7,7:1,7:4,7:7"
PHOTO_W = 1140
PHOTO_H = 760
HEADER_H = 74
SIDEBAR_W = 300
WINDOW = "Chessboard manual annotations | raw image only"


def parse_points(value: str) -> list[tuple[int, int]]:
    points = []
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        try:
            row, col = [int(s.strip()) for s in chunk.split(":")]
        except (ValueError, TypeError) as exc:
            raise ValueError("Grid points must look like 1:1,1:4,4:4") from exc
        if row not in range(1, 8) or col not in range(1, 8):
            raise ValueError(f"Grid intersection {row}:{col} is not an internal point (1..7)")
        if (row, col) in points:
            raise ValueError(f"Duplicate point {row}:{col}")
        points.append((row, col))
    if not points:
        raise ValueError("At least one internal intersection must be requested")
    return points


def get_cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=BASE_DIR / "pose_benchmark")
    p.add_argument("--image", type=Path, help="Annotate only this original image")
    p.add_argument("--grid-points", default=DEFAULT_GRID,
                   help="Nine spread-out internal intersections by default")
    p.add_argument("--include-failed", action="store_true",
                   help="Also annotate images with no accepted pose")
    p.add_argument("--force-all", action="store_true",
                   help="Ignore image_metrics.csv even when available")
    p.add_argument("--resume", action="store_true",
                   help="Skip images already containing at least one manual point")
    p.add_argument("--max-width", type=int, default=PHOTO_W)
    p.add_argument("--max-height", type=int, default=PHOTO_H)
    args = p.parse_args(argv)
    if args.max_width < 300 or args.max_height < 250:
        p.error("--max-width and --max-height are too small")
    try:
        args.points = parse_points(args.grid_points)
    except ValueError as exc:
        p.error(str(exc))
    args.data_dir = args.data_dir.expanduser().resolve()
    return args


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def source_key(path: Path, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is not inside {data_dir}") from exc


def parse_labels(rows: list[dict[str, str]]) -> dict[str, dict[tuple[int, int], tuple[float, float]]]:
    labels: dict[str, dict[tuple[int, int], tuple[float, float]]] = {}
    for entry in rows:
        key = (entry.get("image") or "").strip().replace("\\", "/")
        if not key:
            continue
        r, c = int(entry["grid_row"]), int(entry["grid_col"])
        x, y = float(entry["x_px"]), float(entry["y_px"])
        if not (1 <= r <= 7 and 1 <= c <= 7 and math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Invalid annotation row: {entry}")
        entries = labels.setdefault(key, {})
        if (r, c) in entries:
            raise ValueError(f"Duplicate annotated point in existing file: {key} {r}:{c}")
        entries[(r, c)] = (x, y)
    return labels


def write_annotations(path: Path, all_labels: dict[str, dict[tuple[int, int], tuple[float, float]]]) -> None:
    """Atomic update: never damage the existing CSV on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for key in sorted(all_labels):
            for (row, col), (x, y) in sorted(all_labels[key].items()):
                writer.writerow({"image": key, "grid_row": row, "grid_col": col,
                                 "x_px": f"{x:.3f}", "y_px": f"{y:.3f}"})
    tmp.replace(path)


def list_images(args, existing: dict) -> list[Path]:
    if args.image:
        path = args.image.expanduser()
        if not path.is_absolute():
            candidate = (Path.cwd() / path).resolve()
            # Also accept paths relative to the script's directory.
            path = candidate if candidate.is_file() else (BASE_DIR / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    folder = args.data_dir / "images"
    if not folder.is_dir():
        raise FileNotFoundError(f"No images folder: {folder}")
    images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    metrics_file = args.data_dir / "results/image_metrics.csv"
    if not args.include_failed and not args.force_all and metrics_file.exists():
        accepted = {r["source"].replace("\\", "/") for r in csv_rows(metrics_file)
                    if (r.get("pose_accepted") or "0").strip().lower() in ("1", "true")}
        images = [p for p in images if source_key(p, args.data_dir) in accepted]
        print(f"Selecting {len(images)} images with accepted pose from {metrics_file.name}.")
    elif not args.include_failed and not args.force_all:
        print("NOTE: image_metrics.csv not found; all photos included."
              " Run benchmark_pose.py --mode images first for accepted-only selection.")
    if args.resume:
        images = [p for p in images if not existing.get(source_key(p, args.data_dir))]
    return images


@dataclass
class Editor:
    image: np.ndarray
    key: str
    requested: list[tuple[int, int]]
    labels: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)
    target_index: int = 0
    zoom: int = 1
    centre: Optional[tuple[float, float]] = None
    cursor: tuple[int, int] = (0, 0)
    history: list = field(default_factory=list)
    touched: bool = False
    photo_panel: tuple[int, int] = (1140, 760)

    def __post_init__(self):
        h, w = self.image.shape[:2]
        self.centre = self.centre or (w/2.0, h/2.0)
        self._seek_first_missing()

    def _seek_first_missing(self):
        self.target_index = next((i for i, p in enumerate(self.requested)
                                  if p not in self.labels), len(self.requested))

    @property
    def current_point(self):
        return self.requested[self.target_index] if self.target_index < len(self.requested) else None

    def region(self) -> tuple[float, float, float, float]:
        """Return uncropped original-image viewport (x0,y0,w,h)."""
        h, w = self.image.shape[:2]
        view_w, view_h = w / self.zoom, h / self.zoom
        cx, cy = self.centre
        x0 = float(np.clip(cx-view_w/2, 0, w-view_w))
        y0 = float(np.clip(cy-view_h/2, 0, h-view_h))
        return x0, y0, view_w, view_h

    def displayed_to_native(self, x: int, y: int) -> Optional[tuple[float, float]]:
        pw, ph = self.photo_panel
        if not (0 <= x < pw and HEADER_H <= y < HEADER_H+ph):
            return None
        x0, y0, vw, vh = self.region()
        # Mouse event coordinates are pixel positions, not cell centres.
        xx = x0 + (x+0.5)*vw/pw - 0.5
        yy = y0 + (y-HEADER_H+0.5)*vh/ph - 0.5
        return (float(np.clip(xx, 0, self.image.shape[1]-1)),
                float(np.clip(yy, 0, self.image.shape[0]-1)))

    def native_to_display(self, x: float, y: float) -> tuple[int, int]:
        x0, y0, vw, vh = self.region()
        pw, ph = self.photo_panel
        return (int(round((x-x0+0.5)*pw/vw-0.5)),
                int(round((y-y0+0.5)*ph/vh-0.5))+HEADER_H)

    def _snapshot(self):
        self.history.append((self.labels.copy(), self.target_index))
        self.touched = True

    def click(self, x: float, y: float):
        if self.current_point is None:
            return
        self._snapshot()
        self.labels[self.current_point] = (x, y)
        self.target_index += 1

    def skip(self):
        if self.current_point is None:
            return
        self._snapshot()
        self.target_index += 1

    def undo(self):
        if self.history:
            self.labels, self.target_index = self.history.pop()
            self.touched = True

    def reset_progress(self):
        # Revisits all points while retaining previously recorded observations.
        self.target_index = 0

    def zoom_by(self, factor: int):
        new_zoom = int(np.clip(self.zoom*factor, 1, 8)) if factor > 1 else max(1, self.zoom//2)
        if new_zoom == self.zoom:
            return
        # Zoom around the last cursor position if it's on the raw-image panel.
        original = self.displayed_to_native(*self.cursor)
        self.zoom = new_zoom
        if original is not None:
            self.centre = original

    def pan(self, dx_fraction: float, dy_fraction: float):
        x0,y0,vw,vh = self.region()
        cx,cy = self.centre
        self.centre = (cx + dx_fraction*vw, cy + dy_fraction*vh)

    def draw(self) -> np.ndarray:
        h, w = self.image.shape[:2]
        pw, ph = self.photo_panel
        photo_area = np.empty((ph, pw, 3), dtype=np.uint8)
        x0,y0,vw,vh = self.region()
        x1,y1 = int(round(x0)),int(round(y0))
        x2,y2 = min(w, int(round(x0+vw))), min(h, int(round(y0+vh)))
        x1,y1 = max(0, x1),max(0,y1)
        crop = self.image[y1:max(y1+1,y2), x1:max(x1+1,x2)]
        photo_area[:] = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_LINEAR)
        out = np.full((ph+HEADER_H, pw+SIDEBAR_W, 3), (35, 38, 42), dtype=np.uint8)
        out[HEADER_H:, :pw] = photo_area
        cv2.putText(out, self.key[:83], (14,27), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255,255,255), 2, cv2.LINE_AA)
        current = self.current_point
        message = (f"CLICK intersection row {current[0]}, col {current[1]}"
                   if current is not None else "ALL REQUESTED POINTS DONE. Press N to save + next.")
        cv2.putText(out, message, (14,60), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (100,235,255) if current is not None else (100,240,125), 2, cv2.LINE_AA)
        for (r,c), (x,y) in self.labels.items():
            xx,yy = self.native_to_display(x,y)
            if 0 <= xx < pw and HEADER_H <= yy < ph+HEADER_H:
                cv2.drawMarker(out, (xx,yy), (0,250,255),
                               cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                cv2.putText(out, f"{r}:{c}", (xx+7,yy-8),
                            cv2.FONT_HERSHEY_SIMPLEX,.48,(0,250,255),2,cv2.LINE_AA)
        side_x = pw+18
        cv2.putText(out, "GRID ID GUIDE", (side_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .61, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, "Physical grid intersections", (side_x, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, (204,207,214), 1, cv2.LINE_AA)
        length = min(230, SIDEBAR_W-55)
        ox,oy = side_x+13,HEADER_H+50
        sq = length//8
        for r in range(8):
            for c in range(8):
                light = ((r+c)%2 == 0)
                color = (210,210,210) if light else (55,55,55)
                cv2.rectangle(out,(ox+c*sq,oy+r*sq),(ox+(c+1)*sq,oy+(r+1)*sq),color,-1)
        cv2.rectangle(out,(ox,oy),(ox+8*sq,oy+8*sq),(220,200,80),2)
        for corner,px,py in (("0",ox,oy),("1",ox+8*sq,oy),
                             ("2",ox+8*sq,oy+8*sq),("3",ox,oy+8*sq)):
            cx = px-12 if px>ox else px+4
            cy = py-8 if py>oy else py-8
            cv2.putText(out,corner,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,
                        .65,(75,180,255),2,cv2.LINE_AA)
        if current is not None:
            r,c = current
            cv2.circle(out, (ox+c*sq,oy+r*sq), 8, (60,60,255), -1, cv2.LINE_AA)
            cv2.circle(out, (ox+c*sq,oy+r*sq), 11, (240,240,240), 1, cv2.LINE_AA)
        for (r,c) in self.labels:
            cv2.circle(out, (ox+c*sq,oy+r*sq), 4, (0,250,255), -1, cv2.LINE_AA)
        legend_y = oy+8*sq+47
        txt = [f"Collected: {len(self.labels)} points",
               f"Progress: {min(self.target_index,len(self.requested))}/{len(self.requested)}",
               f"Zoom: {self.zoom}x | raw {w} x {h}",
               "", "Left click: annotate", "SPACE: skip hidden", "U / Backspace: undo",
               "T: revisit points", "Z / +/-: zoom", "I J K L: pan image",
               "V: optional corner-ID view", "N: save and next",
               "B: save and previous", "Q / ESC: save and quit"]
        for i, line in enumerate(txt):
            cv2.putText(out,line,(side_x,legend_y+i*25),cv2.FONT_HERSHEY_SIMPLEX,
                        .46,(222,223,225),1,cv2.LINE_AA)
        return out


def open_image(path: Path) -> np.ndarray:
    # np.fromfile + imdecode works with Unicode filenames on Windows.
    buf = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    return frame


def run_gui(editor: Editor, reference_path: Optional[Path] = None) -> str:
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    ref_window = "Corner IDs from benchmark diagnostic | reference ONLY"
    reference_open = False
    cv2.imshow(WINDOW, editor.draw())

    def cb(event, x, y, flags, param):
        editor.cursor = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = editor.displayed_to_native(x,y)
            if xy is not None:
                editor.click(*xy)
                cv2.imshow(WINDOW, editor.draw())
        elif event == cv2.EVENT_MOUSEWHEEL:
            # OpenCV-Python does not expose getMouseWheelDelta on every build.
            delta = (flags >> 16) & 0xffff
            if delta >= 32768:
                delta -= 65536
            editor.zoom_by(2 if delta > 0 else 0)
            cv2.imshow(WINDOW, editor.draw())
    cv2.setMouseCallback(WINDOW, cb)
    try:
        while True:
            key = cv2.waitKeyEx(50)
            if key < 0:
                continue
            k = key & 0xff
            if k in (ord('q'),ord('Q'),27):
                return "quit"
            if k in (ord('n'),ord('N'),13):
                return "next"
            if k in (ord('b'),ord('B')):
                return "previous"
            if k in (ord('v'),ord('V')):
                if reference_open:
                    try:
                        cv2.destroyWindow(ref_window)
                    except cv2.error:
                        pass
                    reference_open = False
                elif reference_path is not None and reference_path.is_file():
                    reference = open_image(reference_path)
                    sc = min(900/reference.shape[1],620/reference.shape[0],1)
                    reference = cv2.resize(reference,None,fx=sc,fy=sc,
                                           interpolation=cv2.INTER_AREA) if sc<1 else reference
                    cv2.imshow(ref_window,reference)
                    reference_open = True
                else:
                    print("No benchmark orientation figure found. Run benchmark_pose.py --mode images first.")
                continue
            if k == 32:
                editor.skip()
            elif k in (ord('u'),ord('U'),8,127):
                editor.undo()
            elif k in (ord('t'),ord('T')):
                editor.reset_progress()
            elif k in (ord('z'),ord('Z'),ord('+'),ord('=')):
                editor.zoom_by(2)
            elif k in (ord('-'),ord('_'),ord('0')):
                editor.zoom_by(0)
            elif k in (ord('i'),ord('I')):
                editor.pan(0,-.24)
            elif k in (ord('k'),ord('K')):
                editor.pan(0,.24)
            elif k in (ord('j'),ord('J')):
                editor.pan(-.24,0)
            elif k in (ord('l'),ord('L')):
                editor.pan(.24,0)
            else:
                continue
            cv2.imshow(WINDOW, editor.draw())
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    args = get_cli(argv)
    labels_path = args.data_dir / "annotations/image_points.csv"
    all_labels = parse_labels(csv_rows(labels_path))
    images = list_images(args, all_labels)
    if not images:
        print("No matching images. Verify pose_benchmark/images/ and results/image_metrics.csv")
        return 0
    print(f"Annotating {len(images)} images. Original pixels only. Grid points: {args.points}")
    print("IMPORTANT: match the corner IDs in your diagnostic image."
          " The checkerboard has a 180-degree orientation ambiguity.")
    current = 0
    try:
        while 0 <= current < len(images):
            path = images[current]
            key = source_key(path,args.data_dir)
            img = open_image(path)
            scale = min(args.max_width/img.shape[1],args.max_height/img.shape[0],1.0)
            pw,ph = max(1,round(img.shape[1]*scale)),max(1,round(img.shape[0]*scale))
            saved = all_labels.get(key, {}).copy()
            editor = Editor(img,key,args.points,saved,photo_panel=(pw,ph))
            print(f"[{current+1}/{len(images)}] {key}: existing {len(saved)} manual points")
            rel = path.relative_to(args.data_dir / "images")
            reference_path = (args.data_dir / "figures/images" / rel).with_suffix(".jpg")
            command = run_gui(editor, reference_path)
            if editor.touched:
                # Existing points not in this sample set are retained.
                all_labels[key] = editor.labels
                write_annotations(labels_path,all_labels)
                print(f"Saved {len(editor.labels)} independently clicked points: {key}")
            if command == "quit":
                break
            current += -1 if command == "previous" else 1
    except KeyboardInterrupt:
        print("Interrupted; previously saved images are preserved.")
    print(f"Annotations location: {labels_path}")
    print("Recalculate withheld-point metrics with: python benchmark_pose.py --mode images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
