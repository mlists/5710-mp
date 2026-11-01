import os
import base64
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import trimesh
import pyrender
import math

# >>> Change this to whatever your projection file is called (without .py) <<<
import real_board_scene as pc

SQ = 70                      # size of a square on the simulated board (px)
MAX_W, MAX_H = 900, 700      # max size of the image view (px)
FONT = ("Segoe UI Symbol", 34)
GLYPHS = {"King": "♚", "Queen": "♛", "Rook": "♜", "Bishop": "♝", "Knight": "♞", "Pawn": "♟"}
KINDS = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# =============== Board model ===============
def start_position():
    """{(row, col): piece name}. Row 0 is Black's back rank, same as the renderer"""
    back = ["Rook", "Knight", "Bishop", "Queen", "King", "Bishop", "Knight", "Rook"]
    state = {}
    for c in range(8):
        state[(0, c)] = f"Black_{back[c]}"
        state[(1, c)] = "Black_Pawn"
        state[(6, c)] = "White_Pawn"
        state[(7, c)] = f"White_{back[c]}"
    return state


def same_side(a, b):
    """True if two piece names ("White_Pawn", "Black_Rook") are the same colour"""
    return a.split("_")[0] == b.split("_")[0]


def state_to_pieces(state):
    """{(row, col): name} -> {name: ((row, col), ...)} for the renderer"""
    pieces = {}
    for (r, c), name in state.items():
        pieces.setdefault(name, []).append((r, c))
    return {k: tuple(v) for k, v in pieces.items()}


# =============== Rendering with a cached pose ===============
def render_overlay(frame, pose, pieces, occluder=None):
    """Same scene/render/composite as render_frame, but reuses an existing pose.

    occluder: optional float32 (h, w) array in [0, 1] (1 = something real, e.g. a hand,
    is in front of the board there); pieces are not drawn where it is set.
    """
    h, w = frame.shape[:2]
    K = np.asarray(pose["camera_matrix"], dtype=float)
    R = np.asarray(pose["R"], dtype=float)
    tvec = np.asarray(pose["tvec"], dtype=float).reshape(3)

    cam_pose = pc.opencv_to_gl_camera_pose(R, tvec)
    distance = float(np.linalg.norm(tvec))

    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        znear=distance * 0.01, zfar=distance * 10,
    )

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[pc.AMBIENT_LIGHT] * 3)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=pc.LIGHT_INTENSITY)
    tilt = trimesh.transformations.rotation_matrix(math.radians(pc.LIGHT_TILT_DEG), [1, 0, 0])
    scene.add(light, pose=cam_pose @ tilt)

    for piece, squares in pieces.items():
        for row, col in squares:
            mesh, placement = pc.place_piece(piece, row, col, pose)
            scene.add(mesh, pose=placement)

    color, _ = pc.get_renderer(w, h).render(scene, flags=pyrender.RenderFlags.RGBA)

    out = frame.copy()
    alpha_full = color[..., 3].astype(np.float32) / 255
    if occluder is not None:
        alpha_full *= 1.0 - occluder
    mask = alpha_full > 0
    if mask.any():
        alpha = alpha_full[mask][:, None]
        piece_bgr = color[mask][:, [2, 1, 0]].astype(np.float32)
        out[mask] = (piece_bgr * alpha + out[mask] * (1.0 - alpha)).astype(np.uint8)
    return out


# =============== Image <-> board square mapping ===============
class BoardPicker:
    """Converts image pixels to (row, col) squares and back, using the solved pose"""

    def __init__(self, pose):
        self.K = np.asarray(pose["camera_matrix"], dtype=float)
        self.R = np.asarray(pose["R"], dtype=float)
        self.t = np.asarray(pose["tvec"], dtype=float).reshape(3)
        self.centres = np.asarray(pose["square_centres_board"], dtype=float)[..., :3]
        self.spacing = float(np.linalg.norm(self.centres[0, 0] - self.centres[0, 1]))
        self.rvec, _ = cv2.Rodrigues(self.R)

    def pick(self, u, v):
        """Pixel -> (row, col) by intersecting the viewing ray with the board plane"""
        ray_cam = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        origin = -self.R.T @ self.t
        direction = self.R.T @ ray_cam
        z0 = self.centres[..., 2].mean()
        if abs(direction[2]) < 1e-12:
            return None
        s = (z0 - origin[2]) / direction[2]
        if s <= 0:
            return None
        p = origin + s * direction
        dist = np.linalg.norm(self.centres[..., :2] - p[:2], axis=2)
        r, c = np.unravel_index(dist.argmin(), dist.shape)
        if dist[r, c] > 0.75 * self.spacing:
            return None
        return int(r), int(c)

    def project(self, row, col):
        """(row, col) -> pixel of the square centre"""
        pt, _ = cv2.projectPoints(self.centres[row, col].reshape(1, 3), self.rvec, self.t, self.K, None)
        return pt.reshape(2)


# =============== GUI ===============
class App:
    def __init__(self, root, image_path):
        self.root = root
        root.title("Chess projection editor")

        self.state = start_position()
        self.selected = None          # square picked up by the Move tool
        self.frame = None
        self.pose = None
        self.picker = None
        self.scale = 1.0
        self.photo = None
        self.out_bgr = None

        self.tool = tk.StringVar(value="None")   # nothing selected: clicks do nothing
        self.build_ui()

        if image_path:
            self.load_image(image_path)
        else:
            self.open_image()
        self.refresh()

    # ---------- UI ----------
    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side="top", fill="x", padx=6, pady=4)

        tk.Button(top, text="Open image", command=self.open_image).pack(side="left")
        tk.Button(top, text="Reset", command=self.reset).pack(side="left", padx=4)
        tk.Button(top, text="Clear", command=self.clear).pack(side="left")
        tk.Button(top, text="Save image", command=self.save_image).pack(side="left", padx=4)

        palette = tk.Frame(self.root)
        palette.pack(side="top", fill="x", padx=6)
        for i, side in enumerate(("White", "Black")):
            for j, kind in enumerate(KINDS):
                tk.Radiobutton(
                    palette, text=f"{GLYPHS[kind]} {side} {kind}", variable=self.tool,
                    value=f"{side}_{kind}", indicatoron=False, width=14,
                    command=lambda v=f"{side}_{kind}": self.pick_tool(v),
                ).grid(row=i, column=j, padx=1, pady=1)
        tk.Radiobutton(palette, text="Erase", variable=self.tool, value="Erase",
                       indicatoron=False, width=10,
                       command=lambda: self.pick_tool("Erase")).grid(row=0, column=6, padx=6)
        tk.Radiobutton(palette, text="Move", variable=self.tool, value="Move",
                       indicatoron=False, width=10,
                       command=lambda: self.pick_tool("Move")).grid(row=1, column=6, padx=6)

        main = tk.Frame(self.root)
        main.pack(side="top", padx=6, pady=6)

        self.board_cv = tk.Canvas(main, width=8 * SQ, height=8 * SQ, highlightthickness=0)
        self.board_cv.grid(row=0, column=0, padx=(0, 8))
        self.board_cv.bind("<Button-1>", lambda e: self.on_board_click(e, "left"))
        self.board_cv.bind("<Button-3>", lambda e: self.on_board_click(e, "right"))

        self.image_cv = tk.Canvas(main, width=MAX_W, height=MAX_H, highlightthickness=0, bg="#222")
        self.image_cv.grid(row=0, column=1)
        self.image_cv.bind("<Button-1>", lambda e: self.on_image_click(e, "left"))
        self.image_cv.bind("<Button-3>", lambda e: self.on_image_click(e, "right"))

        self.status = tk.Label(self.root, anchor="w",
                               text="No tool selected. Click a piece button to place it (click it again to switch off)")
        self.status.pack(side="bottom", fill="x", padx=6)

    # ---------- Loading ----------
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Pick a board image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.load_image(path)
            self.refresh()

    def load_image(self, path):
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return
        self.frame = frame
        self.pose = None
        self.picker = None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            self.pose, _ = pc.estimate_pose(rgb)       # done once per image, not per edit
            self.picker = BoardPicker(self.pose)
            self.status.config(text=f"Loaded {os.path.basename(path)}")
        except Exception as error:
            messagebox.showerror("Board not found",
                                 f"Pose estimation failed:\n{type(error).__name__}: {error}")

    # ---------- Editing ----------
    def pick_tool(self, value):
        """Clicking the active tool again switches it off (clicks then do nothing)"""
        if value == getattr(self, "_last_tool", "None"):
            self.tool.set("None")
            self._last_tool = "None"
        else:
            self._last_tool = value
        self.selected = None
        self.refresh()

    def get_occluder(self):
        """Override to return an (h, w) float mask of things in front of the board"""
        return None

    def apply(self, square, button):
        if square is None:
            return
        tool = self.tool.get()
        if tool == "None":
            return

        if button == "right" or tool == "Erase":
            self.state.pop(square, None)
            self.selected = None
        elif tool == "Move":
            if self.selected is None:
                if square in self.state:
                    self.selected = square
            else:
                target = self.state.get(square)
                if target is not None and same_side(target, self.state[self.selected]):
                    self.selected = square        # own piece: switch selection, never capture it
                else:
                    if square != self.selected:
                        self.state[square] = self.state.pop(self.selected)
                    self.selected = None
        else:
            self.state[square] = tool
            self.selected = None

        self.refresh()

    def on_board_click(self, event, button):
        r, c = event.y // SQ, event.x // SQ
        if 0 <= r < 8 and 0 <= c < 8:
            self.apply((r, c), button)

    def on_image_click(self, event, button):
        if self.picker is None:
            return
        self.apply(self.picker.pick(event.x / self.scale, event.y / self.scale), button)

    def reset(self):
        self.state = start_position()
        self.selected = None
        self.refresh()

    def clear(self):
        self.state = {}
        self.selected = None
        self.refresh()

    def save_image(self):
        if self.out_bgr is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.out_bgr)

    # ---------- Drawing ----------
    def refresh(self):
        self.draw_board()
        self.draw_image()

    def draw_board(self):
        cv = self.board_cv
        cv.delete("all")
        for r in range(8):
            for c in range(8):
                fill = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
                cv.create_rectangle(c * SQ, r * SQ, (c + 1) * SQ, (r + 1) * SQ,
                                    fill=fill, outline=fill)
        for i in range(8):
            cv.create_text(i * SQ + SQ - 8, 8 * SQ - 8, text="abcdefgh"[i], fill="#555")
            cv.create_text(8, i * SQ + 10, text=str(8 - i), fill="#555")

        for (r, c), name in self.state.items():
            side, kind = name.split("_")
            x, y = c * SQ + SQ // 2, r * SQ + SQ // 2
            fg, bg = ("white", "black") if side == "White" else ("black", "white")
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cv.create_text(x + dx, y + dy, text=GLYPHS[kind], font=FONT, fill=bg)
            cv.create_text(x, y, text=GLYPHS[kind], font=FONT, fill=fg)

        if self.selected is not None:
            r, c = self.selected
            cv.create_rectangle(c * SQ + 2, r * SQ + 2, (c + 1) * SQ - 2, (r + 1) * SQ - 2,
                                outline="#2f9bff", width=4)

    def draw_image(self):
        cv = self.image_cv
        cv.delete("all")
        if self.frame is None:
            return

        if self.pose is not None:
            out = render_overlay(self.frame, self.pose, state_to_pieces(self.state),
                                 self.get_occluder())
        else:
            out = self.frame
        self.out_bgr = out

        h, w = out.shape[:2]
        self.scale = min(MAX_W / w, MAX_H / h, 1.0)
        shown = cv2.resize(out, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", shown, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self.photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))

        cv.config(width=shown.shape[1], height=shown.shape[0])
        cv.create_image(0, 0, anchor="nw", image=self.photo)

        if self.selected is not None and self.picker is not None:
            u, v = self.picker.project(*self.selected) * self.scale
            cv.create_oval(u - 14, v - 14, u + 14, v + 14, outline="#2f9bff", width=3)


def first_image(path):
    if path and os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))
        return os.path.join(path, files[0]) if files else None
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=os.path.join(pc.BASE, "Captures_1"),
                        help="board image, or a folder (first image is used)")
    args = parser.parse_args()

    root = tk.Tk()
    app = App(root, first_image(args.image))

    def on_close():
        pc.close_renderer()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    try:
        root.mainloop()
    finally:
        pc.close_renderer()