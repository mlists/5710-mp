import os
import base64
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

import render_board as pc

# =============== Constants ===============
SQ = 70
MAX_W, MAX_H = 900, 700
FONT = ("Segoe UI Symbol", 34)
GLYPHS = {"King": "♚",
          "Queen": "♛",
          "Rook": "♜",
          "Bishop": "♝",
          "Knight": "♞",
          "Pawn": "♟"
}

PIECES = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# =============== Functions ===============
def start_position():
    """Initialise the board"""
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
    """Convert to format for renderer"""
    pieces = {}
    for (r, c), name in state.items():
        pieces.setdefault(name, []).append((r, c))
    return {k: tuple(v) for k, v in pieces.items()}

def starting_castling_rights():
    """Rights belong to the original, unmoved king and corner rooks"""
    return {"White": {"K": True, "Q": True},
            "Black": {"K": True, "Q": True}}


def empty_castling_rights():
    return {"White": {"K": False, "Q": False},
            "Black": {"K": False, "Q": False}}


def other_side(side):
    return "Black" if side == "White" else "White"


def square_name(square):
    """Board coordinate to algebraic notation"""
    row, col = square
    return f"{'abcdefgh'[col]}{8 - row}"


def path_is_clear(state, start, end):
    """Checks the path is clear for certain pieces"""
    sr, sc = start
    er, ec = end
    step_r = (er > sr) - (er < sr)
    step_c = (ec > sc) - (ec < sc)
    r, c = sr + step_r, sc + step_c
    while (r, c) != end:
        if (r, c) in state:
            return False
        r += step_r
        c += step_c
    return True


def is_square_attacked(state, square, attacker_side):
    """Chess attack map; pawns attack diagonally even onto empty squares"""
    tr, tc = square
    for (sr, sc), piece in state.items():
        side, kind = piece.split("_", 1)
        if side != attacker_side:
            continue

        dr, dc = tr - sr, tc - sc
        adr, adc = abs(dr), abs(dc)
        if kind == "Pawn":
            if dr == (-1 if side == "White" else 1) and adc == 1:
                return True
        elif kind == "Knight":
            if (adr, adc) in ((1, 2), (2, 1)):
                return True
        elif kind == "King":
            if max(adr, adc) == 1:
                return True
        elif kind in ("Bishop", "Rook", "Queen"):
            diagonal = adr == adc and adr != 0
            straight = (dr == 0) != (dc == 0)
            can_attack = ((kind == "Bishop" and diagonal)
                          or (kind == "Rook" and straight)
                          or (kind == "Queen" and (diagonal or straight)))
            if can_attack and path_is_clear(state, (sr, sc), square):
                return True
    return False


def validate_move(state, start, end, castling_rights, en_passant_target=None):
    """Validate a move and simulate it, without mutating the input state"""
    if any(not (0 <= value < 8) for value in (*start, *end)):
        return None, "The square is outside the chessboard."
    if start == end:
        return None, "Choose another destination square."

    piece = state.get(start)
    if piece is None:
        return None, "There is no piece on the selected square."
    side, kind = piece.split("_", 1)
    enemy = other_side(side)
    destination_piece = state.get(end)
    if destination_piece is not None:
        if same_side(piece, destination_piece):
            return None, "You cannot capture your own piece."
        if destination_piece.endswith("_King"):
            return None, "You cannot capture a king."

    sr, sc = start
    er, ec = end
    dr, dc = er - sr, ec - sc
    adr, adc = abs(dr), abs(dc)
    en_passant_capture = None
    rook_from = rook_to = None

    if kind == "Knight":
        if (adr, adc) not in ((1, 2), (2, 1)):
            return None, "Knights must move in an L shape (2 squares and 1)."

    elif kind in ("Bishop", "Rook", "Queen"):
        diagonal = adr == adc and adr != 0
        straight = (dr == 0) != (dc == 0)
        allowed = ((kind == "Bishop" and diagonal)
                   or (kind == "Rook" and straight)
                   or (kind == "Queen" and (diagonal or straight)))
        if not allowed:
            return None, f"That is not a valid {kind.lower()} movement."
        if not path_is_clear(state, start, end):
            return None, f"A piece is blocking the {kind.lower()}'s path."

    elif kind == "King":
        if max(adr, adc) == 1:
            pass  
        elif dr == 0 and adc == 2:
            home_row = 7 if side == "White" else 0
            flank = "K" if dc == 2 else "Q"
            if start != (home_row, 4):
                return None, "Castling requires the king on its starting square."
            if not castling_rights.get(side, {}).get(flank, False):
                return None, "Castling is unavailable because the king or rook moved."
            rook_from = (home_row, 7 if flank == "K" else 0)
            rook_to = (home_row, 5 if flank == "K" else 3)
            if state.get(rook_from) != f"{side}_Rook":
                return None, "The castling rook is missing."
            if not path_is_clear(state, start, rook_from):
                return None, "Pieces are blocking castling."
            direction = 1 if flank == "K" else -1
            for col in (4, 4 + direction, 4 + 2 * direction):
                if is_square_attacked(state, (home_row, col), enemy):
                    return None, "You cannot castle out of, through or into check."
        else:
            return None, "Kings may move one square, or castle when permitted."

    elif kind == "Pawn":
        direction = -1 if side == "White" else 1
        home_row = 6 if side == "White" else 1
        if dc == 0:
            if destination_piece is not None:
                return None, "Pawns cannot capture straight ahead."
            if dr == direction:
                pass
            elif (dr == 2 * direction and sr == home_row
                  and (sr + direction, sc) not in state):
                pass
            else:
                return None, "Pawns move one square forward, or two from their starting rank."
        elif adc == 1 and dr == direction:
            if destination_piece is None:
                adjacent = (sr, ec)
                if (end != en_passant_target
                        or state.get(adjacent) != f"{enemy}_Pawn"):
                    return None, "Pawns capture diagonally only when an enemy is present."
                en_passant_capture = adjacent
        else:
            return None, "That is not a valid pawn movement."

    else:
        return None, f"Unknown chess piece: {kind}."

    next_state = state.copy()
    del next_state[start]
    if en_passant_capture is not None:
        del next_state[en_passant_capture]
    promoted = (kind == "Pawn" and er == (0 if side == "White" else 7))
    next_state[end] = f"{side}_Queen" if promoted else piece
    if rook_from is not None:
        next_state[rook_to] = next_state.pop(rook_from)

    king_pos = next((pos for pos, name in next_state.items()
                     if name == f"{side}_King"), None)
    if king_pos is not None and is_square_attacked(next_state, king_pos, enemy):
        return None, "This move would leave your king in check."

    return next_state, None


def revoke_castling_rights_for_square(rights, square):
    """A manual edit on a king/rook home square invalidates its old rights"""
    for side, row in (("White", 7), ("Black", 0)):
        if square == (row, 4):
            rights[side]["K"] = rights[side]["Q"] = False
        elif square == (row, 0):
            rights[side]["Q"] = False
        elif square == (row, 7):
            rights[side]["K"] = False


def update_castling_rights(rights, state, start, end):
    """Permanently remove castling rights after king/rook moves or captures"""
    moving_piece = state[start]
    side, kind = moving_piece.split("_", 1)
    if kind == "King":
        rights[side]["K"] = rights[side]["Q"] = False
    elif kind == "Rook":
        revoke_castling_rights_for_square(rights, start)

    captured_piece = state.get(end)
    if captured_piece is not None and captured_piece.endswith("_Rook"):
        revoke_castling_rights_for_square(rights, end)


class BoardPicker:
    """Converts image pixels to (row, col) squares and back, using the solved pose"""

    def __init__(self, pose):
        self.K = np.asarray(pose["camera_matrix"], dtype=float)
        self.R = np.asarray(pose["R"], dtype=float)
        self.t = np.asarray(pose["tvec"], dtype=float).reshape(3)
        self.centres = np.asarray(pose["square_centres_board"], dtype=float)
        self.spacing = float(np.linalg.norm(self.centres[0, 0] - self.centres[0, 1]))
        self.rvec, _ = cv2.Rodrigues(self.R)

    def pick(self, u, v):
        """Converting from pinch co-ordinates to a board square"""
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
        self.castling_rights = starting_castling_rights()
        self.en_passant_target = None
        self.selected = None        
        self.invalid_square = None
        self.frame = None
        self.pose = None
        self.picker = None
        self.scale = 1.0
        self.photo = None
        self.out_bgr = None

        self.tool = tk.StringVar(value="None")  
        self._last_tool = "None"
        self.build_ui()

        path = image_path or self.ask_image()
        if path:
            self.load_image(path)
        self.refresh()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side="top", fill="x", padx=6, pady=4)
        self.toolbar = top            

        tk.Button(top, text="Open file", command=self.open_image).pack(side="left")
        tk.Button(top, text="Reset", command=self.reset).pack(side="left", padx=4)
        tk.Button(top, text="Clear", command=self.clear).pack(side="left")
        tk.Button(top, text="Save image", command=self.save_image).pack(side="left", padx=4)

        palette = tk.Frame(self.root)
        palette.pack(side="top", fill="x", padx=6)
        for i, side in enumerate(("White", "Black")):
            for j, kind in enumerate(PIECES):
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

        self.status = tk.Label(self.root, anchor="w", text="No tool selected. Click a piece button to place it (click it again to switch off)")
        self.status.pack(side="bottom", fill="x", padx=6)

    def ask_image(self):
        """Ask for an image file. Returns '' if the dialog is cancelled"""
        return filedialog.askopenfilename(
            title="Pick a board image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])

    def open_image(self):
        path = self.ask_image()
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
            self.pose = pc.estimate_pose(rgb)       # done once per image, not per edit
            self.picker = BoardPicker(self.pose)
            self.status.config(text=f"Loaded {os.path.basename(path)}")
        except Exception as error:
            messagebox.showerror("Board not found",
                                 f"Pose estimation failed:\n{type(error).__name__}: {error}")

    def pick_tool(self, value):
        """Clicking the active tool again switches it off"""
        if value == self._last_tool:
            self.tool.set("None")
            self._last_tool = "None"
        else:
            self._last_tool = value
        self.selected = None
        self.invalid_square = None
        if self.tool.get() == "Move":
            self.status.config(text="Move: choose a piece, then a legal destination. Either colour may move.", fg="#333")
        else:
            self.status.config(text=f"Tool: {self.tool.get()}", fg="#333")
        self.refresh()

    def get_occluder(self):
        """Override to return an (h, w) float mask of things in front of the board"""
        return None

    def attempt_move(self, source, square):
        """Validate and, if legal, commit a move. Shared by mouse and pinch input"""
        next_state, reason = validate_move(
            self.state, source, square,
            self.castling_rights, self.en_passant_target,
        )
        if next_state is None:
            self.invalid_square = square
            return None, reason

        side, kind = self.state[source].split("_", 1)
        is_castle = kind == "King" and abs(square[1] - source[1]) == 2
        is_promotion = kind == "Pawn" and square[0] in (0, 7)

        # Rights must be updated from the position BEFORE the move
        update_castling_rights(self.castling_rights, self.state, source, square)
        self.en_passant_target = (
            ((source[0] + square[0]) // 2, source[1])
            if kind == "Pawn" and abs(square[0] - source[0]) == 2
            else None
        )
        self.state = next_state
        self.invalid_square = None
        suffix = (" (castling)" if is_castle else
                  " (promoted to Queen)" if is_promotion else "")
        return f"{side} {kind} {square_name(source)} to {square_name(square)}{suffix}.", None

    def apply(self, square, button):
        if square is None:
            return
        tool = self.tool.get()
        if tool == "None":
            return

        self.invalid_square = None
        if button == "right" or tool == "Erase":
            removed = self.state.pop(square, None)
            if removed is not None:
                revoke_castling_rights_for_square(self.castling_rights, square)
                self.en_passant_target = None
            self.selected = None
            self.status.config(text=f"Erased {square_name(square)}." if removed else
                               f"{square_name(square)} is already empty.", fg="#333")
        elif tool == "Move":
            if self.selected is None:
                if square in self.state:
                    self.selected = square
                    self.status.config(text=f"Selected {self.state[square].replace('_', ' ')} "
                                            f"on {square_name(square)}. Choose a destination.", fg="#333")
                else:
                    self.status.config(text="Select an occupied square to move a piece.", fg="#333")
            else:
                source = self.selected
                target = self.state.get(square)
                if square == source:
                    self.selected = None  # clicking the selected piece deselects it
                    self.status.config(text="Move cancelled.", fg="#333")
                elif target is not None and same_side(target, self.state[source]):
                    self.selected = square  # clicking a friendly piece selects it instead
                    self.status.config(text=f"Selected {target.replace('_', ' ')} "
                                            f"on {square_name(square)}. Choose a destination.", fg="#333")
                else:
                    description, reason = self.attempt_move(source, square)
                    if description is None:
                        # Keep source selected so the player can retry.
                        self.status.config(text=f"Illegal move: {reason}", fg="#bb2222")
                    else:
                        self.selected = None
                        self.status.config(text=f"Moved {description}", fg="#116633")
        else:
            self.state[square] = tool
            revoke_castling_rights_for_square(self.castling_rights, square)
            self.en_passant_target = None
            self.selected = None
            self.status.config(text=f"Placed {tool.replace('_', ' ')} at {square_name(square)}.",
                               fg="#333")

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
        self.castling_rights = starting_castling_rights()
        self.en_passant_target = None
        self.selected = None
        self.invalid_square = None
        self.status.config(text="Starting position restored. Either colour may move.", fg="#333")
        self.refresh()

    def clear(self):
        self.state = {}
        self.castling_rights = empty_castling_rights()
        self.en_passant_target = None
        self.selected = None
        self.invalid_square = None
        self.status.config(text="Board cleared. Use the piece tools to create a position.", fg="#333")
        self.refresh()

    def save_image(self):
        if self.out_bgr is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.out_bgr)

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
        if self.invalid_square is not None:
            r, c = self.invalid_square
            cv.create_rectangle(c * SQ + 4, r * SQ + 4, (c + 1) * SQ - 4, (r + 1) * SQ - 4,
                                outline="#dd3333", width=3)

    def draw_image(self):
        cv = self.image_cv
        cv.delete("all")
        if self.frame is None:
            return

        self.last_timings = {}
        if self.pose is not None:
            out = pc.render_overlay(self.frame, self.pose, state_to_pieces(self.state),
                                    self.get_occluder(), timings=self.last_timings)
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
        if self.invalid_square is not None and self.picker is not None:
            u, v = self.picker.project(*self.invalid_square) * self.scale
            cv.create_oval(u - 14, v - 14, u + 14, v + 14, outline="#dd3333", width=3)


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
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    try:
        root.mainloop()
    finally:
        pc.close_renderer()