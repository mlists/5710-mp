import os
import math
import time
import pickle

import cv2
import numpy as np
import trimesh
import pyrender

import find_board
from board_tracker_v2 import BoardTracker
from get_pose import get_board_pose
from pose_quality import evaluate_corner_gate

# =============== Paths ===============
BASE = os.path.dirname(os.path.abspath(__file__))
BAKED_DIR = os.path.join(BASE, "Chess_models", "_baked")
OUT_PATH = os.path.join(BASE, "Augmented_Boards")

MODEL_PATHS = {
    "Black_Pawn" : os.path.join(BASE, "Chess_models", "Pawn_Black.obj"),
    "White_Pawn" : os.path.join(BASE, "Chess_models", "Pawn_white.obj"),
    "Black_Bishop" : os.path.join(BASE, "Chess_models", "Bishop_Black.obj"),
    "White_Bishop" : os.path.join(BASE, "Chess_models", "Bishop_White.obj"),
    "Black_Knight" : os.path.join(BASE, "Chess_models", "Knight_Black.obj"),
    "White_Knight" : os.path.join(BASE, "Chess_models", "Knight_White.obj"),
    "Black_Rook" : os.path.join(BASE, "Chess_models", "Rook_Black.obj"),
    "White_Rook" : os.path.join(BASE, "Chess_models", "Rook_White.obj"),
    "Black_Queen" : os.path.join(BASE, "Chess_models", "Queen_Black.obj"),
    "White_Queen" : os.path.join(BASE, "Chess_models", "Queen_White.obj"),
    "Black_King" : os.path.join(BASE, "Chess_models", "King_Black.obj"),
    "White_King" : os.path.join(BASE, "Chess_models", "King_White.obj"),
}

# =============== Constants ===============
SIDES = ("Black", "White")
PIECE_SCALE = {"Pawn": 0.53, 
              "Bishop": 0.82, 
              "Knight": 0.85,
              "Rook": 0.73, 
              "Queen": 0.95, 
              "King": 1.0
}

PIECE_CODE = {"Pawn": 1, 
             "Bishop": 2, 
             "Knight": 3, 
             "Rook": 4, 
             "Queen": 5, 
             "King": 6
}

KING_SCALE = 0.9

MODEL_SCALE = {
    "Black_Pawn" : KING_SCALE * 0.53,
    "White_Pawn" : KING_SCALE * 0.53,
    "Black_Bishop" : KING_SCALE * 0.82,
    "White_Bishop" : KING_SCALE * 0.82,
    "Black_Knight" : KING_SCALE * 0.85,
    "White_Knight" : KING_SCALE * 0.85,
    "Black_Rook" : KING_SCALE * 0.73,
    "White_Rook" : KING_SCALE * 0.73,
    "Black_Queen" : KING_SCALE * 0.95,
    "White_Queen" : KING_SCALE * 0.95,
    "Black_King" : KING_SCALE,
    "White_King" : KING_SCALE,
}

# Encoded piece: White positive, Black negative
PIECE_TO_CODE = {f"{s}_{k}": (code if s == "White" else -code)
                 for s in SIDES for k, code in PIECE_CODE.items()}
CODE_TO_PIECE = {code: name for name, code in PIECE_TO_CODE.items()}

# Rotation (degrees) applied to every model before placing
ROT_X = 90
ROT_Y = 0
ROT_Z = 0

# Lighting and texture
AMBIENT_LIGHT = 0.35
LIGHT_INTENSITY = 3.0
LIGHT_TILT_DEG = -30

PIECE_COLOUR = {
    "White": [0.85, 0.85, 0.85, 1.0],
    "Black": [0.12, 0.12, 0.12, 1.0],
}

# Camera calibration
with open("cal_data.pkl", "rb") as file:
    calibration = pickle.load(file)

camera_matrix = calibration["mtx"]
dist_coeffs = calibration["dist"]

# Cache to store mesh
_MESH_CACHE = {}

# Pyrender instance
_RENDERER = None

# Error tolerance
MAX_CORNER_ERROR_SQUARES = 0.6
# =============== Fucntions ===============
def rotate_piece(rx, ry, rz):
    """Rotation matrix (Rz @ Ry @ Rx) from angles in degrees"""
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_piece_mesh(piece):
    """Load a piece's mesh from its baked .npz, building the .npz from the .obj if needed"""
    # Checks if the is a chached model in either short term memory or in a baked .npz file
    obj_path = MODEL_PATHS[piece]
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Could not find {piece} model at: \n{obj_path}")

    baked_name = os.path.splitext(os.path.basename(obj_path))[0] + ".npz"
    baked_path = os.path.join(BAKED_DIR, baked_name)

    if os.path.isfile(baked_path) and os.path.getmtime(baked_path) >= os.path.getmtime(obj_path):
        with np.load(baked_path) as data:
            return trimesh.Trimesh(vertices=data["v"], faces=data["f"], process=False)

    loaded = trimesh.load(obj_path, force="scene", skip_materials=True)
    geoms = list(loaded.dump())
    if len(geoms) == 0:
        raise ValueError(f"No geometry found in {piece} model at: \n{obj_path}")

    # Flip inside-out meshes and save into cache
    with np.errstate(divide="ignore", invalid="ignore"):
        for g in geoms:
            trimesh.repair.fix_normals(g, multibody=True)

    mesh = trimesh.util.concatenate(geoms)

    os.makedirs(BAKED_DIR, exist_ok=True)
    np.savez(baked_path, v=mesh.vertices, f=mesh.faces)
    return mesh


def retrieve_mesh(piece):
    """Return (pyrender mesh, normalising transform) for a piece, cached in memory"""
    if piece in _MESH_CACHE:
        return _MESH_CACHE[piece]

    # Normalise piece and place the base of model at origin
    loaded_mesh = load_piece_mesh(piece)
    verts = loaded_mesh.vertices

    # Base centre (centred in x/z, resting at the minimum height) and model height
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    base_centre = np.array([(vmin[0] + vmax[0]) / 2, vmin[1], (vmin[2] + vmax[2]) / 2])
    piece_height = vmax[1] - vmin[1]

    # Scale to the target height, rotate upright, and move the base to the origin
    scale = MODEL_SCALE[piece] / piece_height
    rotated_piece = rotate_piece(ROT_X, ROT_Y, ROT_Z) * scale

    normalise = np.eye(4)
    normalise[:3, :3] = rotated_piece
    normalise[:3, 3] = -(rotated_piece @ base_centre)

    # Apply material
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=PIECE_COLOUR[piece.split("_")[0]],
        metallicFactor=0.0,
        roughnessFactor=0.7,
        alphaMode="OPAQUE",
        doubleSided=True,
    )
    mesh = pyrender.Mesh.from_trimesh(loaded_mesh, material=material)

    _MESH_CACHE[piece] = (mesh, normalise)
    return _MESH_CACHE[piece]


def place_piece(piece, row, col, board_pose):
    """Return a piece's mesh and the matrix that places it on square (row, col)"""
    mesh, normalise = retrieve_mesh(piece)
    translation = np.eye(4)
    translation[:3, 3] = np.asarray(board_pose["square_centres_board"][row, col], dtype=float)
    return mesh, translation @ normalise

def _solve_pose(result):
    """TrackerResult -> pose dict (raises RuntimeError if the pose is rejected)."""
    if not result.tracking_ok:
        raise RuntimeError("Board orientation/tracking not verified")

    # Raw solve; the 8 px absolute gate is bypassed, the relative gate is used below
    pose = get_board_pose(result, camera_matrix, dist_coeffs, square_size=1.0, max_corner_error_px=1_000_000.0)
    
    if not pose.success:
        raise RuntimeError(pose.reason)

    gate = evaluate_corner_gate(result.homography, pose.corner_errors_px,
                                mode="relative",
                                max_corner_error_squares=MAX_CORNER_ERROR_SQUARES)
    if not gate.accepted:
        raise RuntimeError(gate.reason)
    
    if pose.camera_position_board[2] <= 0:
        raise RuntimeError("Camera is behind the board plane")

    rows, cols = np.mgrid[0:8, 0:8]
    centres = np.stack([cols + 0.5, 8 - rows - 0.5, np.zeros((8, 8))], axis=-1)

    return {
        "camera_matrix": np.asarray(camera_matrix, dtype=float),
        "R": pose.R,
        "tvec": pose.tvec.reshape(3),
        "square_centres_board": centres,
        "pose_result": pose,
    }


def estimate_pose(board_rgb):
    """One-shot pose from a single still image (used by real_gui)"""
    # Convert to correct coour and finds and tracks board
    bgr = cv2.cvtColor(np.rint(np.clip(board_rgb, 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_RGB2BGR)
    detection = find_board.detect_chessboard(bgr)
    result = BoardTracker().initialize(bgr, detection)
    return _solve_pose(result)

class BoardLost(RuntimeError):
    """The board has not been detected for several attempts: stop drawing pieces"""

class LivePoseTracker:
    """Persistent tracker for video.

    Runs the (slow, ~250 ms) detector only to start, every `detect_every`
    frames, and after tracking is lost. Every other frame it follows the board
    with optical flow (a few ms), so the overlay moves with the board
    """

    # Initialise detection rate, retry and failure amounts and thresholds
    def __init__(self, detect_every=30, retry_every=5, max_failed_detections=3,
                 corner_contrast_threshold=12.0):
        self.detect_every = int(detect_every)
        self.retry_every = int(retry_every)
        self.max_failed = int(max_failed_detections)
        self.tracker = BoardTracker(corner_contrast_threshold=corner_contrast_threshold)
        self.n = 0
        self.next_detect = 0
        self.failed = 0

    # Reset behaviour
    def reset(self):
        self.tracker.reset()
        self.n = 0
        self.next_detect = 0
        self.failed = 0

    def solve(self, bgr):
        t = self.tracker
        n = self.n
        self.n += 1
        run_detector = (not t.initialised or n >= self.next_detect or not t.last_tracking_ok)

        result, error = None, None

        # Run slow detector
        if run_detector:
            try:
                detection = find_board.detect_chessboard(bgr)
                result = (t.update(bgr, detection=detection) if t.initialised
                          else t.initialize(bgr, detection))
                self.failed = 0
                self.next_detect = n + self.detect_every
            except (RuntimeError, ValueError, cv2.error) as exc:
                error = exc
                self.failed += 1
                self.next_detect = n + self.retry_every
                if self.failed >= self.max_failed:
                    self.reset()
                    raise BoardLost("Board not detected") from exc
                
        # Use optical flow
        if result is None and t.initialised:
            result = t.update(bgr)       

        if result is None:
            raise error if error is not None else RuntimeError("Board not found")
        return _solve_pose(result)


def opencv_to_gl_camera_pose(R, tvec):
    """Convert an OpenCV world-to-camera pose to a pyrender camera pose"""
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = tvec

    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(world_to_cam) @ flip_yz


def get_renderer(width, height):
    """Return the shared offscreen renderer, recreating it if the frame size changed"""
    global _RENDERER

    if _RENDERER is not None and (_RENDERER.viewport_width, _RENDERER.viewport_height) != (width, height):
        _RENDERER.delete()
        _RENDERER = None

    if _RENDERER is None:
        _RENDERER = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    return _RENDERER


def close_renderer():
    """Delete the shared renderer"""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.delete()
        _RENDERER = None


def board_to_pieces(board_array):
    """Converts array to a dictionary"""
    board_array = np.asarray(board_array)
    if board_array.shape != (8, 8):
        raise ValueError(f"Board array must be 8x8, got shape {board_array.shape}")

    pieces = {}
    for row, col in zip(*np.nonzero(board_array)):
        code = int(board_array[row, col])
        if code not in CODE_TO_PIECE:
            raise ValueError(f"Unknown piece code {code} at row {row}, col {col}")
        pieces.setdefault(CODE_TO_PIECE[code], []).append((int(row), int(col)))

    return {piece: tuple(squares) for piece, squares in pieces.items()}


def as_piece_dict(board_state):
    """Accept either an 8x8 code array or a {piece: squares} dict"""
    if isinstance(board_state, np.ndarray):
        return board_to_pieces(board_state)
    return board_state


def render_overlay(frame, pose, pieces, occluder=None, timings=None):
    """Render the pieces with a known pose and composite them onto the frame"""

    # Convert pieces and attain frame dim
    pieces = as_piece_dict(pieces)
    h, w = frame.shape[:2]
    t0 = time.perf_counter()

    # Retrieve extrinsic and intrinsic
    K = np.asarray(pose["camera_matrix"], dtype=float)
    R = np.asarray(pose["R"], dtype=float)
    tvec = np.asarray(pose["tvec"], dtype=float).reshape(3)

    # Convert to pyrender camera extrinsic
    cam_pose = opencv_to_gl_camera_pose(R, tvec)
    distance = float(np.linalg.norm(tvec))
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        znear=distance * 0.01, zfar=distance * 10,
    )

    # Create scene with light and camera
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[AMBIENT_LIGHT] * 3)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=LIGHT_INTENSITY)
    tilt = trimesh.transformations.rotation_matrix(math.radians(LIGHT_TILT_DEG), [1, 0, 0])
    scene.add(light, pose=cam_pose @ tilt)

    # Get piece mesh and place it in scene
    for piece, squares in pieces.items():
        for row, col in squares:
            mesh, placement = place_piece(piece, row, col, pose)
            scene.add(mesh, pose=placement)

    t_scene = time.perf_counter()

    # Render
    color, _ = get_renderer(w, h).render(scene, flags=pyrender.RenderFlags.RGBA)

    t_render = time.perf_counter()

    # Composite the entire scene
    out = frame.copy()
    alpha_full = color[..., 3].astype(np.float32) / 255
    if occluder is not None:
        alpha_full *= 1.0 - occluder
    mask = alpha_full > 0

    if mask.any():
        alpha = alpha_full[mask][:, None]
        piece_bgr = color[mask][:, [2, 1, 0]].astype(np.float32)  
        out[mask] = (piece_bgr * alpha + out[mask] * (1.0 - alpha)).astype(np.uint8)

    if timings is not None:
        timings.update(scene=t_scene - t0,
                       render=t_render - t_scene,
                       composite=time.perf_counter() - t_render)
    return out


def render_frame(board, pieces):
    """Estimate the pose of a still image and draw the pieces onto it"""
    t0 = time.perf_counter()
    board_rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    pose = estimate_pose(board_rgb)

    timings = {"pose": time.perf_counter() - t0}
    out_bgr = render_overlay(board, pose, pieces, timings=timings)
    return out_bgr, timings


def project_board(board_dir, pieces, image_name, show=True):
    """Projects the pieces onto a single board image. Used for static image only"""
    board = cv2.imread(board_dir)

    if board is None:
        raise FileNotFoundError(f"No board found at: \n{board_dir}")

    out_bgr, timings = render_frame(board, pieces)

    os.makedirs(OUT_PATH, exist_ok=True)
    out_file = os.path.join(OUT_PATH, image_name)
    if not cv2.imwrite(out_file, out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise IOError(f"Failed to write image to: {out_file}")

    for stage, seconds in timings.items():
        print(f"{stage:<10}{seconds * 1000:8.1f} ms")
    print(f"{'total':<10}{sum(timings.values()) * 1000:8.1f} ms")

    if show:
        cv2.imshow("Board state projected", out_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    board_state = np.array([
        [-4, -2, -3, -5, -6, -3, -2, -4],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  1,  1,  1,  1,  1,  1,  1],
        [ 4,  2,  3,  5,  6,  3,  2,  4],
    ], dtype=np.int8)
    board = os.path.join(BASE, "Captures_1", "WIN_20261019_11_05_33_Pro.jpg")
    project_board(board, board_state, "test.png", show=True)