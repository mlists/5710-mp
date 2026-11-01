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
from get_pose_v6 import get_board_pose
from pose_quality import evaluate_corner_gate

# =============== Paths ===============
BASE = os.path.dirname(os.path.abspath(__file__))

# Defining the dictionary to store each piece and their model directory
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

BAKED_DIR = os.path.join(BASE, "Chess_models", "_baked")

OUT_PATH = os.path.join(BASE, "Augmented_Boards")

# =============== Constants ===============
# Rotation (degrees) applied to every model before placing
ROT_X = 90
ROT_Y = 0
ROT_Z = 0

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

# Lighting: a directional light attached to the camera, tilted to come from above
AMBIENT_LIGHT = 0.35
LIGHT_INTENSITY = 3.0
LIGHT_TILT_DEG = -30

# Colour texture of each piece (mtl was broken)
PIECE_COLOUR = {
    "White": [0.85, 0.85, 0.85, 1.0],
    "Black": [0.12, 0.12, 0.12, 1.0],
}

# Encoded piece
PIECE_TO_CODE = {
    "Black_Pawn" : -1,
    "White_Pawn" : 1,
    "Black_Bishop" : -2,
    "White_Bishop" : 2,
    "Black_Knight" : -3,
    "White_Knight" : 3,
    "Black_Rook" : -4,
    "White_Rook" : 4,
    "Black_Queen" : -5,
    "White_Queen" : 5,
    "Black_King" : -6,
    "White_King" : 6,
}
CODE_TO_PIECE = {code: name for name, code in PIECE_TO_CODE.items()}

# Camera calibration
with open("cal_data.pkl", "rb") as file:
    calibration = pickle.load(file)

camera_matrix = calibration["mtx"]
dist_coeffs = calibration["dist"]

# Cache to store mesh
_MESH_CACHE = {}

# Pyrender instance
_RENDERER = None

# =============== Functions ===============
def rotate_piece(rx, ry, rz):
    """Function to rotate the model before placing

    Args:
        rx (float): The rotation in degrees in x axis
        ry (float): The rotation in degrees in y axis
        rz (float): The rotation in degrees in z axis

    Returns:
        np.array: The rotation matrix
    """

    # Convert to radians
    rx, ry, rz = map(math.radians, (rx, ry, rz))

    # Define the sine and cosine of each rotation
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    
    # Define the transformation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ])

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ])

    return Rz @ Ry @ Rx

def load_piece_mesh(piece):
    """Loads the mesh of model. Either it creates and stores as .npz or retrieves it

    Args:
        piece (string): The piece to get the model for

    Returns:
        Trimesh: mesh of the model
    """
    # Check if there is an .obj file at a given directory
    obj_path = MODEL_PATHS[piece]
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Could not find {piece} model at: \n{obj_path}")

    # Change the format of an .obj to .npz to prepocess
    baked_name = os.path.splitext(os.path.basename(obj_path))[0] + ".npz"
    baked_path = os.path.join(BAKED_DIR, baked_name)

    # Checks if a baked copy of the model exists and is the mot recent
    if os.path.isfile(baked_path) and os.path.getmtime(baked_path) >= os.path.getmtime(obj_path):
        with np.load(baked_path) as data:
            return trimesh.Trimesh(vertices = data["v"], faces = data["f"], process = False)

    # Create the baked model to store
    loaded = trimesh.load(obj_path, force = "scene", skip_materials = True)
    geoms = list(loaded.dump())
    if len(geoms) == 0:
        raise ValueError(f"No geometry found in {piece} model at: \n{obj_path}")

    # Flip inside-out meshes
    with np.errstate(divide = "ignore", invalid = "ignore"):
        for g in geoms:
            trimesh.repair.fix_normals(g, multibody = True)

    mesh = trimesh.util.concatenate(geoms)

    os.makedirs(BAKED_DIR, exist_ok = True)
    np.savez(baked_path, v = mesh.vertices, f = mesh.faces)
    return mesh

def retrieve_mesh(piece):
    """Retrieve the mesh and prep for rendering

    Args:
        piece (string): Piece to render

    Returns:
        Trimesh: Mesh of piece
    """
    # If mesh has been stored retrieve
    if piece in _MESH_CACHE:
        return _MESH_CACHE[piece]

    # Retrieve the raw mesh
    loaded_mesh = load_piece_mesh(piece)
    verts = loaded_mesh.vertices

    # Find the centre of the model and dimensions
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    base_centre = np.array([(vmin[0] + vmax[0]) / 2, vmin[1], (vmin[2] + vmax[2]) / 2])
    piece_height = vmax[1] - vmin[1]

    # Get a matrix for scaling and rotation
    scale = MODEL_SCALE[piece] / piece_height 
    rotated_piece = rotate_piece(ROT_X, ROT_Y, ROT_Z) * scale

    # Normalise a piece to the orgiin and apply scaling and rotation
    normalise = np.eye(4)
    normalise[:3, :3] = rotated_piece
    normalise[:3, 3] = -(rotated_piece @ base_centre)

    # Define materials to render objects
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor = PIECE_COLOUR[piece.split("_")[0]],
        metallicFactor = 0.0,
        roughnessFactor = 0.7,
        alphaMode = "OPAQUE",
        doubleSided = True,
    )

    # Render mesh in pyrenderer
    mesh = pyrender.Mesh.from_trimesh(loaded_mesh, material = material)

    # Cache the piece mesh and the transformation matrix
    _MESH_CACHE[piece] = (mesh, normalise)
    return _MESH_CACHE[piece]

def place_piece(piece, row, col, board_pose):
    """Gets the mesh and defines where to place it

    Args:
        piece (String): Piece name
        row (int): Row to place piece
        col (int): Col to place piece
        board_pose (dict): Board pose

    Returns:
        trimesh, np.array: The mesh of the piece and matrix to place
    """
    mesh, normalise = retrieve_mesh(piece)
    translation = np.eye(4)

    # Defines where to place the piece
    translation[:3, 3] = np.asarray(board_pose["square_centres_board"][row, col], dtype=float)[:3]
    return mesh, translation @ normalise


# Board units are SQUARES (1 square = 1.0), which is what MODEL_SCALE assumes.
# Pose is accepted if every board corner reprojects within this fraction of a
# locally projected square (same "relative" gate as find_board_video_v8).
# 0.35 is right for a CORRECT calibration at the video's resolution. If the
# status bar keeps saying "relative gate FAIL", recalibrate at that resolution;
# raising this only hides the error (the overlay will sit slightly off).
MAX_CORNER_ERROR_SQUARES = 0.6


class BoardLost(RuntimeError):
    """The board has not been detected for several attempts: stop drawing pieces."""


def _solve_pose(result):
    """TrackerResult -> pose dict (raises RuntimeError if the pose is rejected)."""
    if not result.tracking_ok:
        raise RuntimeError("Board orientation/tracking not verified")

    # Raw solve; the 8 px absolute gate is bypassed, the relative gate is used below
    pose = get_board_pose(result, camera_matrix, dist_coeffs,
                          square_size = 1.0, max_corner_error_px = 1_000_000.0)
    if not pose.success:
        raise RuntimeError(pose.reason)

    gate = evaluate_corner_gate(result.homography, pose.corner_errors_px,
                                mode = "relative",
                                max_corner_error_squares = MAX_CORNER_ERROR_SQUARES)
    if not gate.accepted:
        raise RuntimeError(gate.reason)
    if pose.camera_position_board[2] <= 0:
        raise RuntimeError("Camera is behind the board plane")

    # Square centres in board coordinates, matching get_pose_v6's convention:
    # origin = tracker corner 3 (bottom-left), +X along columns, +Y up the rows
    rows, cols = np.mgrid[0:8, 0:8]
    centres = np.stack([cols + 0.5, 8 - rows - 0.5, np.zeros((8, 8))], axis = -1)

    return {
        "camera_matrix": np.asarray(camera_matrix, dtype = float),
        "R": pose.R,
        "tvec": pose.tvec.reshape(3),
        "square_centres_board": centres,
        "pose_result": pose,
    }


def estimate_pose(board_rgb):
    """One-shot pose from a single still image (used by real_gui).

    Pipeline (same as find_board_video_v8): find_board detector ->
    board_tracker_v2 (orientation-safe corner IDs) -> get_pose_v6 (4-corner PnP)
    -> relative corner-error gate.

    Args:
        board_rgb: float32 RGB image in [0, 1].

    Returns:
        (pose dict, True). Raises if the board is not found, its orientation
        cannot be verified, or the pose is rejected.
    """
    # find_board and the tracker expect a uint8 BGR image
    bgr = cv2.cvtColor(np.rint(np.clip(board_rgb, 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_RGB2BGR)
    detection = find_board.detect_chessboard(bgr)
    result = BoardTracker().initialize(bgr, detection)
    return _solve_pose(result), True


class LivePoseTracker:
    """Persistent tracker for video.

    Runs the (slow, ~250 ms) detector only to start, every `detect_every`
    frames, and after tracking is lost. Every other frame it follows the board
    with optical flow (a few ms), so the overlay moves with the board.

    If a re-detection fails (e.g. a hand covers the corners) it keeps tracking
    and retries every `retry_every` frames. After `max_failed_detections`
    failed attempts in a row the board is considered gone: the tracker resets
    and BoardLost is raised, so the caller stops drawing pieces. It starts
    again automatically as soon as the board is detected.
    """

    def __init__(self, detect_every = 30, retry_every = 5, max_failed_detections = 3,
                 corner_contrast_threshold = 12.0):
        self.detect_every = int(detect_every)
        self.retry_every = int(retry_every)
        self.max_failed = int(max_failed_detections)
        self.tracker = BoardTracker(corner_contrast_threshold = corner_contrast_threshold)
        self.n = 0
        self.next_detect = 0
        self.failed = 0

    def reset(self):
        self.tracker.reset()
        self.n = 0
        self.next_detect = 0
        self.failed = 0

    def solve(self, bgr):
        """bgr: uint8 BGR frame. Returns a pose dict or raises (BoardLost = board gone)."""
        t = self.tracker
        n = self.n
        self.n += 1
        run_detector = (not t.initialised or n >= self.next_detect or not t.last_tracking_ok)

        result, error = None, None
        if run_detector:
            try:
                detection = find_board.detect_chessboard(bgr)
                result = (t.update(bgr, detection = detection) if t.initialised
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
        if result is None and t.initialised:
            result = t.update(bgr)               # optical flow only
        if result is None:
            raise error if error is not None else RuntimeError("Board not found")
        return _solve_pose(result)

def opencv_to_gl_camera_pose(R, tvec):
    """Conver from opencv to pyrender

    Args:
        R (np.array): Rotation matrix
        tvec (np.array): Translation matrix

    Returns:
        np.array: Transformation matrix from opencv to pyrender
    """
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = tvec

    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(world_to_cam) @ flip_yz


def get_renderer(width, height):
    """Generates the pyrenderer instance

    Args:
        width (int): Image width
        height (int): Image height

    Returns:
        pyrender.OffscreenRenderer: Rendere to recall multiple times
    """
    global _RENDERER

    # Delete renderer if the image width and hieght do not match current instance
    if _RENDERER is not None and (_RENDERER.viewport_width, _RENDERER.viewport_height) != (width, height):
        _RENDERER.delete()
        _RENDERER = None
    
    if _RENDERER is None:
        _RENDERER = pyrender.OffscreenRenderer(viewport_width = width, viewport_height = height)
    return _RENDERER

def close_renderer():
    """Deletes the renderer
    """
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.delete()
        _RENDERER = None

def board_to_pieces(board_array):
    """Generates a dictionary of the board state

    Args:
        board_array (np.array): Numpy array of the board

    Returns:
        Dictionary: Contains the piece name and the square its located in
    """
    # Convert the board as a numpy array
    board_array = np.asarray(board_array)
    if board_array.shape != (8, 8):
        raise ValueError(f"Board array must be 8x8, got shape {board_array.shape}")

    # Dictionary to store all occupied squrs
    pieces = {}

    # For all rows and columns that have a number thats not 0
    for row, col in zip(*np.nonzero(board_array)):
        # Get the code representing the piece
        code = int(board_array[row, col])
        if code not in CODE_TO_PIECE:
            raise ValueError(f"Unknown piece code {code} at row {row}, col {col}")

        # Add pieces to dictionary and defaults to empty list
        pieces.setdefault(CODE_TO_PIECE[code], []).append((int(row), int(col)))

    return {piece: tuple(squares) for piece, squares in pieces.items()}

def as_piece_dict(board_state):
    """Conversion between dictionary and numpy array

    Args:
        board_state (numpy array): Board state

    Returns:
        dictionary: Dictionary of board state
    """
    if isinstance(board_state, np.ndarray):
        return board_to_pieces(board_state)
    return board_state

def render_frame(board, pieces):
    """Draws the pieces onto one frame

    Args:
        board (np.array): Board state
        pieces (np.array or dict): Pieces and their location

    Returns:
        tuple: The rendered board and timings of each process
    """
    pieces = as_piece_dict(pieces)

    t0 = time.perf_counter()

    Height, Width = board.shape[:2]

    # Normalise and convert to RGB
    board_rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    # Find the board and the camera pose
    pose, _ = estimate_pose(board_rgb)
    t_pose = time.perf_counter()

    # Retrieve camera intrinsic, rotation and translation vectors
    K = np.asarray(pose["camera_matrix"], dtype=float)
    R = np.asarray(pose["R"], dtype=float)
    tvec = np.asarray(pose["tvec"], dtype=float).reshape(3)

    cam_pose = opencv_to_gl_camera_pose(R, tvec)

    # Clip planes relative to camera distance so this works in any board units
    distance = float(np.linalg.norm(tvec))

    # Define pyrender camera
    camera = pyrender.IntrinsicsCamera(
        fx = K[0, 0], fy = K[1, 1],
        cx = K[0, 2], cy = K[1, 2],
        znear = distance * 0.01,
        zfar = distance * 10,
    )

    # Build the scene
    scene = pyrender.Scene(bg_color = [0, 0, 0, 0], ambient_light = [AMBIENT_LIGHT] * 3)
    scene.add(camera, pose = cam_pose)

    light = pyrender.DirectionalLight(color = np.ones(3), intensity = LIGHT_INTENSITY)
    light_tilt = trimesh.transformations.rotation_matrix(math.radians(LIGHT_TILT_DEG), [1, 0, 0])
    scene.add(light, pose = cam_pose @ light_tilt)

    # Place the pieces into the scene
    for piece, squares in pieces.items():
        for row, col in squares:
            mesh, placement = place_piece(piece, row, col, pose)
            scene.add(mesh, pose = placement)

    t_scene = time.perf_counter()

    # Render (the renderer and its GPU buffers are reused between calls)
    renderer = get_renderer(Width, Height)
    color, _ = renderer.render(scene, flags = pyrender.RenderFlags.RGBA)

    t_render = time.perf_counter()

    # Composite using the render's alpha channel, only where a piece was drawn
    mask = color[..., 3] > 0
    out_bgr = board.copy()
    if mask.any():
        alpha = color[mask, 3:4].astype(np.float32) / 255
        piece_bgr = color[mask][:, [2, 1, 0]].astype(np.float32)   # RGB -> BGR
        out_bgr[mask] = (piece_bgr * alpha + out_bgr[mask] * (1.0 - alpha)).astype(np.uint8)

    t_end = time.perf_counter()

    timings = {
        "pose": t_pose - t0,
        "scene": t_scene - t_pose,
        "render": t_render - t_scene,
        "composite": t_end - t_render,
    }
    return out_bgr, timings

def project_board(board_dir, pieces, image_name, show = True):
    """Projects the pieces onto a single board image

    Args:
        board_dir (dir): Directory to the board image to project onto
        pieces (np.array or dict): Pieces and their location
        image_name (string): File name to write image as
        show (bool): Whether to open a preview window (blocks until a key is pressed)
    """
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
    return

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
        ], dtype = np.int8)
    board = os.path.join(BASE, "Captures_1", "WIN_20261019_11_05_33_Pro.jpg")
    project_board(board, board_state, "test.png", show = True)