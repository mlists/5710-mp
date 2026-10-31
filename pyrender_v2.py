import os
import math
import time
import pickle

import cv2
import numpy as np
import trimesh
import pyrender

import find_board
from get_pose import get_board_pose

# On a headless Linux box, set this BEFORE pyrender is imported:
#   os.environ["PYOPENGL_PLATFORM"] = "egl"     (GPU)  or  "osmesa"  (CPU)
# On Windows/macOS with a display, no setting is needed.

# =============== Defining Paths ===============
# Defining the base path the file is located in
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

# Defining the output path
OUT_PATH = os.path.join(BASE, "Augmented_Boards")

# Fixed models are saved here the first time each piece is loaded, so later runs
# skip the slow OBJ parsing and normal fixing. Delete this folder to force a
# rebuild (it also rebuilds by itself if an .obj is newer than its baked copy).
BAKED_DIR = os.path.join(BASE, "Chess_models", "_baked")

# =============== Constants ===============
# Final height of each piece, in board units
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

# Rotation (degrees) applied to every model before placing
ROT_X = 90
ROT_Y = 0
ROT_Z = 0

# Lighting: a directional light attached to the camera, tilted to come from above
AMBIENT_LIGHT = 0.35
LIGHT_INTENSITY = 3.0
LIGHT_TILT_DEG = -30

# Flat colour per side (the .mtl colours in the models are ignored)
PIECE_COLOUR = {
    "White": [0.85, 0.85, 0.85, 1.0],
    "Black": [0.12, 0.12, 0.12, 1.0],
}

# --- Undistort ---------------------------------------------------------------
# The pose is solved on the raw (distorted) photo using dist_coeffs, so it is
# physically correct. pyrender, however, draws with an ideal pinhole camera (no
# lens distortion), so the pieces only line up with the photo if the photo you
# draw on is undistorted too.
#
# CAN YOU REMOVE IT? Set UNDISTORT = False if either:
#   * the "Undistort moves the image corners by up to N px" line printed at run
#     time shows about 2 px or less (this lens has negligible distortion), or
#   * the pieces sit on their squares near the edges and corners of the image
#     without it.
# KEEP IT if pieces drift off their squares toward the edges of the image.
# Do NOT use it if the photo was already undistorted before reaching this
# script, or it would be corrected twice.
# It is skipped automatically when the pose was solved without calibration data.
UNDISTORT = True

with open("cal_data.pkl", "rb") as file:
    calibration = pickle.load(file)

camera_matrix = calibration["mtx"]
dist_coeffs = calibration["dist"]

# piece name -> (pyrender.Mesh, 4x4 normalising matrix)
_MESH_CACHE = {}

# One OpenGL context is created and reused (creating it every call is slow)
_RENDERER = None

# =============== Functions ===============
def rotate_piece(rx, ry, rz):
    """Function to rotate the model before placing

    Args:
        rx (float): The rotation in degrees in x axis
        ry (float): The rotation in degrees in y axis
        rz (float): The rotation in degrees in z axis

    Returns:
        np.ndarray: The 3x3 rotation matrix
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


def load_piece_geometry(piece):
    """Returns the fixed geometry of a piece as a single trimesh.Trimesh

    First call for a piece: parses the OBJ (materials skipped, since colours are
    overridden anyway), flips inside-out faces, and saves the result to
    BAKED_DIR as a small .npz. Later calls just load that file.

    Raises:
        FileNotFoundError: If the model file is missing
        ValueError: If the model contains no geometry
    """
    obj_path = MODEL_PATHS[piece]
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Could not find {piece} model at: \n{obj_path}")

    baked_name = os.path.splitext(os.path.basename(obj_path))[0] + ".npz"
    baked_path = os.path.join(BAKED_DIR, baked_name)

    # Fast path: baked copy exists and is newer than the OBJ
    if os.path.isfile(baked_path) and os.path.getmtime(baked_path) >= os.path.getmtime(obj_path):
        with np.load(baked_path) as data:
            return trimesh.Trimesh(vertices = data["v"], faces = data["f"], process = False)

    # Slow path
    loaded = trimesh.load(obj_path, force = "scene", skip_materials = True)
    geoms = list(loaded.dump())
    if len(geoms) == 0:
        raise ValueError(f"No geometry found in {piece} model at: \n{obj_path}")

    # Flip inside-out meshes. Open meshes (e.g. the knights) have no volume,
    # which makes numpy warn, so silence those.
    with np.errstate(divide = "ignore", invalid = "ignore"):
        for g in geoms:
            trimesh.repair.fix_normals(g, multibody = True)

    mesh = trimesh.util.concatenate(geoms)

    os.makedirs(BAKED_DIR, exist_ok = True)
    np.savez(baked_path, v = mesh.vertices, f = mesh.faces)
    return mesh


def get_base_mesh(piece):
    """Loads a piece once and caches it

    The mesh itself is never modified. Instead we store a 4x4 matrix that
    centres the base on the origin, scales to the piece's final height and
    rotates it. Placing a piece is then just: translation @ matrix.

    Args:
        piece (string): Name of piece

    Returns:
        tuple: (pyrender.Mesh, 4x4 np.ndarray normalising matrix)
    """
    # Checks if piece has already been loaded
    if piece in _MESH_CACHE:
        return _MESH_CACHE[piece]

    geometry = load_piece_geometry(piece)
    verts = geometry.vertices

    # Calculate the midpoint of X and Z and minimum of Y as base
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    base_centre = np.array([(vmin[0] + vmax[0]) / 2, vmin[1], (vmin[2] + vmax[2]) / 2])
    piece_height = vmax[1] - vmin[1]

    # Same maths as before: rotate @ ((v - base_centre) / height * scale)
    scale = MODEL_SCALE[piece] / piece_height
    linear = rotate_piece(ROT_X, ROT_Y, ROT_Z) * scale

    normalise = np.eye(4)
    normalise[:3, :3] = linear
    normalise[:3, 3] = -(linear @ base_centre)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor = PIECE_COLOUR[piece.split("_")[0]],
        metallicFactor = 0.0,
        roughnessFactor = 0.7,
        alphaMode = "OPAQUE",
        doubleSided = True,
    )

    mesh = pyrender.Mesh.from_trimesh(geometry, material = material)
    _MESH_CACHE[piece] = (mesh, normalise)
    return _MESH_CACHE[piece]


def piece_pose(piece, row, col, board_pose):
    """4x4 matrix that places a cached piece on a square

    Args:
        piece (string): Piece to place
        row (int): Row to place piece at
        col (int): Col to place piece at
        board_pose (dict): Output from get_board_pose

    Returns:
        tuple: (pyrender.Mesh, 4x4 np.ndarray pose in board coordinates)
    """
    mesh, normalise = get_base_mesh(piece)

    translation = np.eye(4)
    translation[:3, 3] = np.asarray(board_pose["square_centres_board"][row, col], dtype=float)[:3]

    return mesh, translation @ normalise


def estimate_pose(board_rgb):
    """Detects the board and estimates the camera pose relative to it

    Returns:
        tuple: (pose dict, bool) where the bool is True when the pose was solved
        with the calibration (camera matrix + distortion coefficients)
    """
    square_centres, board_corners, inner = find_board.detect_chessboard(board_rgb)

    valid_point_count = np.isfinite(inner).all(axis = 2).sum()

    if valid_point_count >= 4:
        pose = get_board_pose(
            board_corners,
            square_centres,
            observed_image_points = inner,
            camera_matrix = camera_matrix,
            dist_coeffs = dist_coeffs
        )
        return pose, True

    return get_board_pose(board_corners, square_centres), False


def opencv_to_gl_camera_pose(R, tvec):
    """Converts an OpenCV extrinsic (X_cam = R X + t) to a pyrender camera pose

    OpenCV cameras look down +Z with +Y pointing down. OpenGL/pyrender cameras
    look down -Z with +Y pointing up, and the pose is camera-to-world, not
    world-to-camera.

    Returns:
        np.ndarray: 4x4 camera-to-world matrix in the OpenGL convention
    """
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = tvec

    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(world_to_cam) @ flip_yz


def get_renderer(width, height):
    """Returns the shared offscreen renderer, recreating it only if the size changes"""
    global _RENDERER
    if _RENDERER is not None and (_RENDERER.viewport_width, _RENDERER.viewport_height) != (width, height):
        _RENDERER.delete()
        _RENDERER = None
    if _RENDERER is None:
        _RENDERER = pyrender.OffscreenRenderer(viewport_width = width, viewport_height = height)
    return _RENDERER


def close_renderer():
    """Frees the OpenGL context. Call once when you are finished rendering"""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.delete()
        _RENDERER = None


def project_board(board_dir, pieces, image_name, show = True):
    """Projects the pieces onto the board

    Args:
        board_dir (dir): Directory to the board image to project onto
        pieces (dict): Dictionary of piece and (row,col). Must be tuple even if single piece
        image_name (string): File name to write image as
        show (bool): Whether to open a preview window (blocks until a key is pressed)

    Raises:
        FileNotFoundError: If the board image can't be read
    """
    t0 = time.perf_counter()

    # Load the board in, convert to RGB and normalise (detection wants float RGB)
    board = cv2.imread(board_dir)

    if board is None:
        raise FileNotFoundError(f"No board found at: \n{board_dir}")

    Height, Width = board.shape[:2]

    board_rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    # Find the board and the camera pose (done on the raw, distorted photo)
    pose, used_calibration = estimate_pose(board_rgb)
    t_pose = time.perf_counter()

    # Retrieve camera intrinsic, rotation and translation vectors
    K = np.asarray(pose["camera_matrix"], dtype=float)
    R = np.asarray(pose["R"], dtype=float)
    tvec = np.asarray(pose["tvec"], dtype=float).reshape(3)

    cam_pose = opencv_to_gl_camera_pose(R, tvec)

    # Clip planes relative to camera distance so this works in any board units
    distance = float(np.linalg.norm(tvec))

    camera = pyrender.IntrinsicsCamera(
        fx = K[0, 0], fy = K[1, 1],
        cx = K[0, 2], cy = K[1, 2],
        znear = distance * 0.01,
        zfar = distance * 10,
    )

    # Build the scene. Transparent background so alpha = where a piece was drawn
    scene = pyrender.Scene(bg_color = [0, 0, 0, 0], ambient_light = [AMBIENT_LIGHT] * 3)
    scene.add(camera, pose = cam_pose)

    light = pyrender.DirectionalLight(color = np.ones(3), intensity = LIGHT_INTENSITY)
    light_tilt = trimesh.transformations.rotation_matrix(math.radians(LIGHT_TILT_DEG), [1, 0, 0])
    scene.add(light, pose = cam_pose @ light_tilt)

    # The same pyrender.Mesh is reused for every instance of a piece
    for piece, squares in pieces.items():
        for row, col in squares:
            mesh, placement = piece_pose(piece, row, col, pose)
            scene.add(mesh, pose = placement)

    t_mesh = time.perf_counter()

    # Render (the renderer and its GPU buffers are reused between calls)
    renderer = get_renderer(Width, Height)
    color, _ = renderer.render(scene, flags = pyrender.RenderFlags.RGBA)

    t_render = time.perf_counter()
    
    t_undistort = time.perf_counter()

    # Composite using the render's alpha channel, only where a piece was drawn
    mask = color[..., 3] > 0
    out_bgr = board.copy()
    if mask.any():
        alpha = color[mask, 3:4].astype(np.float32) / 255
        piece_bgr = color[mask][:, [2, 1, 0]].astype(np.float32)   # RGB -> BGR
        out_bgr[mask] = (piece_bgr * alpha + out_bgr[mask] * (1.0 - alpha)).astype(np.uint8)

    os.makedirs(OUT_PATH, exist_ok=True)
    out_file = os.path.join(OUT_PATH, image_name)
    if not cv2.imwrite(out_file, out_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise IOError(f"Failed to write image to: {out_file}")

    t_end = time.perf_counter()
    print(f"Detect + pose: {t_pose - t0:.2f}s")
    print(f"Scene build:   {t_mesh - t_pose:.2f}s")
    print(f"Render:        {t_render - t_mesh:.2f}s")
    print(f"Undistort:     {t_undistort - t_render:.2f}s")
    print(f"Composite+save:{t_end - t_undistort:.2f}s")
    print(f"Total:         {t_end - t0:.2f}s")

    if show:
        cv2.imshow("Board state projected", out_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    board = os.path.join(BASE, "Captures_1", "WIN_20261019_11_04_35_Pro.jpg")
    board_state = {
        "Black_King": ((0,4), ),
        "White_King": ((7,4), ),
        "Black_Queen": ((0,3), ),
        "White_Queen": ((7,3), ),
        "Black_Rook": ((0,0), (0,7)),
        "White_Rook": ((7,0), (7,7)),
        "Black_Knight": ((0,1), (0,6)),
        "White_Knight": ((7,1), (7,6)),
        "Black_Bishop": ((0,2), (0,5)),
        "White_Bishop": ((7,2), (7,5)),
        "Black_Pawn": ((1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7)),
        "White_Pawn": ((6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7)),
    }

    project_board(board, board_state, "Starting_board_4.png")
    close_renderer()