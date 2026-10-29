import os
import math

import cv2
import numpy as np
import torch
import time

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import (Meshes, join_meshes_as_scene)
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    TexturesVertex,
)

import find_board

from get_pose import get_board_pose

import pickle

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

# =============== Constants ===============
# Defining scale and rotation of the piece
KING_SCALE = 0.9

MODEL_SCALE = {
    "Black_Pawn" : KING_SCALE * 0.53,
    "White_Pawn" : KING_SCALE * 0.53,
    "Black_Bishop" : KING_SCALE * 0.74,
    "White_Bishop" : KING_SCALE * 0.74,
    "Black_Knight" : KING_SCALE * 0.8,
    "White_Knight" : KING_SCALE * 0.8,
    "Black_Rook" : KING_SCALE * 0.58,
    "White_Rook" : KING_SCALE * 0.58,
    "Black_Queen" : KING_SCALE * 0.89,
    "White_Queen" : KING_SCALE * 0.89,
    "Black_King" : KING_SCALE,
    "White_King" : KING_SCALE,
}

PIECE_SCALE = 0.8
PAWN_SCALE = PIECE_SCALE / 2
ROT_X = 90
ROT_Y = 0
ROT_Z = 0

# Define the rendering method
device = torch.device("cpu")

with open("cal_data.pkl", "rb") as file:
    calibration = pickle.load(file)

camera_matrix = calibration["mtx"]
dist_coeffs = calibration["dist"]

_MESH_CACHE = {}

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
    Rx = torch.tensor([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ], dtype=torch.float32)

    Ry = torch.tensor([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=torch.float32)

    Rz = torch.tensor([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ], dtype=torch.float32)

    return Rz @ Ry @ Rx

def get_base_mesh(piece):
    """Gets the mesh and caches it for efficiency

    Args:
        piece (string): Name of piece

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_

    Returns:
        torch.Tensor: Mesh of piece
    """
    # Checks if piece has already had mesh loaded
    if piece in _MESH_CACHE:
        return _MESH_CACHE[piece]

    # Find mesh and texture
    if not os.path.isfile(MODEL_PATHS[piece]):
        raise FileNotFoundError(
            f"Could not find {piece} model at: \n{MODEL_PATHS[piece]}"
        )
    
    piece_mesh = load_objs_as_meshes([MODEL_PATHS[piece]], device = device, create_texture_atlas = True, texture_atlas_size = 2)
    if piece_mesh.textures is None:
        raise FileNotFoundError(
            f"No material supplied for : {piece}"
        )

    verts = piece_mesh.verts_packed()

    # Centre model horizontally
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values

    # Calculate the midpoint of X and Z and minimum of Y as base
    base_centre = torch.stack([(vmin[0] + vmax[0]) / 2, vmin[1], (vmin[2] + vmax[2]) / 2])

    piece_height = float((vmax[1] - vmin[1]).item())

    # Centre base to origin
    verts = (verts - base_centre) / piece_height

    # Scale piece
    verts = verts * MODEL_SCALE[piece]
    # if ((piece == "Black_Pawn") or (piece == "White_Pawn")):
    #     verts = verts * PAWN_SCALE
    # else:
    #     verts = verts * PIECE_SCALE

    # Rotate the piece
    rotation_matrix = rotate_piece(ROT_X, ROT_Y, ROT_Z).to(device)
    verts = verts @ rotation_matrix.T
    
    piece_mesh = piece_mesh.update_padded(verts.unsqueeze(0))
    _MESH_CACHE[piece] = piece_mesh
    return piece_mesh

def build_piece_mesh(piece, row, col, pose):
    """Builds the one mesh at a time

    Args:
        piece (string): Piece to build
        row (int): Row to place piece at
        col (int): Col to place piece at
        pose (dict): Output from get_board_pos

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_

    Returns:
        torch.Tensor: The mesh of the piece and where its located
    """
    piece_mesh = get_base_mesh(piece)
    target_square = pose["square_centres_board"][row, col]
    piece_x_coord = float(target_square[0])
    piece_y_coord = float(target_square[1])
    piece_z_coord = float(target_square[2])

    # Move the piece to the correct location
    piece_position = torch.tensor([piece_x_coord, piece_y_coord, piece_z_coord], dtype = torch.float32, device = device)
    verts = piece_mesh.verts_packed() + piece_position
    return piece_mesh.update_padded(verts.unsqueeze(0))

def project_board(board_dir, pieces, image_name):
    """Projects the pieces onto the board

    Args:
        board_dir (dir): Directory to the board image to project onto
        pieces (dict): Dictionary of piece and (row,col). Must be tuple even if single piece
        image_name (string): File name to write image as

    Raises:
        FileNotFoundError: _description_
    """
    start = time.perf_counter()

    # Load the board in, convert to RGB and normalise
    board = cv2.imread(board_dir)

    if board is None:
        raise FileNotFoundError(f"No board found at: \n{board_dir}")

    Height, Width = board.shape[:2]

    board_rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    # Find centres, corners and inner
    square_centres, board_corners, inner = find_board.detect_chessboard(board_rgb)

    valid_point_count = np.isfinite(inner).all(axis = 2).sum()

    # Get pose
    if valid_point_count >=4:
        pose = get_board_pose(
            board_corners, 
            square_centres, 
            observed_image_points = inner,
            camera_matrix = camera_matrix,
            dist_coeffs = dist_coeffs
        )
        
    else:
        pose = get_board_pose(
            board_corners,
            square_centres
        )

    meshes = []
    for piece, squares in pieces.items():
        for row, col in squares:
            meshes.append(build_piece_mesh(piece, row, col, pose))

    scene = join_meshes_as_scene(meshes)

    # Retrieve camera intrinsic, rotation and translation vectors
    K = pose["camera_matrix"]
    R = pose["R"]
    tvec = pose["tvec"].reshape(3)

    # Conversion between OpenCV and Pytorch
    OpenCV_to_PyTorch = torch.diag(torch.tensor([-1, -1, 1], dtype = torch.float32, device = device))

    # Define camera params in Pytorch
    R_tran = torch.tensor(R.T, dtype = torch.float32, device = device)
    T_tran = torch.tensor(tvec.T, dtype = torch.float32, device = device)

    R_camera = (R_tran @ OpenCV_to_PyTorch).unsqueeze(0)
    T_camera = (T_tran @ OpenCV_to_PyTorch).unsqueeze(0)
    K_torch = torch.tensor(K, dtype = torch.float32, device = device)

    focal_length = torch.stack([K_torch[0, 0], K_torch[1, 1]]).reshape(1, 2)
    principal_point = torch.stack([K_torch[0, 2], K_torch[1, 2]]).reshape(1, 2)

    image_size = torch.tensor([[Height, Width]], dtype = torch.float32, device = device)

    # Define the camera in pytorch
    cameras = PerspectiveCameras(
        device = device,
        R = R_camera,
        T = T_camera,
        focal_length = focal_length,
        principal_point = principal_point,
        image_size = image_size,
        in_ndc = False
    )

    # Settings to display image
    raster_settings = RasterizationSettings(
        image_size = (Height, Width),
        blur_radius = 0,
        faces_per_pixel = 1
    )

    lights = PointLights(
        device = device,
        location = [[2, 3, 3]],
    )

    rasterizer = MeshRasterizer(
        cameras = cameras,
        raster_settings = raster_settings,
    )

    shader = HardPhongShader(
        device = device,
        cameras = cameras,
        lights = lights,
    )

    with torch.no_grad():
        fragments = rasterizer(scene)
        rendered = shader(fragments, scene)

    rgb = rendered[0, ..., :3].cpu().numpy()

    # Create a mask where the pawn covers the image.
    alpha = (fragments.pix_to_face[0, ..., 0:1] >= 0).float().cpu().numpy()

    # Render final image and save
    out = rgb * alpha + board_rgb * (1.0 - alpha)
    out_bgr = cv2.cvtColor((np.clip(out, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    os.makedirs(OUT_PATH, exist_ok=True)
    out_file = os.path.join(OUT_PATH, image_name)
    cv2.imwrite(out_file, out_bgr)
    print(f"Total: {time.perf_counter() - start:.2f}s")

    cv2.imshow(f"Board state projected", out_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    board = os.path.join(BASE, "Captures_1", "WIN_20261019_11_05_33_Pro.jpg")
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

    project_board(board, board_state, "Starting_board_3.png")