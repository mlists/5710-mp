import cv2
import numpy as np
import matplotlib.pyplot as plt

# Chessboard geometry
BOARD_SIZE = 8
INNER_CORNERS_PER_SIDE = BOARD_SIZE - 1
GRID_ANCHOR_COORDINATE = np.array([5, 5], dtype=np.float32)
# The standard OpenCV detector fits the 7x7 internal intersections.
GRID_COORDINATES = np.array(
    [(x, y) for y in range(1, BOARD_SIZE) for x in range(1, BOARD_SIZE)],
    dtype=np.float32,
)

# The fallback searches a larger 9x9 lattice, including the board perimeter.
# Coordinates 0 and BOARD_SIZE represent the outer edge of the board.
EXPANDED_GRID_COORDINATES = np.array(
    [(x, y) for y in range(BOARD_SIZE + 1) for x in range(BOARD_SIZE + 1)],
    dtype=np.float32,
)
BOARD_BOUNDARY_COORDINATES = np.array(
    [[0, 0], [BOARD_SIZE, 0], [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE]],
    dtype=np.float32,
)

# Homography fitting
DEFAULT_INLIER_THRESHOLD = 10
MIN_HOMOGRAPHY_INLIERS = 12
HOMOGRAPHY_METHOD_DIRECT = 0
HOMOGRAPHY_REFINEMENT_ITERATIONS = 3

# Standard chessboard detector
CHESSBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIXEL_MAX_ITERATIONS = 30
SUBPIXEL_EPSILON = 0.001
SUBPIXEL_WINDOW = (5, 5)
SUBPIXEL_ZERO_ZONE = (-1, -1)

# Fallback corner candidate detection
BLUR_KERNEL_SIZE = (5, 5)
MAX_CORNER_CANDIDATES = 700
CORNER_QUALITY_LEVEL = 0.005
CORNER_MIN_DISTANCE = 5
CORNER_BLOCK_SIZE = 5
HARRIS_K = 0.04
CANDIDATE_SUBPIXEL_WINDOW = (4, 4)
CANDIDATE_SUBPIXEL_MAX_ITERATIONS = 20
CANDIDATE_SUBPIXEL_EPSILON = 0.01
MIN_CANDIDATE_POINTS = 16
MAX_SEARCH_POINTS = 220
RANDOM_SEED = 7
MAX_ANCHORS = 100
NEAREST_NEIGHBOUR_SLICE_START = 1
NEAREST_NEIGHBOUR_COUNT = 8
MIN_GRID_STEP_LENGTH = 8
MIN_STEP_LENGTH_RATIO = 0.65
MAX_STEP_LENGTH_RATIO = 1.55
MAX_ABSOLUTE_COS_ANGLE = 0.65
INITIAL_MATCH_MIN_DISTANCE = 6
INITIAL_MATCH_STEP_FRACTION = 0.30
MIN_MATCHED_INTERSECTIONS = 12

# Candidate scoring and geometry validation
SCORE_RESIDUAL_PENALTY = 0.02
MIN_ACCEPTABLE_GRID_SCORE = 20
BOUNDARY_MARGIN_FRACTION = 0.25
BOUNDARY_MAX_EXTENT_FACTOR = 1.25

# Display settings
BOARD_BOUNDARY_COLOR = (0, 0, 255)
SQUARE_CENTRE_COLOR = (0, 255, 0)
BOARD_CORNER_COLOR = (255, 0, 0)
BOARD_BOUNDARY_THICKNESS = 3
SQUARE_CENTRE_RADIUS = 4
BOARD_CORNER_RADIUS = 7
DISPLAY_FIGURE_SIZE = (12, 8)

# Debug display settings (OpenCV uses BGR colour order)
ALL_CANDIDATE_COLOR = (0, 165, 255)
WINNING_GRID_COLOR = (255, 0, 255)
PREDICTED_GRID_COLOR = (255, 255, 0)
DEBUG_CANDIDATE_RADIUS = 4
DEBUG_WINNER_RADIUS = 7
DEBUG_PREDICTED_RADIUS = 5
PREDICTED_GRID_THICKNESS = 1


def load_image(image_path):
    """Load an image and return both its BGR and grayscale versions."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def fit_homography(src, dst, inlier_threshold=DEFAULT_INLIER_THRESHOLD):
    """Fit a RANSAC homography, rejecting fits with too few inliers."""
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, inlier_threshold)
    if homography is None or mask is None:
        return None
    if int(mask.sum()) < MIN_HOMOGRAPHY_INLIERS:
        return None
    return homography


def grid_to_pixel(homography, x, y):
    """Convert one board-grid coordinate to an image pixel coordinate."""
    point = np.array([[[x, y]]], dtype=np.float32)
    return cv2.perspectiveTransform(point, homography)[0, 0]


def detect_standard_chessboard(gray, inlier_threshold):
    """Try OpenCV's built-in detector and return its homography, if successful."""
    found, corners = cv2.findChessboardCorners(
        gray,
        (INNER_CORNERS_PER_SIDE, INNER_CORNERS_PER_SIDE),
        CHESSBOARD_FLAGS,
    )
    if not found:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        SUBPIXEL_MAX_ITERATIONS,
        SUBPIXEL_EPSILON,
    )
    corners = cv2.cornerSubPix(gray, corners, SUBPIXEL_WINDOW, SUBPIXEL_ZERO_ZONE, criteria)
    return fit_homography(GRID_COORDINATES, corners.reshape(-1, 2), inlier_threshold)


def detect_corner_candidates(gray):
    """Detect and subpixel-refine corner candidates for the fallback search."""
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)
    raw = cv2.goodFeaturesToTrack(
        blurred,
        maxCorners=MAX_CORNER_CANDIDATES,
        qualityLevel=CORNER_QUALITY_LEVEL,
        minDistance=CORNER_MIN_DISTANCE,
        blockSize=CORNER_BLOCK_SIZE,
        useHarrisDetector=True,
        k=HARRIS_K,
    )

    if raw is None:
        raise RuntimeError("No corner candidates found.")

    raw = raw.reshape(-1, 1, 2).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        CANDIDATE_SUBPIXEL_MAX_ITERATIONS,
        CANDIDATE_SUBPIXEL_EPSILON,
    )
    cv2.cornerSubPix(gray, raw, CANDIDATE_SUBPIXEL_WINDOW, SUBPIXEL_ZERO_ZONE, criteria)
    points = raw.reshape(-1, 2)

    if len(points) < MIN_CANDIDATE_POINTS:
        raise RuntimeError("Too few distinct corner candidates.")

    return points


def limit_search_points(points):
    """Reduce candidate count while keeping candidates spread across the image."""
    if len(points) <= MAX_SEARCH_POINTS:
        return points

    selected = [int(np.argmax(points[:, 0] + points[:, 1]))]
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)

    while len(selected) < MAX_SEARCH_POINTS:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[idx], axis=1))
        min_dist[selected] = -1

    return points[selected]


def greedy_match_grid_points(predicted, points, match_threshold):
    """
    Match predicted grid intersections to observed candidates one-to-one.
    Returns matched grid indices and candidate-point indices.
    """
    distances = np.linalg.norm(predicted[:, None, :] - points[None, :, :], axis=2)
    nearest_ids = distances.argmin(axis=1)
    nearest_dist = distances.min(axis=1)

    pairs = sorted(
        [
            (float(nearest_dist[k]), k, int(nearest_ids[k]))
            for k in range(len(predicted))
            if nearest_dist[k] < match_threshold
        ]
    )

    used_grid = set()
    used_points = set()
    matched_grid_ids = []
    matched_point_ids = []

    for _, grid_id, point_id in pairs:
        if grid_id not in used_grid and point_id not in used_points:
            used_grid.add(grid_id)
            used_points.add(point_id)
            matched_grid_ids.append(grid_id)
            matched_point_ids.append(point_id)

    return matched_grid_ids, matched_point_ids


def validate_projected_boundary(homography, gray_shape):
    """Reject projected board boundaries that are non-finite or far out of frame."""
    outer = cv2.perspectiveTransform(
        BOARD_BOUNDARY_COORDINATES.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)

    if not np.isfinite(outer).all():
        return False

    height, width = gray_shape
    if np.any(outer[:, 0] < -BOUNDARY_MARGIN_FRACTION * width) or np.any(
        outer[:, 0] > BOUNDARY_MAX_EXTENT_FACTOR * width
    ):
        return False
    if np.any(outer[:, 1] < -BOUNDARY_MARGIN_FRACTION * height) or np.any(
        outer[:, 1] > BOUNDARY_MAX_EXTENT_FACTOR * height
    ):
        return False

    return True


def apply_grid_offset(homography, offset_x, offset_y):
    """Shift the projected grid by integer grid steps in board coordinates."""
    translation = np.array(
        [[1.0, 0.0, float(offset_x)], [0.0, 1.0, float(offset_y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return homography @ translation


def find_best_grid_offset(homography, points, gray_shape, inlier_threshold):
    """Test the winning grid and its eight neighbouring one-cell displacements."""
    best_homography = homography
    best_score = score_homography(homography, points, inlier_threshold)
    best_offset = (0, 0)

    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            if offset_x == 0 and offset_y == 0:
                continue

            candidate = apply_grid_offset(homography, offset_x, offset_y)
            candidate = refine_grid_homography(candidate, points, inlier_threshold)
            if candidate is None or not validate_projected_boundary(candidate, gray_shape):
                continue

            score = score_homography(candidate, points, inlier_threshold)
            # print(f"Grid offset ({offset_x:+d}, {offset_y:+d}): score={score:.2f}")

            if score > best_score:
                best_score = score
                best_homography = candidate
                best_offset = (offset_x, offset_y)

    print(f"Selected grid offset: ({best_offset[0]:+d}, {best_offset[1]:+d}), score={best_score:.2f}")
    return best_homography


def score_homography(homography, points, inlier_threshold):
    """Score a grid while penalising unmatched intersections and incomplete lines."""
    projected = cv2.perspectiveTransform(
        EXPANDED_GRID_COORDINATES.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)

    matched_grid_ids, matched_point_ids = greedy_match_grid_points(
        projected, points, inlier_threshold
    )
    matched_count = len(matched_grid_ids)

    if matched_count:
        matched_points = points[matched_point_ids]
        residuals = np.linalg.norm(projected[matched_grid_ids] - matched_points, axis=1)
        residual_penalty = SCORE_RESIDUAL_PENALTY * float(residuals.sum())
    else:
        residual_penalty = 0.0


    return (
        matched_count
        - residual_penalty
    )


def refine_grid_homography(candidate_homography, points, inlier_threshold):
    """Repeatedly match the expanded grid, including its perimeter, and refit the transform."""
    matched_grid_ids = []
    matched_point_ids = []

    for _ in range(HOMOGRAPHY_REFINEMENT_ITERATIONS):
        projected = cv2.perspectiveTransform(
            EXPANDED_GRID_COORDINATES.reshape(-1, 1, 2), candidate_homography
        ).reshape(-1, 2)

        matched_grid_ids, matched_point_ids = greedy_match_grid_points(
            projected, points, inlier_threshold
        )

        if len(matched_grid_ids) < MIN_MATCHED_INTERSECTIONS:
            break

        src = np.asarray(EXPANDED_GRID_COORDINATES[matched_grid_ids], dtype=np.float32)
        dst = np.asarray(points[matched_point_ids], dtype=np.float32)
        refined_homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, inlier_threshold)

        if refined_homography is None:
            break

        candidate_homography = refined_homography

    if len(matched_grid_ids) < MIN_MATCHED_INTERSECTIONS:
        return None

    return candidate_homography


def fallback_grid_search(gray, points, inlier_threshold):
    """Search local grid hypotheses and return the highest-scoring homography."""
    points = limit_search_points(points)

    # Precompute distances between all candidate points.
    pair_dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    best_score = -1
    best_homography = None
    best_grid_points = np.empty((0, 2), dtype=np.float32)
    rng = np.random.default_rng(RANDOM_SEED)

    # Use spatially distributed anchor points to start local grids.
    anchor_count = min(len(points), MAX_ANCHORS)
    anchor_ids = rng.choice(len(points), anchor_count, replace=False)

    for anchor_id in anchor_ids:
        anchor = points[anchor_id]
        neighbour_ids = np.argsort(pair_dist[anchor_id])[
            NEAREST_NEIGHBOUR_SLICE_START:
            NEAREST_NEIGHBOUR_SLICE_START + NEAREST_NEIGHBOUR_COUNT
        ]
        neighbours = points[neighbour_ids]

        for i in range(len(neighbours)):
            u = neighbours[i] - anchor
            length_u = np.linalg.norm(u)
            if length_u < MIN_GRID_STEP_LENGTH:
                continue

            for j in range(i + 1, len(neighbours)):
                v = neighbours[j] - anchor
                length_v = np.linalg.norm(v)

                if length_v < MIN_GRID_STEP_LENGTH or not MIN_STEP_LENGTH_RATIO < length_u / length_v < MAX_STEP_LENGTH_RATIO:
                    continue

                cos_angle = abs(np.dot(u, v) / (length_u * length_v))
                if cos_angle > MAX_ABSOLUTE_COS_ANGLE:
                    continue

                # Local affine hypothesis for the expanded grid, anchored at grid (4,4).
                predicted = anchor + (
                    EXPANDED_GRID_COORDINATES - GRID_ANCHOR_COORDINATE
                ) @ np.column_stack((u, v)).T

                initial_match_threshold = max(
                    INITIAL_MATCH_MIN_DISTANCE,
                    INITIAL_MATCH_STEP_FRACTION * (length_u + length_v) / 2,
                )
                matched_grid_ids, matched_point_ids = greedy_match_grid_points(
                    predicted, points, initial_match_threshold
                )

                if len(matched_grid_ids) < MIN_MATCHED_INTERSECTIONS:
                    continue

                src = np.asarray(EXPANDED_GRID_COORDINATES[matched_grid_ids], dtype=np.float32)
                dst = np.asarray(points[matched_point_ids], dtype=np.float32)

                # Upgrade the affine hypothesis to a projective fit.
                candidate_homography, _ = cv2.findHomography(
                    src, dst, HOMOGRAPHY_METHOD_DIRECT
                )
                if candidate_homography is None:
                    continue

                candidate_homography = refine_grid_homography(
                    candidate_homography, points, inlier_threshold
                )
                if candidate_homography is None:
                    continue

                score = score_homography(candidate_homography, points, inlier_threshold)

                if not validate_projected_boundary(candidate_homography, gray.shape):
                    continue

                if score > best_score:
                    best_score = score
                    best_homography = candidate_homography

                    # Keep the observed candidates that match the winning grid.
                    projected = cv2.perspectiveTransform(
                        EXPANDED_GRID_COORDINATES.reshape(-1, 1, 2), candidate_homography
                    ).reshape(-1, 2)
                    _, winning_point_ids = greedy_match_grid_points(
                        projected, points, inlier_threshold
                    )
                    best_grid_points = points[winning_point_ids].copy()

    if best_homography is None or best_score < MIN_ACCEPTABLE_GRID_SCORE:
        raise RuntimeError(
            "Could not fit a reliable chessboard grid. "
            "Try increasing candidate detection sensitivity or checking image quality."
        )

    # The best-scoring grid may still be displaced by one complete square.
    # Test all neighbouring offsets before returning the final homography.
    best_homography = find_best_grid_offset(
        best_homography, points, gray.shape, inlier_threshold
    )

    # Refresh the matched-corner set for the selected offset.
    projected = cv2.perspectiveTransform(
        EXPANDED_GRID_COORDINATES.reshape(-1, 1, 2), best_homography
    ).reshape(-1, 2)
    _, winning_point_ids = greedy_match_grid_points(projected, points, inlier_threshold)
    best_grid_points = points[winning_point_ids].copy()

    return best_homography, best_grid_points


def calculate_board_coordinates(homography):
    """Calculate the four outer corners and all 64 square centres."""
    board_corners = np.array(
        [
            grid_to_pixel(homography, 0, 0),
            grid_to_pixel(homography, BOARD_SIZE, 0),
            grid_to_pixel(homography, BOARD_SIZE, BOARD_SIZE),
            grid_to_pixel(homography, 0, BOARD_SIZE),
        ],
        dtype=np.float32,
    )

    square_centres = np.array(
        [
            [grid_to_pixel(homography, col + 0.5, row + 0.5) for col in range(BOARD_SIZE)]
            for row in range(BOARD_SIZE)
        ],
        dtype=np.float32,
    )

    return square_centres, board_corners


def display_detection(image, square_centres, board_corners):
    """Draw the detected board boundary, square centres, and outer corners."""
    display = image.copy()
    cv2.polylines(
        display,
        [np.round(board_corners).astype(np.int32)],
        True,
        BOARD_BOUNDARY_COLOR,
        BOARD_BOUNDARY_THICKNESS,
    )

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x, y = np.round(square_centres[row, col]).astype(int)
            cv2.circle(display, (x, y), SQUARE_CENTRE_RADIUS, SQUARE_CENTRE_COLOR, -1)

    for x, y in np.round(board_corners).astype(int):
        cv2.circle(display, (x, y), BOARD_CORNER_RADIUS, BOARD_CORNER_COLOR, -1)

    plt.figure(figsize=DISPLAY_FIGURE_SIZE)
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.title("Chessboard corners and square centres")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def display_corner_debug(image, all_candidates, winning_grid_points, homography):
    """Show candidates, matched corners, and all homography-predicted grid intersections."""
    display = image.copy()
    height, width = image.shape[:2]
    projected = cv2.perspectiveTransform(
        EXPANDED_GRID_COORDINATES.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    projected_grid = projected.reshape(BOARD_SIZE + 1, BOARD_SIZE + 1, 2)

    # Draw the predicted 9x9 lattice first so detections remain visible over it.
    for row in range(BOARD_SIZE + 1):
        line = np.round(projected_grid[row]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(display, [line], False, PREDICTED_GRID_COLOR, PREDICTED_GRID_THICKNESS)
    for col in range(BOARD_SIZE + 1):
        line = np.round(projected_grid[:, col]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(display, [line], False, PREDICTED_GRID_COLOR, PREDICTED_GRID_THICKNESS)

    # Draw all detected corner candidates.
    for x, y in np.round(all_candidates).astype(int):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(display, (x, y), DEBUG_CANDIDATE_RADIUS, ALL_CANDIDATE_COLOR, -1)

    # Mark every predicted intersection, including locations with no detected corner.
    for x, y in np.round(projected).astype(int):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(display, (x, y), DEBUG_PREDICTED_RADIUS, PREDICTED_GRID_COLOR, 1)

    # Highlight observed candidates assigned to the winning grid.
    for x, y in np.round(winning_grid_points).astype(int):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(display, (x, y), DEBUG_WINNER_RADIUS, WINNING_GRID_COLOR, 2)

    plt.figure(figsize=DISPLAY_FIGURE_SIZE)
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.title("Debug: predicted grid (cyan), candidates (orange), matched points (magenta)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def detect_chessboard(image_path, display_output=False, inlier_threshold=DEFAULT_INLIER_THRESHOLD, display_debug=False):
    """
    Detect a chessboard and return its square centres and outer corners.

    Args:
        image_path: Path to the image.
        display_output: If True, display the detected board overlay.
        inlier_threshold: Pixel-distance threshold used for robust homography fitting.
        display_debug: If True, display all corner candidates and highlight the winning grid.

    Returns:
        square_centres: Array of shape (8, 8, 2), with each point as (x, y).
        board_corners: Array of shape (4, 2), ordered TL, TR, BR, BL.
    """
    image, gray = load_image(image_path)

    # Fast path: use OpenCV's standard chessboard detector.
    homography = detect_standard_chessboard(gray, inlier_threshold)
    all_candidates = None
    winning_grid_points = None

    # Fallback: estimate an expanded 9x9 grid, including the board perimeter,
    # from individual corner candidates.
    if homography is None:
        all_candidates = detect_corner_candidates(gray)
        homography, winning_grid_points = fallback_grid_search(
            gray, all_candidates, inlier_threshold
        )
    elif display_debug:
        # On the standard path, show all candidates too, while using OpenCV's
        # detected internal intersections as the winning-grid points.
        all_candidates = detect_corner_candidates(gray)
        found, standard_corners = cv2.findChessboardCorners(
            gray, (INNER_CORNERS_PER_SIDE, INNER_CORNERS_PER_SIDE), CHESSBOARD_FLAGS
        )
        if found:
            winning_grid_points = standard_corners.reshape(-1, 2)
        else:
            winning_grid_points = np.empty((0, 2), dtype=np.float32)

    square_centres, board_corners = calculate_board_coordinates(homography)

    if display_output:
        display_detection(image, square_centres, board_corners)

    if display_debug:
        if all_candidates is None:
            all_candidates = detect_corner_candidates(gray)
        if winning_grid_points is None:
            projected = cv2.perspectiveTransform(
                EXPANDED_GRID_COORDINATES.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            _, winning_ids = greedy_match_grid_points(
                projected, all_candidates, inlier_threshold
            )
            winning_grid_points = all_candidates[winning_ids]
        display_corner_debug(image, all_candidates, winning_grid_points, homography)

    return square_centres, board_corners


if __name__ == "__main__":
    # Note image #5 fails but the image is pretty unreasonable
    centres, corners = detect_chessboard(r"occluded-photos\9.jpg", display_output=True, display_debug=False)
