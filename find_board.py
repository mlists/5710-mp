
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_chessboard(image_path, display_output=False):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the 7x7 internal intersections of an 8x8 chessboard
    pattern_size = (7, 7)

    
    # Standard detector with adaptive thresholding and normalization
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if found:
        # Refine the corner coordinates to subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    else:
        raise RuntimeError("Chessboard not detected.")

    # Convert detected corners to an array of shape (7, 7, 2)
    inner = corners.reshape(7, 7, 2)

    # Assign ideal grid coordinates to the detected intersections.
    # The board spans x = 0..8 and y = 0..8 in grid coordinates.
    grid_points = np.array(
        [(x, y) for y in range(1, 8) for x in range(1, 8)],
        dtype=np.float32
    )

    image_points = inner.reshape(-1, 2).astype(np.float32)

    # Find the perspective transformation from grid coordinates
    # to image pixel coordinates.
    H, _ = cv2.findHomography(grid_points, image_points, 0)

    if H is None:
        raise RuntimeError("Could not calculate board geometry.")

    # Helper function to convert chess board grid coordinates into pixel coordinates.
    def grid_to_pixel(x, y):
        point = np.array([[[x, y]]], dtype=np.float32)
        return cv2.perspectiveTransform(point, H)[0, 0]

    # Calculate the four outer board corners.
    # Order: top-left, top-right, bottom-right, bottom-left.
    board_corners = np.array([
        grid_to_pixel(0, 0),
        grid_to_pixel(8, 0),
        grid_to_pixel(8, 8),
        grid_to_pixel(0, 8)
    ])

    # Calculate the centres of all 64 squares.
    # Row 0 is the top row; column 0 is the left column.
    square_centres = np.zeros((8, 8, 2), dtype=np.float32)

    for row in range(8):
        for col in range(8):
            square_centres[row, col] = grid_to_pixel(
                col + 0.5, row + 0.5
            )

    if display_output:
        # Print pixel coordinates as (x, y).
        print("Board corners (x, y):")
        for name, point in zip(
            ["Top-left", "Top-right", "Bottom-right", "Bottom-left"],
            board_corners
        ):
            print(f"{name}: ({point[0]:.2f}, {point[1]:.2f})")

        print("\nSquare centres (x, y):")
        for row in range(8):
            for col in range(8):
                x, y = square_centres[row, col]
                print(f"Square [{row}, {col}]: ({x:.2f}, {y:.2f})")

        # Visualise the detected points and board outline.
        display = image.copy()

        polygon = np.round(board_corners).astype(np.int32)
        cv2.polylines(display, [polygon], True, (0, 0, 255), 3)

        for row in range(8):
            for col in range(8):
                x, y = np.round(square_centres[row, col]).astype(int)
                cv2.circle(display, (x, y), 4, (0, 255, 0), -1)

        for x, y in np.round(board_corners).astype(int):
            cv2.circle(display, (x, y), 7, (255, 0, 0), -1)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        plt.title("Chessboard corners and square centres")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return square_centres, board_corners


if __name__ == "__main__":
    centres, corners = detect_chessboard("5710-mp\\train-photos\\3.jpg")
