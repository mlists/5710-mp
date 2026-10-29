from pathlib import Path

import cv2
import numpy as np
import find_board
import get_pose

PROJECT_DIR = Path(__file__).resolve().parent
# IMAGE_PATH = PROJECT_DIR / 'train-photos' / '3.jpg'
IMAGE_PATH = PROJECT_DIR / 'occluded-photos' / '2.jpg'


def main():
    for img in (PROJECT_DIR / 'occluded-photos').glob('*.jpg'):
        frame = cv2.imread(str(img))
        if frame is None:
            raise FileNotFoundError(f'Could not load image: {img}')

        # Detector returns directly observed 7x7 internal intersections
        centres_px, corners_px, inner_px = find_board.detect_chessboard(frame)
        print('Measured corners:', int(np.isfinite(inner_px).all(axis=2).sum()), '/ 49')

        # Estimate 3D pose using points
        pose = get_pose.get_board_pose(corners_px, centres_px, observed_image_points=inner_px, square_size=1.0)

        print('Fit points:', pose['num_fit_points'])
        print('Actual observed intersections provided:', pose['num_observations'])
        print('Fit reprojection RMSE (pixels):', f"{pose['reprojection_rmse']:.3f}")
        print('R (board to camera):\n', pose['R'])
        print('tvec (board units):', pose['tvec'].ravel())
        print('Board normal (camera coordinates):', pose['board_normal'])

        overlay = get_pose.draw_pose_overlay(frame, pose)
        output_path = PROJECT_DIR / 'pose-estimations' / 'pose_overlay' f'_{img.stem}.jpg'
        if not cv2.imwrite(str(output_path), overlay):
            raise OSError(f'Could not save pose overlay to {output_path}')
        print('Saved overlay:', output_path)


if __name__ == '__main__':
    main()
