import argparse
from pathlib import Path
from collections import namedtuple
import json

import numpy as np
import cv2
from tqdm import tqdm
from natsort import natsorted

_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

CameraIntrinsics = namedtuple('CalibrationResult', (
    'img_ids', 'reprojection_error', 'camera_matrix', 'dist_coeffs', 'rvecs', 'tvecs',
    'std_deviations_intrinsics', 'std_deviations_extrinsics', 'per_view_errors'
))

CharucoDetectionResult = namedtuple('CharucoDetectionResult', ('img_ids', 'all_charuco_corners', 'all_charuco_ids'))


def calibrate(img_ids, charuco_corners, charuco_ids, board, h, w):
    return CameraIntrinsics(img_ids, *cv2.aruco.calibrateCameraCharucoExtended(
        list(charuco_corners), list(charuco_ids), board, (w, h), None, None
    ))


def create_charuco_board(squares_x: int, squares_y: int, square_length: float,
                         marker_length: float, dictionary=cv2.aruco.DICT_4X4_100):
    return cv2.aruco.CharucoBoard_create(
        squares_x, squares_y, square_length, marker_length, get_dict(dictionary)
    )


def get_dict(aruco_dict):
    return cv2.aruco.getPredefinedDictionary(aruco_dict)


def detect_charuco_corners(gray_images, board, criteria=_criteria, log_progress=True):
    img_ids = []
    all_charuco_corners = []
    all_charuco_ids = []
    iterator = tqdm(gray_images, desc='detecting markers') if log_progress else gray_images

    for i, gray_img in enumerate(iterator):
        marker_corners, marker_ids, rejected_img_points = cv2.aruco.detectMarkers(gray_img, board.dictionary)
        for corner in marker_corners:
            cv2.cornerSubPix(gray_img, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray_img, board
        )

        if charuco_corners is None or len(charuco_corners) < 3:
            print('Not enough markers detected. Skipping frame..', i)
            continue

        w = board.getChessboardSize()[0] - 1  # one less corner than squares
        rows = set([i[0] // w for i in charuco_ids])
        cols = set([i[0] % w for i in charuco_ids])
        if len(rows) == 1 or len(cols) == 1:
            print('Only one row / column represented. Skipping frame..', i)
            continue

        img_ids.append(i)
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

    return CharucoDetectionResult(np.array(img_ids), np.array(all_charuco_corners), np.array(all_charuco_ids))


def draw_markers(img, marker_corners, marker_ids, charuco_corners, charuco_ids):
    cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)


def calc_whisker(per_view_errors):
    sorted_per_view_error = sorted(per_view_errors)
    uqp = int(len(sorted_per_view_error) * 3 / 4)
    lqp = int(len(sorted_per_view_error) / 4)
    mp = int(len(sorted_per_view_error) / 2)
    IQR = sorted_per_view_error[uqp] - sorted_per_view_error[lqp]
    mean = sorted_per_view_error[mp]
    return mean + 1.5 * IQR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    folder = Path(args.folder)
    image_folder = folder / 'calibration-images'

    img_paths = natsorted([str(fp) for fp in image_folder.glob('*.png')])
    imgs_gray = [cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE) for fp in tqdm(img_paths, desc='loading images')]
    h, w = imgs_gray[0].shape[:2]
    board = create_charuco_board(14, 9, 0.02, 0.015)
    d = detect_charuco_corners(imgs_gray, board)

    if args.debug:
        for img_id, corners, ids in zip(d.img_ids, d.all_charuco_corners, d.all_charuco_ids):
            img = imgs_gray[img_id]
            cv2.aruco.drawDetectedCornersCharuco(img, corners, ids, (0, 0, 255))
            cv2.imshow("", img)
            cv2.waitKey()
        quit()

    print('starting calibration...')
    calib = calibrate(None, d.all_charuco_corners, d.all_charuco_ids, board, h, w)
    print('did first calibration', calib.reprojection_error)

    wb = calc_whisker(calib.per_view_errors)
    wb_mask = np.argwhere(calib.per_view_errors < wb)[:, 0]
    print('starting re-calibration...')
    calib = calibrate(d.img_ids[wb_mask], d.all_charuco_corners[wb_mask], d.all_charuco_ids[wb_mask], board, h, w)
    print('recalibrated', calib.reprojection_error)

    def to_list(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif hasattr(val, '__len__'):
            return [to_list(c) for c in val]
        return val

    calib_dict = {key: to_list(val) for key, val in calib._asdict().items()}
    json.dump(calib_dict, open(folder / 'camera_intrinsics.json', 'w'), indent=4)


if __name__ == '__main__':
    main()
