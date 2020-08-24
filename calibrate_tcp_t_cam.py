import json
import argparse
from pathlib import Path

import numpy as np
from transform3d import Transform

from calibrate_cam import CameraIntrinsics
import park_martin

parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True)
args = parser.parse_args()

folder = Path(args.folder)

calib = CameraIntrinsics(**json.load(open(folder / 'camera_calibration.json')))

rvecs, tvecs = np.array(calib.rvecs)[..., 0], np.array(calib.tvecs)[..., 0]

tcp_T_bases = [Transform.from_xyz_rotvec(d) for d in np.load(folder / 'image_tcp_poses.npy')[calib.img_ids]]
cam_T_boards = [Transform(p=tvec, rotvec=rvec).inv() for rvec, tvec in zip(rvecs, tvecs)]

n = len(cam_T_boards)
assert n == len(tcp_T_bases), '{}, {}'.format(n, len(tcp_T_bases))
A, B = [], []

shuffle_idx = np.arange(n)
for j in range(10):
    np.random.RandomState(seed=j).shuffle(shuffle_idx)
    for i in range(1, n):
        a, b = shuffle_idx[i], shuffle_idx[i - 1]
        A.append((tcp_T_bases[a].inv() @ tcp_T_bases[b]).as_matrix())
        B.append((cam_T_boards[a].inv() @ cam_T_boards[b]).as_matrix())

R, t = park_martin.calibrate(A, B)
cam_T_tcp = Transform(R=R, p=t)
cam_T_tcp.save(folder / 'cam_T_tcp')

board_T_bases = []
for tcp_T_base, cam_T_board in zip(tcp_T_bases, cam_T_boards):
    board_T_bases.append(tcp_T_base @ cam_T_tcp @ cam_T_board.inv())

poses = np.array([T.as_xyz_rotvec() for T in board_T_bases])
std = poses.std(axis=0)
print('translation std [m]', np.linalg.norm(std[:3]))
print('rotation std [deg]', np.linalg.norm(np.rad2deg(std[3:])))
