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

calib = CameraIntrinsics(**json.load(open(folder / 'camera_intrinsics.json')))

rvecs, tvecs = np.array(calib.rvecs)[..., 0], np.array(calib.tvecs)[..., 0]

base_t_tcps = [Transform.from_xyz_rotvec(d) for d in np.load(folder / 'image_tcp_poses.npy')[calib.img_ids]]
cam_t_boards = [Transform(p=tvec, rotvec=rvec) for rvec, tvec in zip(rvecs, tvecs)]

n = len(cam_t_boards)
assert n == len(base_t_tcps), '{}, {}'.format(n, len(base_t_tcps))
A, B = [], []

shuffle_idx = np.arange(n)
for j in range(10):
    np.random.RandomState(seed=j).shuffle(shuffle_idx)
    for i in range(1, n):
        a, b = shuffle_idx[i], shuffle_idx[i - 1]
        A.append((base_t_tcps[a].inv @ base_t_tcps[b]).matrix)
        B.append((cam_t_boards[a] @ cam_t_boards[b].inv).matrix)

R, t = park_martin.calibrate(A, B)
tcp_t_cam = Transform(R=R, p=t)
tcp_t_cam.save(folder / 'tcp_t_cam')

base_t_boards = []
for base_t_tcp, cam_t_board in zip(base_t_tcps, cam_t_boards):
    base_t_boards.append(base_t_tcp @ tcp_t_cam @ cam_t_board)

poses = np.array([t.xyz_rotvec for t in base_t_boards])
std = poses.std(axis=0)
print('translation std [mm]', np.linalg.norm(std[:3]) * 1e3)
print('rotation std [deg]', np.linalg.norm(np.rad2deg(std[3:])))
