import time
import argparse
from typing import List

import numpy as np
from ur_control import Robot

from transform3d import Transform

pose_elements = 'x', 'y', 'z', 'rx', 'ry', 'rz'
tcp_t_tool = Transform(rpy=(np.pi, 0, np.pi / 2))  # TODO: translation?


def kabsch(P, Q):  # P, Q: [N, d]
    assert P.shape == Q.shape, '{}, {}'.format(P.shape, Q.shape)
    d = P.shape[1]
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    P, Q = P - Pc, Q - Qc
    H = P.T @ Q
    u, _, vt = np.linalg.svd(H, full_matrices=False)
    s = np.eye(d)
    s[-1, -1] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ s @ u.T
    t = Qc - R @ Pc
    return R, t


def settle(robot: Robot, t=1., damping_sequence=(0.05, 0.1)):
    t = t / len(damping_sequence)
    for damping in damping_sequence:
        robot.ctrl.forceModeSetDamping(damping)
        time.sleep(t)


def center(robot: Robot, axis, ft=None, t=2.):
    base_t_tcp = robot.base_t_tcp()
    limits = np.ones(6)

    axis_idx = pose_elements.index(axis)
    rot = axis_idx > 2
    if ft is None:
        ft = 2 if rot else 15
    if rot:
        selection = np.zeros(3, int)
        selection[axis_idx - 3] = 1
        selection = (*(1 - selection), *selection)
    else:
        selection = np.zeros(6, int)
        selection[axis_idx] = 1
    wrench = np.zeros(6)
    wrench[axis_idx] = ft

    transforms = []  # type: List[Transform]
    for m in 1, -1:
        robot.ctrl.forceMode(base_t_tcp, selection, wrench * m, 2, limits)
        settle(robot, t=t / 2)
        transforms.append(robot.base_t_tcp())
    robot.ctrl.forceModeStop()
    a, b = transforms

    transform_center = Transform.lerp(a, b, 0.5)
    robot.ctrl.moveL(transform_center)
    return transform_center


def get_stable_table_pose(robot: Robot, force_down=25, force_side=15, torque=2):
    # align with table (z, rx, ry)
    robot.ctrl.forceMode(
        robot.base_t_tcp(), (1, 1, 1, 1, 1, 1),
        (0, 0, force_down, 0, 0, 0), 2, np.ones(6)
    )
    settle(robot, t=2)

    center(robot, 'rz', ft=torque)
    for axis in 'xy':
        center(robot, axis, ft=force_side)

    base_t_tcp = robot.base_t_tcp()
    robot.ctrl.teachMode()
    return base_t_tcp


def calibrate_from_table_poses(table_poses, start_offset, grid_size, tcp_tool_offset):
    table_pts_in_base_frame = np.array([base_t_tcp @ (0, 0, tcp_tool_offset) for base_t_tcp in table_poses])

    # estimate table_t_base from the first table pose
    base_t_tcp = table_poses[0]
    table_offset_x, table_offset_y = (np.array(start_offset) + .5) * grid_size
    base_t_table = table_poses[0] @ tcp_t_tool @ Transform(p=(-table_offset_x, -table_offset_y, -tcp_tool_offset))
    table_t_base = base_t_table.inv

    table_pts_in_table_frame = np.array([table_t_base @ base_t_tcp @ (0, 0, tcp_tool_offset)
                                         for base_t_tcp in table_poses])

    # round to nearest possible table positions
    table_pts_in_table_frame[:, 2] = 0
    table_pts_in_table_frame[:, :2] = np.round(
        ((table_pts_in_table_frame[:, :2] - grid_size / 2) / grid_size)) * grid_size + grid_size / 2

    R, t = kabsch(table_pts_in_base_frame, table_pts_in_table_frame)
    table_t_base = Transform(R=R, p=t)
    return table_t_base


def collect_table_poses(robot: Robot, n=3):
    assert n >= 3, 'You need 3 or more samples'
    table_poses = []
    robot.ctrl.teachMode()
    input('Insert the calibration tool close to the table origin with aligned tool axes')
    robot.ctrl.endTeachMode()
    table_poses.append(get_stable_table_pose(robot))
    start_offset = [
        int(input('Table hole index (zero-indexed) ({})? '.format(axis)))
        for axis in 'xy'
    ]
    for i in range(1, n):
        input('Insert the calibration tool at another position. '
              '(orientation doesn\'t matter) ({}/{})'.format(i + 1, n))
        table_poses.append(get_stable_table_pose(robot))
    return table_poses, start_offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ur-ip', type=str, required=True)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--grid-size', type=float, default=0.05)
    parser.add_argument('--tcp-tool-offset', type=float, default=0.012)
    args = parser.parse_args()

    robot = Robot.from_ip(args.ur_ip)
    robot.ctrl.zeroFtSensor()

    table_poses, start_offset = collect_table_poses(robot, n=args.n)
    table_t_base = calibrate_from_table_poses(
        table_poses=table_poses,
        start_offset=start_offset,
        grid_size=args.grid_size,
        tcp_tool_offset=args.tcp_tool_offset,
    )
    table_t_base.save('table_t_base')


if __name__ == '__main__':
    main()
