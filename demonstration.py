import time
import argparse
from threading import Thread

import numpy as np
from ur_control import Robot

from transform3d import Transform


class EnterPoll(Thread):
    pressed = False

    def __init__(self):
        super().__init__()
        self.start()

    def run(self):
        input()
        self.pressed = True


def capture(robot: Robot, capture_frequency=10.):
    robot.ctrl.teachMode()
    input('Put the robot into the start position and press enter to start capture')
    enter_poll = EnterPoll()
    demonstration = []
    while not enter_poll.pressed:
        demonstration.append((
            robot.recv.getActualQ(),
            robot.recv.getActualTCPPose()
        ))
        time.sleep(1 / capture_frequency)
    print('Ended capture')

    return np.array(demonstration)


def prune(demonstration, min_dist=0.01, min_angle=np.deg2rad(10), min_q_dist=0.01, use_q_space=True):
    idxs = [0]
    for i, (q, pose) in enumerate(demonstration):
        if use_q_space:
            q_last = demonstration[idxs[-1]][0]
            if np.linalg.norm(q_last - q) > min_q_dist:
                idxs.append(i)
        else:
            T_last = Transform.from_xyz_rotvec(demonstration[idxs[-1]][1])
            T = Transform.from_xyz_rotvec(pose)
            diff = T @ T_last.inv()
            dist, angle = np.linalg.norm(diff.t), np.linalg.norm(diff.r.as_rotvec())
            if dist > min_dist or angle > min_angle:
                idxs.append(i)
    return np.array(idxs)


def farthest_point_sampling(demonstration: np.ndarray, n: int, use_q_space=False):
    N = demonstration.shape[0]
    assert N >= n, 'not enough points in original demonstration'
    if use_q_space:
        t = demonstration[:, 0]  # [N, 6]
    else:
        t = demonstration[:, 1, :3]  # [N, 3]
    dists = np.linalg.norm(t[None] - t[:, None], axis=-1)  # [N, N]
    chosen = {0}
    others = set(range(N)) - chosen
    for _ in range(n - 1):
        others_l = list(others)
        d = dists[others_l][:, list(chosen)].min(axis=1)
        farthest_other = others_l[np.argmax(d)]
        chosen.add(farthest_other)
        others.remove(farthest_other)
    return np.sort(list(chosen))


def replay(demonstration: np.ndarray, robot: Robot,
           velocity=1.0, acceleration=1.0, blend=.02):
    qs = demonstration[:, 0]
    path = np.concatenate((
        qs,
        np.tile([[velocity, acceleration, blend]], (len(qs), 1))
    ), axis=1)
    path[(0, -1), -1] = 0  # zero blend at end-points
    path = path.tolist()
    robot.ctrl.endTeachMode()
    robot.ctrl.moveJ(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ur-ip', required=True)
    parser.add_argument('--file', required=True)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--backwards', action='store_true')
    args = parser.parse_args()

    cam = args.cam
    robot = Robot.from_ip(args.ur_ip)

    if args.replay:
        demonstration = np.load('data/demonstration_{}.npy'.format(cam))
        if args.backwards:
            demonstration = demonstration[::-1]
        print(len(demonstration), 'samples in original')
        prune_idxs = prune(demonstration)
        pruned = demonstration[prune_idxs]
        print(len(pruned), 'samples after pruning')
        # fps_idxs = prune_idxs[farthest_point_sampling(pruned, n=50)]
        # fps = demonstration[fps_idxs]
        # print(len(fps), 'samples after fps')
        input('replay?')
        replay(pruned, robot)
    else:
        demonstration = capture(robot)
        np.save('data/demonstration_{}.npy'.format(cam), demonstration)


if __name__ == '__main__':
    main()
