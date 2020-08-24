import time
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from ur_control import Robot

import rospy
from sensor_msgs.msg import Image
from ros_numpy.image import image_to_numpy

import demonstration as demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ur-ip', required=True)
    parser.add_argument('--image-topic', required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--demonstration', required=True)
    parser.add_argument('--backwards', action='store_true')

    rospy.init_node('image-capture-node', anonymous=True)

    args = parser.parse_args()

    folder = Path(args.folder)
    folder.mkdir()

    image_folder = folder / 'calibration-images'
    image_folder.mkdir()

    robot = Robot.from_ip(args.ur_ip)

    demonstration = np.load(args.demonstration)
    if args.backwards:
        demonstration = demonstration[::-1]

    print('demonstration length:', len(demonstration))
    prune_idxs = demo.prune(demonstration, min_q_dist=0.02)
    print('pruned length:', len(prune_idxs))
    pruned = demonstration[prune_idxs]
    fps_idxs = demo.farthest_point_sampling(pruned, n=100)

    cur_idx = 0
    poses = []
    for i, fps_idx in enumerate(tqdm(fps_idxs)):
        assert fps_idx >= cur_idx
        # subpath = pruned[cur_idx:fps_idx + 1]
        # demo.replay(subpath, rtde_c, blend=0.01)  # TODO: blending is still shaky?

        des_q = pruned[fps_idx][0]
        robot.ctrl.moveJ(des_q)

        now = time.time()
        while True:
            img = rospy.wait_for_message(args.image_topic, Image)  # type: Image
            if img.header.stamp.to_sec() > now:
                break
        img = image_to_numpy(img)
        img_path = image_folder / '{}.png'.format(i)
        cv2.imwrite(str(img_path), img[..., ::-1])

        poses.append(robot.recv.getActualTCPPose())

        cur_idx = fps_idx
    np.save(folder / 'image_tcp_poses.npy', poses)


if __name__ == '__main__':
    main()
