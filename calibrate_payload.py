import time
from typing import List
from transform3d import Transform
import numpy as np
from ur_control import Robot


def get_force_data_point(base_t_tcp: Transform, force_tcp):
    # force_tcp = tcp_force_zero + tcp_R_base @ base_fg
    tcp_t_base = base_t_tcp.inv
    A = np.concatenate((
        np.eye(3), tcp_t_base.R
    ), axis=-1)
    # x: tcp_force_zero (3), base_fg (3)
    b = force_tcp
    return A, b


def get_torque_datapoint(base_t_tcp: Transform, torque_tcp, base_fg):
    # torque_tcp = tcp_torque_zero + np.cross(tcp_p_cog, tcp_fg)
    tcp_t_base = base_t_tcp.inv
    tcp_fg = tcp_t_base.rotate(base_fg)
    A = np.concatenate((
        np.eye(3), np.cross(tcp_fg, np.eye(3))
    ), axis=-1)
    # x: tcp_torque_zero (3), tcp_p_cog (3)
    b = torque_tcp
    return A, b


def calibrate_payload(base_t_tcps: List[Transform], ft_tcps):
    ds = [get_force_data_point(base_t_tcp, ft_tcp[:3]) for base_t_tcp, ft_tcp in zip(base_t_tcps, ft_tcps)]
    A = np.concatenate([d[0] for d in ds])
    b = np.concatenate([d[1] for d in ds])
    x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    tcp_force_zero, base_fg = x.reshape(2, 3)
    print('tcp_force_zero:', tcp_force_zero, 'base_fg:', base_fg)
    print('res:', res, 's:', s)
    print('max(abs(res))', np.abs(A @ x - b).max())
    print('mean(abs(res))', np.abs(A @ x - b).mean())
    print('rms', np.sqrt(((A @ x - b) ** 2).mean()))
    print()

    ds = [get_torque_datapoint(base_t_tcp, ft_tcp[3:], base_fg) for base_t_tcp, ft_tcp in zip(base_t_tcps, ft_tcps)]
    A = np.concatenate([d[0] for d in ds])
    b = np.concatenate([d[1] for d in ds])
    x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    tcp_torque_zero, tcp_p_cog = x.reshape(2, 3)
    print('tcp_torque_zero:', tcp_torque_zero, 'tcp_p_cog:', tcp_p_cog)
    print('res:', res, 's:', s)
    print('max(abs(res))', np.abs(A @ x - b).max())
    print('mean(abs(res))', np.abs(A @ x - b).mean())
    print('rms', np.sqrt(((A @ x - b) ** 2).mean()))
    print()

    tool_mass = np.linalg.norm(base_fg) / 9.807

    return tool_mass, tcp_p_cog


def record_qs():
    r = Robot.from_ip('192.168.1.130')
    r.ctrl.setPayload(1.7, (0, -0.008, 0.07))
    r.ctrl.teachMode()
    data = []
    while True:
        if 'n' in input().lower():
            break
        else:
            data.append(r.recv.getActualQ())
    r.ctrl.endTeachMode()

    np.save('qs.npy', data)
    return data


def record_data(name='data.npy', zero_payload=True):
    qs = np.load('qs.npy')
    r = Robot.from_ip('192.168.1.130')
    if zero_payload:
        r.ctrl.setPayload(0, (0, 0, 0))
    r.ctrl.zeroFtSensor()
    data = []
    for q in qs:
        r.ctrl.moveJ(q, 0.25)
        time.sleep(0.25)
        data.append(list(r.base_t_tcp()) + list(r.ft_tcp()))

    np.save(name, data)
    return data


def main():
    # record_data('data.npy', zero_payload=True)
    # quit()

    data = np.load('data.npy')
    base_t_tcps = [Transform.from_xyz_rotvec(d[:6]) for d in data]
    ft_tcps = [d[6:] for d in data]
    tool_mass, tcp_p_cog = calibrate_payload(base_t_tcps, ft_tcps)
    print(tool_mass, tcp_p_cog)
    r = Robot.from_ip('192.168.1.130')
    r.ctrl.setPayload(tool_mass, tcp_p_cog)

    data_after = np.load('data.npy')
    print('force max', np.abs(data_after[:, 6:9]).max())
    print('force mean', np.abs(data_after[:, 6:9]).mean())
    print('force rms', np.sqrt((data_after[:, 6:9] ** 2).mean()))
    print()
    print('torque max', np.abs(data_after[:, 9:]).max())
    print('torque mean', np.abs(data_after[:, 9:]).mean())


if __name__ == '__main__':
    main()
