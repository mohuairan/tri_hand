"""
visualize_pinch_3f.py - 三指版 MuJoCo 捏合轨迹可视化

加载三指版 pinch_trajectories_3f 生成的轨迹，在 MuJoCo 中播放动画。
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pinch_trajectories_3f import PinchTrajectoryGenerator

_POLY = np.array([-0.01462, 1.27107, 0.07658, 0.05314, 0.10674])


def poly_q4(q3_rad):
    return _POLY[0] + _POLY[1]*q3_rad + _POLY[2]*q3_rad**2 + _POLY[3]*q3_rad**3 + _POLY[4]*q3_rad**4


def apply_trajectory_frame(model, data, thumb_q_deg, finger_q_deg, finger_name):
    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    data.qpos[jid('thumb_q1')] = np.radians(thumb_q_deg[0])
    data.qpos[jid('thumb_q2')] = np.radians(thumb_q_deg[1])
    data.qpos[jid('thumb_q3')] = np.radians(thumb_q_deg[2])
    data.qpos[jid('thumb_q4')] = np.radians(thumb_q_deg[3])

    q3_rad = np.radians(finger_q_deg[2])
    data.qpos[jid(f'{finger_name}_q1')] = np.radians(finger_q_deg[0])
    data.qpos[jid(f'{finger_name}_q2')] = np.radians(finger_q_deg[1])
    data.qpos[jid(f'{finger_name}_q3')] = q3_rad
    data.qpos[jid(f'{finger_name}_q4')] = poly_q4(q3_rad)

    mujoco.mj_forward(model, data)


def main():
    print("生成三指版捏合轨迹...")
    gen = PinchTrajectoryGenerator()

    n_pts = 60
    result_index = gen.get_thumb_index_pinch(n_points=n_pts)
    result_middle = gen.get_thumb_middle_pinch(n_points=n_pts)

    if not result_index['success'] or not result_middle['success']:
        print("轨迹生成失败")
        return

    xml_path = os.path.join(os.path.dirname(__file__), 'mujoco_model', 'jack_hand_3f.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("\n启动 MuJoCo 可视化 (关闭窗口退出)")
    print("  Phase 1: 拇指-食指捏合")
    print("  Phase 2: 张开")
    print("  Phase 3: 拇指-中指捏合")
    print("  Phase 4: 张开")
    print("  循环播放")

    dt_frame = 0.04

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for i in range(n_pts):
                if not viewer.is_running():
                    return
                apply_trajectory_frame(model, data,
                                       result_index['thumb_traj'][i],
                                       result_index['finger_traj'][i],
                                       'index')
                viewer.sync()
                time.sleep(dt_frame)

            time.sleep(0.5)

            for i in range(n_pts - 1, -1, -1):
                if not viewer.is_running():
                    return
                apply_trajectory_frame(model, data,
                                       result_index['thumb_traj'][i],
                                       result_index['finger_traj'][i],
                                       'index')
                viewer.sync()
                time.sleep(dt_frame)

            data.qpos[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.5)

            for i in range(n_pts):
                if not viewer.is_running():
                    return
                apply_trajectory_frame(model, data,
                                       result_middle['thumb_traj'][i],
                                       result_middle['finger_traj'][i],
                                       'middle')
                viewer.sync()
                time.sleep(dt_frame)

            time.sleep(0.5)

            for i in range(n_pts - 1, -1, -1):
                if not viewer.is_running():
                    return
                apply_trajectory_frame(model, data,
                                       result_middle['thumb_traj'][i],
                                       result_middle['finger_traj'][i],
                                       'middle')
                viewer.sync()
                time.sleep(dt_frame)

            data.qpos[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0)


if __name__ == '__main__':
    main()
