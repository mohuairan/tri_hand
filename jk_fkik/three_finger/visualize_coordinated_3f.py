"""
visualize_coordinated_3f.py - 三指版协同捏合运动 MuJoCo 可视化

展示拇指-食指/拇指-中指在捏合状态下的协同运动。
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from coordinated_pinch_3f import CoordinatedPinchGenerator

_POLY = np.array([-0.01462, 1.27107, 0.07658, 0.05314, 0.10674])


def poly_q4(q3_rad):
    return (_POLY[0] + _POLY[1] * q3_rad + _POLY[2] * q3_rad ** 2
            + _POLY[3] * q3_rad ** 3 + _POLY[4] * q3_rad ** 4)


def apply_frame(model, data, thumb_q_deg, finger_q_deg, finger_name):
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


def play_phase(viewer, model, data, thumb_traj, finger_traj, finger_name, dt):
    for i in range(len(thumb_traj)):
        if not viewer.is_running():
            return False
        apply_frame(model, data, thumb_traj[i], finger_traj[i], finger_name)
        viewer.sync()
        time.sleep(dt)
    return True


def main():
    print("生成协同捏合轨迹...")
    gen = CoordinatedPinchGenerator()

    result_index = gen.generate_coordinated_trajectory('index', sweep_idx='auto',
                                                       n_sweep=30, n_interp=4)
    result_middle = gen.generate_coordinated_trajectory('middle', sweep_idx='auto',
                                                        n_sweep=30, n_interp=4)

    if result_index is None or not result_index['success']:
        print("食指协同轨迹生成失败")
        return
    if result_middle is None or not result_middle['success']:
        print("中指协同轨迹生成失败")
        return

    xml_path = os.path.join(os.path.dirname(__file__), 'mujoco_model', 'jack_hand_3f.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("\n启动 MuJoCo 可视化 (关闭窗口退出)")
    print("  Phase 1: 接近 → 食指捏合")
    print("  Phase 2: 食指协同运动 (前进)")
    print("  Phase 3: 食指协同运动 (后退)")
    print("  Phase 4: 撤回")
    print("  Phase 5-8: 中指同上")
    print("  循环播放")

    dt_approach = 0.03
    dt_sweep = 0.04
    dt_retract = 0.03

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for result in [result_index, result_middle]:
                fn = result['finger_name']

                # 接近
                ok = play_phase(viewer, model, data,
                                result['approach_thumb'],
                                result['approach_finger'],
                                fn, dt_approach)
                if not ok:
                    return
                time.sleep(0.3)

                # 协同运动 - 前进
                ok = play_phase(viewer, model, data,
                                result['sweep_thumb'],
                                result['sweep_finger'],
                                fn, dt_sweep)
                if not ok:
                    return
                time.sleep(0.3)

                # 协同运动 - 后退
                ok = play_phase(viewer, model, data,
                                result['sweep_thumb'][::-1],
                                result['sweep_finger'][::-1],
                                fn, dt_sweep)
                if not ok:
                    return
                time.sleep(0.3)

                # 撤回
                ok = play_phase(viewer, model, data,
                                result['retract_thumb'],
                                result['retract_finger'],
                                fn, dt_retract)
                if not ok:
                    return

                # 归零
                data.qpos[:] = 0
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.5)


if __name__ == '__main__':
    main()
