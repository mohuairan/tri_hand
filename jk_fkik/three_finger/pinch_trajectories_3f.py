"""
pinch_trajectories_3f.py - 三指版捏合动作轨迹生成

生成拇指-食指、拇指-中指正捏合的关节角轨迹。
通过 FK 网格搜索 + scipy 局部优化找到最近接触配置，
关节空间 minimum-jerk 插值生成平滑轨迹。

所有关节角均为 MJCF 约定（度），可直接转换为弧度发送给 MuJoCo actuator。
"""

import numpy as np
import math
import sys
import os
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy.optimize import minimize
from finger_fkik_py.params import FingerParams
from thumb_fkik_py import ThumbParams

_XML_PATH = os.path.join(os.path.dirname(__file__), 'mujoco_model', 'jack_hand_3f.xml')


def _load_joint_limits_deg(xml_path=_XML_PATH):
    model = mujoco.MjModel.from_xml_path(xml_path)

    def _range_deg(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        return (math.degrees(lo), math.degrees(hi))

    thumb = [_range_deg(f'thumb_q{i}') for i in range(1, 5)]
    finger = {}
    for fn in ('index', 'middle'):
        finger[fn] = [_range_deg(f'{fn}_q{i}') for i in range(1, 4)]
    return thumb, finger


# ==================== 坐标变换 ====================

class PalmFrameTransform:
    def __init__(self):
        self._Ry_neg90 = self._Ry(np.radians(-90))

        self._finger_offsets = {
            'index':  np.array([-18.0, 0.0, 68.0]),
            'middle': np.array([0.0, 0.0, 72.0]),
        }
        self._thumb_offset = np.array([-26.0, 5.0, 5.0])

    @staticmethod
    def _Ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def thumb_local_to_palm(self, P_local):
        return self._thumb_offset + self._Ry_neg90 @ P_local

    def finger_local_to_palm(self, T_pos, finger_name):
        return self._finger_offsets[finger_name] + T_pos


# ==================== MJCF FK 计算 ====================

_Q3Q4_POLY = np.array([-0.01462306, 1.27106752, 0.07657957, 0.05314150, 0.10674332])


def _Rx(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def thumb_fk_mjcf(q1_deg, q2_deg, q3_deg, q4_deg, params):
    """MJCF 拇指 FK: Rx(q1)·Rz(π+q2)·Ry(α1) → OM → Ry(th3) → MN → Ry(th4) → NT"""
    q1 = math.radians(q1_deg)
    q2 = math.radians(q2_deg)
    th3 = math.radians(params.alpha2 + q3_deg)
    th4 = math.radians(params.alpha3 + q4_deg)
    alpha1 = math.radians(params.alpha1)

    R_base = _Rx(q1) @ _Rz(math.pi + q2) @ _Ry(alpha1)
    v = np.array([0.0, 0.0, params.L3])
    v = np.array([0.0, 0.0, params.L2]) + _Ry(th4) @ v
    v = np.array([0.0, 0.0, params.L1]) + _Ry(th3) @ v
    P_local = R_base @ v

    _thumb_offset = np.array([-26.0, 5.0, 5.0])
    return _thumb_offset + _Ry(math.radians(-90)) @ P_local


def finger_fk_mjcf(q1_deg, q2_deg, q3_deg, params):
    """MJCF 运动链 FK: Rx(θ_init+q1)·Ry(q2) → PM → Rx(q3) → MN → Rx(q4) → NT"""
    q1 = math.radians(q1_deg)
    q2 = math.radians(q2_deg)
    q3 = math.radians(q3_deg)
    theta_init = math.radians(-5.41)

    q4 = np.polyval(_Q3Q4_POLY[::-1], q3)

    R_base = _Rx(theta_init + q1) @ _Ry(q2)
    v_NT = np.array([0.0, 0.0, params.L_NT])
    v_MN = np.array([0.0, 0.0, params.L_MN]) + _Rx(q4) @ v_NT
    v_chain = np.array([0.0, 0.0, params.L_PM]) + _Rx(q3) @ v_MN
    return R_base @ v_chain


# ==================== 轨迹生成器 ====================

class PinchTrajectoryGenerator:

    def __init__(self):
        self.transform = PalmFrameTransform()
        self.finger_params = FingerParams()
        self.thumb_params = ThumbParams()
        self.thumb_limits, self.finger_limits = _load_joint_limits_deg()

    def thumb_tip_palm(self, q):
        return thumb_fk_mjcf(q[0], q[1], q[2], q[3], self.thumb_params)

    def finger_tip_palm(self, q, finger_name):
        T = finger_fk_mjcf(q[0], q[1], q[2], self.finger_params)
        return self.transform.finger_local_to_palm(T, finger_name)

    # ===== FK 网格搜索 =====

    def search_pinch_contact(self, finger_name, n_samples=12):
        print(f"  [搜索] 网格采样 n={n_samples}")

        tl = self.thumb_limits
        q1_t = np.linspace(tl[0][0], tl[0][1], n_samples)
        q2_t = np.linspace(tl[1][0], tl[1][1], max(n_samples // 2, 4))
        q3_t = np.linspace(tl[2][0], tl[2][1], n_samples)
        q4_t = np.linspace(tl[3][0], tl[3][1], max(n_samples // 2, 4))

        fl = self.finger_limits[finger_name]
        q1_f = np.linspace(fl[0][0], fl[0][1], n_samples)
        q2_f = np.linspace(fl[1][0], fl[1][1], max(n_samples // 2, 4))
        q3_f = np.linspace(fl[2][0], fl[2][1], n_samples)

        thumb_tips = []
        thumb_joints = []
        for _q1 in q1_t:
            for _q2 in q2_t:
                for _q3 in q3_t:
                    for _q4 in q4_t:
                        tip = self.thumb_tip_palm([_q1, _q2, _q3, _q4])
                        thumb_tips.append(tip)
                        thumb_joints.append([_q1, _q2, _q3, _q4])

        finger_tips = []
        finger_joints = []
        for _q1 in q1_f:
            for _q2 in q2_f:
                for _q3 in q3_f:
                    tip = self.finger_tip_palm([_q1, _q2, _q3], finger_name)
                    finger_tips.append(tip)
                    finger_joints.append([_q1, _q2, _q3])

        thumb_arr = np.array(thumb_tips)
        finger_arr = np.array(finger_tips)
        print(f"  [搜索] 拇指 {len(thumb_arr)} 样本, 四指 {len(finger_arr)} 样本")

        best_dist = float('inf')
        best_ti, best_fi = 0, 0
        chunk = 500
        for start in range(0, len(thumb_arr), chunk):
            end = min(start + chunk, len(thumb_arr))
            diff = thumb_arr[start:end, np.newaxis, :] - finger_arr[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            d = dists[min_idx]
            if d < best_dist:
                best_dist = d
                best_ti = start + min_idx[0]
                best_fi = min_idx[1]

        return {
            'thumb_joints': thumb_joints[best_ti],
            'finger_joints': finger_joints[best_fi],
            'thumb_tip_palm': thumb_arr[best_ti],
            'finger_tip_palm': finger_arr[best_fi],
            'min_distance': best_dist,
        }

    # ===== 局部优化 =====

    def refine_pinch(self, thumb_q0, finger_q0, finger_name):
        def objective(x):
            tq, fq = x[:4], x[4:]
            t_tip = self.thumb_tip_palm(tq)
            f_tip = self.finger_tip_palm(fq, finger_name)
            return np.sum((t_tip - f_tip)**2)

        x0 = np.concatenate([thumb_q0, finger_q0])
        bounds = self.thumb_limits + self.finger_limits[finger_name]

        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-10})

        opt_thumb = res.x[:4]
        opt_finger = res.x[4:]
        t_tip = self.thumb_tip_palm(opt_thumb)
        f_tip = self.finger_tip_palm(opt_finger, finger_name)
        dist = np.linalg.norm(t_tip - f_tip)

        return opt_thumb, opt_finger, t_tip, f_tip, dist

    # ===== 轨迹生成 =====

    @staticmethod
    def _minimum_jerk(t):
        return 10 * t**3 - 15 * t**4 + 6 * t**5

    def generate_pinch_trajectory(self, finger_name, n_points=50):
        search = self.search_pinch_contact(finger_name)
        if search is None:
            return {'success': False, 'error': 'FK 搜索失败'}

        print(f"  [网格] 距离 = {search['min_distance']:.2f} mm")
        print(f"  拇指 tip: [{search['thumb_tip_palm'][0]:.1f}, "
              f"{search['thumb_tip_palm'][1]:.1f}, {search['thumb_tip_palm'][2]:.1f}]")
        print(f"  手指 tip: [{search['finger_tip_palm'][0]:.1f}, "
              f"{search['finger_tip_palm'][1]:.1f}, {search['finger_tip_palm'][2]:.1f}]")

        thumb_end, finger_end, t_tip, f_tip, opt_dist = self.refine_pinch(
            np.array(search['thumb_joints']),
            np.array(search['finger_joints']),
            finger_name
        )
        contact_palm = (t_tip + f_tip) / 2.0
        print(f"  [优化] 距离 = {opt_dist:.2f} mm")
        print(f"  拇指 tip: [{t_tip[0]:.1f}, {t_tip[1]:.1f}, {t_tip[2]:.1f}]")
        print(f"  手指 tip: [{f_tip[0]:.1f}, {f_tip[1]:.1f}, {f_tip[2]:.1f}]")

        thumb_start = np.zeros(4)
        finger_start = np.zeros(3)

        t_norm = np.linspace(0, 1, n_points)
        s = self._minimum_jerk(t_norm)
        thumb_traj = np.outer(1 - s, thumb_start) + np.outer(s, thumb_end)
        finger_traj = np.outer(1 - s, finger_start) + np.outer(s, finger_end)

        thumb_tips = np.zeros((n_points, 3))
        finger_tips = np.zeros((n_points, 3))
        for i in range(n_points):
            thumb_tips[i] = self.thumb_tip_palm(thumb_traj[i])
            finger_tips[i] = self.finger_tip_palm(finger_traj[i], finger_name)

        end_dist = np.linalg.norm(thumb_tips[-1] - finger_tips[-1])

        return {
            'success': True,
            'finger_name': finger_name,
            'thumb_traj': thumb_traj,
            'finger_traj': finger_traj,
            'thumb_tips_palm': thumb_tips,
            'finger_tips_palm': finger_tips,
            'timestamps': t_norm,
            'contact_point': contact_palm,
            'end_tip_distance': end_dist,
            'thumb_end_q': thumb_end,
            'finger_end_q': finger_end,
        }

    def get_thumb_index_pinch(self, n_points=50):
        print("=" * 60)
        print("  拇指-食指 捏合轨迹")
        print("=" * 60)
        return self.generate_pinch_trajectory('index', n_points)

    def get_thumb_middle_pinch(self, n_points=50):
        print("=" * 60)
        print("  拇指-中指 捏合轨迹")
        print("=" * 60)
        return self.generate_pinch_trajectory('middle', n_points)


# ==================== 结果打印 ====================

def print_trajectory_summary(result):
    if not result['success']:
        print(f"  FAILED: {result.get('error', '未知')}")
        return

    print(f"\n--- {result['finger_name']} pinch summary ---")
    print(f"  Contact (palm): [{result['contact_point'][0]:.1f}, "
          f"{result['contact_point'][1]:.1f}, {result['contact_point'][2]:.1f}] mm")
    print(f"  End tip distance: {result['end_tip_distance']:.2f} mm")

    te = result['thumb_end_q']
    fe = result['finger_end_q']
    print(f"\n  Thumb  q (MJCF deg): [{te[0]:7.2f}, {te[1]:7.2f}, {te[2]:7.2f}, {te[3]:7.2f}]")
    print(f"  Finger q (MJCF deg): [{fe[0]:7.2f}, {fe[1]:7.2f}, {fe[2]:7.2f}]")

    tt = result['thumb_tips_palm']
    ft = result['finger_tips_palm']
    print(f"\n  Thumb  tip start: [{tt[0][0]:.1f}, {tt[0][1]:.1f}, {tt[0][2]:.1f}] mm")
    print(f"  Thumb  tip end:   [{tt[-1][0]:.1f}, {tt[-1][1]:.1f}, {tt[-1][2]:.1f}] mm")
    print(f"  Finger tip start: [{ft[0][0]:.1f}, {ft[0][1]:.1f}, {ft[0][2]:.1f}] mm")
    print(f"  Finger tip end:   [{ft[-1][0]:.1f}, {ft[-1][1]:.1f}, {ft[-1][2]:.1f}] mm")


if __name__ == '__main__':
    gen = PinchTrajectoryGenerator()

    print("\n>>> Thumb-Index Pinch <<<")
    result_index = gen.get_thumb_index_pinch(n_points=30)
    print_trajectory_summary(result_index)

    print("\n>>> Thumb-Middle Pinch <<<")
    result_middle = gen.get_thumb_middle_pinch(n_points=30)
    print_trajectory_summary(result_middle)
