"""
coordinated_pinch_3f.py - 捏合状态下协同运动轨迹生成

在拇指-食指/拇指-中指捏合约束下，生成在可达空间内的协同运动轨迹。
通过扫描拇指关节角度并优化其余关节保持指尖接触来探索捏合流形。

7 DOF (拇指4 + 手指3) - 3 接触约束 = 4 DOF 自由运动。
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
from pinch_trajectories_3f import (
    PalmFrameTransform, thumb_fk_mjcf, finger_fk_mjcf,
    _load_joint_limits_deg, _XML_PATH,
)


class CoordinatedPinchGenerator:

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

    # ===== 初始捏合搜索 (复用网格搜索 + 优化) =====

    def _find_initial_pinch(self, finger_name, n_grid=10):
        tl = self.thumb_limits
        fl = self.finger_limits[finger_name]

        q1t = np.linspace(tl[0][0], tl[0][1], n_grid)
        q2t = np.linspace(tl[1][0], tl[1][1], max(n_grid // 2, 4))
        q3t = np.linspace(tl[2][0], tl[2][1], n_grid)
        q4t = np.linspace(tl[3][0], tl[3][1], max(n_grid // 2, 4))

        q1f = np.linspace(fl[0][0], fl[0][1], n_grid)
        q2f = np.linspace(fl[1][0], fl[1][1], max(n_grid // 2, 4))
        q3f = np.linspace(fl[2][0], fl[2][1], n_grid)

        thumb_tips, thumb_qs = [], []
        for a in q1t:
            for b in q2t:
                for c in q3t:
                    for d in q4t:
                        thumb_tips.append(self.thumb_tip_palm([a, b, c, d]))
                        thumb_qs.append([a, b, c, d])

        finger_tips, finger_qs = [], []
        for a in q1f:
            for b in q2f:
                for c in q3f:
                    finger_tips.append(self.finger_tip_palm([a, b, c], finger_name))
                    finger_qs.append([a, b, c])

        thumb_arr = np.array(thumb_tips)
        finger_arr = np.array(finger_tips)

        best_dist = float('inf')
        best_tq, best_fq = np.zeros(4), np.zeros(3)
        chunk = 500
        for start in range(0, len(thumb_arr), chunk):
            end = min(start + chunk, len(thumb_arr))
            diff = thumb_arr[start:end, None, :] - finger_arr[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            d = dists[idx]
            if d < best_dist:
                best_dist = d
                best_tq = np.array(thumb_qs[start + idx[0]])
                best_fq = np.array(finger_qs[idx[1]])

        def obj(x):
            t = self.thumb_tip_palm(x[:4])
            f = self.finger_tip_palm(x[4:], finger_name)
            return np.sum((t - f) ** 2)

        x0 = np.concatenate([best_tq, best_fq])
        bounds = self.thumb_limits + self.finger_limits[finger_name]
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-10})

        return res.x[:4].copy(), res.x[4:].copy()

    # ===== 关节空间扫描 =====

    def _sweep_from(self, init_tq, init_fq, finger_name, sweep_idx, param_values):
        """固定一个拇指关节，扫描其值，优化其余关节保持接触"""
        other_idx = [i for i in range(4) if i != sweep_idx]
        configs = []
        prev_tq = init_tq.copy()
        prev_fq = init_fq.copy()

        for val in param_values:
            x0 = np.array([prev_tq[i] for i in other_idx] + list(prev_fq))
            bounds_list = ([self.thumb_limits[i] for i in other_idx]
                           + self.finger_limits[finger_name])

            def obj(x, _val=val):
                tq = np.zeros(4)
                tq[sweep_idx] = _val
                for k, idx in enumerate(other_idx):
                    tq[idx] = x[k]
                fq = x[len(other_idx):]
                t_tip = self.thumb_tip_palm(tq)
                f_tip = self.finger_tip_palm(fq, finger_name)
                return np.sum((t_tip - f_tip) ** 2)

            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds_list,
                           options={'maxiter': 300, 'ftol': 1e-10})

            tq = np.zeros(4)
            tq[sweep_idx] = val
            for k, idx in enumerate(other_idx):
                tq[idx] = res.x[k]
            fq = res.x[len(other_idx):]

            t_tip = self.thumb_tip_palm(tq)
            f_tip = self.finger_tip_palm(fq, finger_name)
            dist = np.linalg.norm(t_tip - f_tip)

            if dist < 2.0:
                configs.append({
                    'thumb_q': tq.copy(),
                    'finger_q': fq.copy(),
                    'contact_point': (t_tip + f_tip) / 2,
                    'distance': dist,
                })
                prev_tq = tq
                prev_fq = fq
            else:
                break

        return configs

    def sweep_joint_pinch(self, finger_name, sweep_idx=0, n_steps=30):
        """
        扫描一个拇指关节 (sweep_idx: 0=q1屈伸, 1=q2内收外展)
        同时优化其余关节保持捏合接触。
        返回沿扫描路径的捏合配置列表。
        """
        print(f"  [初始] 搜索初始捏合配置...")
        init_tq, init_fq = self._find_initial_pinch(finger_name)
        init_val = init_tq[sweep_idx]

        t_tip = self.thumb_tip_palm(init_tq)
        f_tip = self.finger_tip_palm(init_fq, finger_name)
        print(f"  [初始] 接触距离 = {np.linalg.norm(t_tip - f_tip):.3f} mm")
        print(f"  [初始] 接触点 = [{t_tip[0]:.1f}, {t_tip[1]:.1f}, {t_tip[2]:.1f}]")

        lo, hi = self.thumb_limits[sweep_idx]
        joint_names = ['q1(屈伸)', 'q2(内收外展)', 'q3', 'q4']
        print(f"  [扫描] 拇指 {joint_names[sweep_idx]}: "
              f"{lo:.1f}° → {hi:.1f}°, 初始={init_val:.1f}°, {n_steps} 步")

        vals_lo = np.linspace(init_val, lo, n_steps // 2 + 1)
        configs_lo = self._sweep_from(init_tq, init_fq, finger_name, sweep_idx, vals_lo)

        vals_hi = np.linspace(init_val, hi, n_steps // 2 + 1)
        configs_hi = self._sweep_from(init_tq, init_fq, finger_name, sweep_idx, vals_hi)

        all_configs = configs_lo[::-1] + configs_hi[1:]
        print(f"  [扫描] 有效配置: {len(all_configs)} 个 "
              f"(向下 {len(configs_lo)}, 向上 {len(configs_hi)})")

        return all_configs

    # ===== 轨迹生成 =====

    @staticmethod
    def _minimum_jerk(t):
        return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5

    def _select_best_sweep(self, finger_name, n_sweep=30):
        """自动尝试扫描 4 个拇指关节，选择接触点跨度最大的"""
        joint_names = ['q1(屈伸)', 'q2(内收外展)', 'q3', 'q4']
        best_configs = None
        best_span = 0
        best_idx = 0

        for si in range(4):
            configs = self.sweep_joint_pinch(finger_name, si, n_sweep)
            if len(configs) < 3:
                continue
            pts = np.array([c['contact_point'] for c in configs])
            span = np.linalg.norm(pts[0] - pts[-1])
            print(f"  [自动选择] {joint_names[si]}: "
                  f"{len(configs)} 配置, 跨度 {span:.1f} mm")
            if span > best_span:
                best_span = span
                best_configs = configs
                best_idx = si

        if best_configs is not None:
            print(f"  [自动选择] → 选用 {joint_names[best_idx]}, 跨度 {best_span:.1f} mm")
        return best_configs, best_idx

    def generate_coordinated_trajectory(self, finger_name, sweep_idx='auto',
                                        n_sweep=30, n_interp=4):
        """
        生成完整的协同捏合运动轨迹。

        sweep_idx: 0-3 指定扫描的拇指关节, 'auto' 自动选择跨度最大的。

        返回:
            approach_thumb/finger: 从零位到第一个捏合点的轨迹
            sweep_thumb/finger: 捏合状态下的协同运动轨迹
            retract_thumb/finger: 从最后捏合点回到零位的轨迹
            contact_points: 协同运动中的接触点轨迹
            distances: 指尖距离
        """
        if sweep_idx == 'auto':
            configs, chosen_idx = self._select_best_sweep(finger_name, n_sweep)
        else:
            configs = self.sweep_joint_pinch(finger_name, sweep_idx, n_sweep)

        if configs is None or len(configs) < 3:
            print(f"  [失败] 仅找到 {len(configs) if configs else 0} 个有效配置，不足以生成轨迹")
            return None

        # 提取 waypoint 关节角
        wp_thumb = np.array([c['thumb_q'] for c in configs])
        wp_finger = np.array([c['finger_q'] for c in configs])
        wp_contact = np.array([c['contact_point'] for c in configs])
        wp_dist = np.array([c['distance'] for c in configs])

        print(f"\n  [可达空间] 接触点范围:")
        print(f"    X: [{wp_contact[:, 0].min():.1f}, {wp_contact[:, 0].max():.1f}] mm")
        print(f"    Y: [{wp_contact[:, 1].min():.1f}, {wp_contact[:, 1].max():.1f}] mm")
        print(f"    Z: [{wp_contact[:, 2].min():.1f}, {wp_contact[:, 2].max():.1f}] mm")
        span = np.linalg.norm(wp_contact[0] - wp_contact[-1])
        print(f"    总跨度: {span:.1f} mm")
        print(f"  [精度] waypoint 最大指尖距离: {wp_dist.max():.3f} mm")

        # 插值: waypoints 之间 minimum-jerk
        n_total = (len(configs) - 1) * n_interp + 1
        sweep_thumb = np.zeros((n_total, 4))
        sweep_finger = np.zeros((n_total, 3))
        contact_points = np.zeros((n_total, 3))
        distances = np.zeros(n_total)

        idx = 0
        for i in range(len(configs) - 1):
            t = np.linspace(0, 1, n_interp + 1)[:-1]
            s = self._minimum_jerk(t)
            for sj in s:
                sweep_thumb[idx] = (1 - sj) * wp_thumb[i] + sj * wp_thumb[i + 1]
                sweep_finger[idx] = (1 - sj) * wp_finger[i] + sj * wp_finger[i + 1]
                t_tip = self.thumb_tip_palm(sweep_thumb[idx])
                f_tip = self.finger_tip_palm(sweep_finger[idx], finger_name)
                contact_points[idx] = (t_tip + f_tip) / 2
                distances[idx] = np.linalg.norm(t_tip - f_tip)
                idx += 1

        sweep_thumb[idx] = wp_thumb[-1]
        sweep_finger[idx] = wp_finger[-1]
        t_tip = self.thumb_tip_palm(sweep_thumb[idx])
        f_tip = self.finger_tip_palm(sweep_finger[idx], finger_name)
        contact_points[idx] = (t_tip + f_tip) / 2
        distances[idx] = np.linalg.norm(t_tip - f_tip)
        idx += 1

        sweep_thumb = sweep_thumb[:idx]
        sweep_finger = sweep_finger[:idx]
        contact_points = contact_points[:idx]
        distances = distances[:idx]

        print(f"  [轨迹] 协同运动 {len(sweep_thumb)} 帧")
        print(f"  [精度] 插值后最大指尖距离: {distances.max():.3f} mm")

        # 接近轨迹: 零位 → 第一帧
        n_approach = 30
        t_app = np.linspace(0, 1, n_approach)
        s_app = self._minimum_jerk(t_app)
        approach_thumb = np.outer(s_app, sweep_thumb[0])
        approach_finger = np.outer(s_app, sweep_finger[0])

        # 撤回轨迹: 最后帧 → 零位
        retract_thumb = np.outer(1 - s_app, sweep_thumb[-1])
        retract_finger = np.outer(1 - s_app, sweep_finger[-1])

        return {
            'success': True,
            'finger_name': finger_name,
            'approach_thumb': approach_thumb,
            'approach_finger': approach_finger,
            'sweep_thumb': sweep_thumb,
            'sweep_finger': sweep_finger,
            'retract_thumb': retract_thumb,
            'retract_finger': retract_finger,
            'contact_points': contact_points,
            'distances': distances,
            'n_waypoints': len(configs),
        }

    # ===== 捏合可达空间探索 =====

    def map_pinch_workspace(self, finger_name, n_random=200):
        """
        通过多起点随机优化探索捏合可达空间。
        返回所有可达接触点及其对应关节角。
        """
        print(f"  [空间探索] {n_random} 次随机优化...")
        tl = self.thumb_limits
        fl = self.finger_limits[finger_name]

        configs = []
        for i in range(n_random):
            x0 = np.array([
                np.random.uniform(tl[0][0], tl[0][1]),
                np.random.uniform(tl[1][0], tl[1][1]),
                np.random.uniform(tl[2][0], tl[2][1]),
                np.random.uniform(tl[3][0], tl[3][1]),
                np.random.uniform(fl[0][0], fl[0][1]),
                np.random.uniform(fl[1][0], fl[1][1]),
                np.random.uniform(fl[2][0], fl[2][1]),
            ])

            def obj(x):
                t = self.thumb_tip_palm(x[:4])
                f = self.finger_tip_palm(x[4:], finger_name)
                return np.sum((t - f) ** 2)

            bounds = tl + fl
            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8})

            t_tip = self.thumb_tip_palm(res.x[:4])
            f_tip = self.finger_tip_palm(res.x[4:], finger_name)
            dist = np.linalg.norm(t_tip - f_tip)

            if dist < 1.0:
                configs.append({
                    'thumb_q': res.x[:4].copy(),
                    'finger_q': res.x[4:].copy(),
                    'contact_point': (t_tip + f_tip) / 2,
                    'distance': dist,
                })

        points = np.array([c['contact_point'] for c in configs])
        print(f"  [空间探索] 找到 {len(configs)} 个有效捏合配置")
        if len(configs) > 0:
            print(f"    X: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] mm")
            print(f"    Y: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] mm")
            print(f"    Z: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] mm")

        return configs


def print_coordinated_summary(result):
    if result is None or not result['success']:
        print("  FAILED")
        return

    print(f"\n--- {result['finger_name']} 协同捏合轨迹 ---")
    print(f"  waypoints: {result['n_waypoints']}")
    print(f"  协同运动帧数: {len(result['sweep_thumb'])}")
    print(f"  最大指尖距离: {result['distances'].max():.3f} mm")

    cp = result['contact_points']
    print(f"  接触点起点: [{cp[0][0]:.1f}, {cp[0][1]:.1f}, {cp[0][2]:.1f}] mm")
    print(f"  接触点终点: [{cp[-1][0]:.1f}, {cp[-1][1]:.1f}, {cp[-1][2]:.1f}] mm")
    print(f"  接触点移动距离: {np.linalg.norm(cp[-1] - cp[0]):.1f} mm")


if __name__ == '__main__':
    gen = CoordinatedPinchGenerator()

    print("\n" + "=" * 60)
    print("  拇指-食指 协同捏合运动 (自动选择最优扫描维度)")
    print("=" * 60)
    result_index = gen.generate_coordinated_trajectory('index', sweep_idx='auto')
    print_coordinated_summary(result_index)

    print("\n" + "=" * 60)
    print("  拇指-中指 协同捏合运动 (自动选择最优扫描维度)")
    print("=" * 60)
    result_middle = gen.generate_coordinated_trajectory('middle', sweep_idx='auto')
    print_coordinated_summary(result_middle)

    print("\n" + "=" * 60)
    print("  捏合可达空间探索 (食指)")
    print("=" * 60)
    workspace = gen.map_pinch_workspace('index', n_random=200)
