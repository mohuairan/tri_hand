"""
retargeting_3f.py - 三指版 MediaPipe → MuJoCo 遥操作映射

仅处理拇指、食指、中指三个手指的映射。
"""

import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from finger_fkik_py.inverse_kinematics import InverseKinematics
from finger_fkik_py.params import FingerParams
from thumb_fkik_py.thumb_ik_solve import thumb_ik_solve
from thumb_fkik_py.params import ThumbParams


class HandRetargeter:
    def __init__(self):
        self.finger_params = FingerParams()
        self.finger_ik = InverseKinematics(self.finger_params)
        self.thumb_params = ThumbParams()

        self.finger_indices = {
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
        }

        self.last_q = {
            'index': [0, 0, 0],
            'middle': [0, 0, 0],
            'thumb': [-30, 0, -20, -10]
        }

        self.R_thumb_base = np.array([
            [0, 0, -1],
            [0, 1,  0],
            [1, 0,  0]
        ])

    def build_palm_frame(self, landmarks):
        wrist = landmarks[0]
        mcp_index = landmarks[5]
        mcp_little = landmarks[17]
        mcp_middle = landmarks[9]

        y_vec = mcp_middle - wrist
        y_vec = y_vec / (np.linalg.norm(y_vec) + 1e-6)

        x_vec_raw = mcp_little - mcp_index

        z_vec = np.cross(y_vec, x_vec_raw)
        z_vec = z_vec / (np.linalg.norm(z_vec) + 1e-6)

        x_vec = np.cross(y_vec, z_vec)
        x_vec = x_vec / (np.linalg.norm(x_vec) + 1e-6)

        R_palm = np.column_stack((x_vec, y_vec, z_vec))
        return R_palm

    def mp_to_jack_finger(self, v_mp, R_palm):
        v_local = R_palm.T @ v_mp
        return np.array([v_local[0], -v_local[2], v_local[1]])

    def mp_to_jack_thumb(self, v_mp, R_palm):
        v_local = R_palm.T @ v_mp
        return np.array([v_local[2], v_local[1], -v_local[0]])

    def process(self, landmarks_3d):
        ctrl_dict = {}
        if landmarks_3d is None:
            return ctrl_dict

        R_palm = self.build_palm_frame(landmarks_3d)

        # ========== 处理食指和中指 ==========
        L_jack_finger = self.finger_params.L_PM + self.finger_params.L_MN + self.finger_params.L_NT

        for name, idxs in self.finger_indices.items():
            mcp_idx, pip_idx, dip_idx, tip_idx = idxs

            l_mp1 = np.linalg.norm(landmarks_3d[pip_idx] - landmarks_3d[mcp_idx])
            l_mp2 = np.linalg.norm(landmarks_3d[dip_idx] - landmarks_3d[pip_idx])
            l_mp3 = np.linalg.norm(landmarks_3d[tip_idx] - landmarks_3d[dip_idx])
            L_mp_finger = l_mp1 + l_mp2 + l_mp3
            scale = L_jack_finger / (L_mp_finger + 1e-6)

            v_mp = landmarks_3d[tip_idx] - landmarks_3d[mcp_idx]

            v_jack = self.mp_to_jack_finger(v_mp, R_palm) * scale

            P_0 = np.array(self.finger_params.P)
            T_target = P_0 + v_jack

            q1_sols, q2_sols, info = self.finger_ik.solve_q1q2(T_target)

            if info['success'] and len(q1_sols) > 0:
                q1, q2 = q1_sols[0], q2_sols[0]
                q3 = info['q3']
                self.last_q[name] = [q1, q2, q3]
            else:
                q1, q2, q3 = self.last_q[name]

            ctrl_dict[f'{name}_act_q1'] = np.radians(q1)
            ctrl_dict[f'{name}_act_q2'] = np.radians(-q2)
            ctrl_dict[f'{name}_act_q3'] = np.radians(q3)

        # ========== 处理拇指 ==========
        cmc_idx, mcp_idx, ip_idx, tip_idx = 1, 2, 3, 4

        l_mp1 = np.linalg.norm(landmarks_3d[mcp_idx] - landmarks_3d[cmc_idx])
        l_mp2 = np.linalg.norm(landmarks_3d[ip_idx] - landmarks_3d[mcp_idx])
        l_mp3 = np.linalg.norm(landmarks_3d[tip_idx] - landmarks_3d[ip_idx])
        L_mp_thumb = l_mp1 + l_mp2 + l_mp3
        L_jack_thumb = self.thumb_params.L1 + self.thumb_params.L2 + self.thumb_params.L3
        scale_thumb = (L_jack_thumb * 0.95) / (L_mp_thumb + 1e-6)

        v_mp_thumb = landmarks_3d[tip_idx] - landmarks_3d[cmc_idx]
        v_jack_thumb = self.mp_to_jack_thumb(v_mp_thumb, R_palm) * scale_thumb
        P_target_thumb = v_jack_thumb

        vec_a = landmarks_3d[mcp_idx] - landmarks_3d[cmc_idx]
        vec_b = landmarks_3d[tip_idx] - landmarks_3d[mcp_idx]
        n_mp = np.cross(vec_a, vec_b)
        norm_n = np.linalg.norm(n_mp)
        if norm_n < 1e-6:
            n_mp = R_palm[:, 2]
        else:
            n_mp = n_mp / norm_n

        n_c_thumb = self.mp_to_jack_thumb(n_mp, R_palm)
        n_c_thumb = n_c_thumb / (np.linalg.norm(n_c_thumb) + 1e-6)

        q_thumb, info_thumb = thumb_ik_solve(P_target_thumb, n_c_thumb, self.thumb_params)

        if q_thumb is not None:
            self.last_q['thumb'] = q_thumb
        else:
            error_reason = info_thumb.get('error', '未知错误') if isinstance(info_thumb, dict) else "求解器未返回错误信息"
            print(f"[警告] 大拇指逆解失败，保持上一帧！原因: {error_reason}")
            q_thumb = self.last_q['thumb']

        ctrl_dict['thumb_act_q1'] = np.radians(q_thumb[0])
        ctrl_dict['thumb_act_q2'] = np.radians(q_thumb[1])
        ctrl_dict['thumb_act_q3'] = np.radians(q_thumb[2])
        ctrl_dict['thumb_act_q4'] = np.radians(q_thumb[3])

        return ctrl_dict
