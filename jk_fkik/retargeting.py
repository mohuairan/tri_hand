import numpy as np
import math

# Try importing IK solvers
try:
    from finger_fkik_py.inverse_kinematics import InverseKinematics
    from finger_fkik_py.params import FingerParams
    from thumb_fkik_py.thumb_ik_solve import thumb_ik_solve
    from thumb_fkik_py.params import ThumbParams
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from finger_fkik_py.inverse_kinematics import InverseKinematics
    from finger_fkik_py.params import FingerParams
    from thumb_fkik_py.thumb_ik_solve import thumb_ik_solve
    from thumb_fkik_py.params import ThumbParams

class HandRetargeter:
    def __init__(self):
        self.finger_params = FingerParams()
        self.finger_ik = InverseKinematics(self.finger_params)
        self.thumb_params = ThumbParams()
        
        # 指节索引映射
        # 注意：摄像头做了水平镜像翻转 (cv2.flip)，因此 MediaPipe 关键点顺序与屏幕上目视方向相反
        # 屏幕上从左到右看是：大拇指、食指、中指、无名指、小拇指
        # 但翻转后 MediaPipe 内部 ID 的空间位置也跟着翻了
        # 因此需要把 little(17) 映射到屏幕左侧(真实食指位)，index(5) 映射到屏幕右侧(真实小指位)
        self.finger_indices = {
            'little': [5, 6, 7, 8],
            'ring': [9, 10, 11, 12],
            'middle': [13, 14, 15, 16],
            'index': [17, 18, 19, 20]
        }
        
        # 缓存上一帧的解（防抖与防错误）
        self.last_q = {
            'index': [0, 0, 0],
            'middle': [0, 0, 0],
            'ring': [0, 0, 0],
            'little': [0, 0, 0],
            'thumb': [-30, 0, -20, -10]
        }
        
        # 拇指 Base 旋转矩阵 (Ry(-90))
        self.R_thumb_base = np.array([
            [0, 0, -1],
            [0, 1,  0],
            [1, 0,  0]
        ])

    def build_palm_frame(self, landmarks):
        """构建手掌局部坐标系 (恢复到上一版验证通过的结构)"""
        wrist = landmarks[0]
        mcp_index = landmarks[5]
        mcp_little = landmarks[17]
        mcp_middle = landmarks[9]
        
        # Y_local: 手指方向 (手腕到中指根部)
        y_vec = mcp_middle - wrist
        y_vec = y_vec / (np.linalg.norm(y_vec) + 1e-6)
        
        # X_raw: 食指到小指
        x_vec_raw = mcp_little - mcp_index
        
        # Z_local: 掌心法向 (Palmar) — 上一版验证 bending 方向正确
        z_vec = np.cross(y_vec, x_vec_raw)
        z_vec = z_vec / (np.linalg.norm(z_vec) + 1e-6)
        
        # 正交化 X_local
        x_vec = np.cross(y_vec, z_vec)
        x_vec = x_vec / (np.linalg.norm(x_vec) + 1e-6)
        
        R_palm = np.column_stack((x_vec, y_vec, z_vec))
        return R_palm

    def mp_to_jack_finger(self, v_mp, R_palm):
        """将 MediaPipe 向量转换到 Jack 四指 IK 求解器坐标系"""
        v_local = R_palm.T @ v_mp
        # 坐标交换: IK 求解器的 X=lateral, Y=palmar, Z=finger
        # v_local = [v·X_orth, v·Y_finger, v·Z_palmar]
        # Solver  = [v·X_orth, -v·Z_palmar, v·Y_finger]
        return np.array([v_local[0], -v_local[2], v_local[1]])

    def mp_to_jack_thumb(self, v_mp, R_palm):
        """将 MediaPipe 向量转换到 Jack 大拇指局部基座"""
        v_local = R_palm.T @ v_mp
        # 大拇指保留未经翻转的 Y 轴（Palmar 为正），避免计算 q1 时 ny 为负导致角度翻转
        return np.array([v_local[0], v_local[2], v_local[1]])

    def process(self, landmarks_3d):
        """
        处理一帧 21 点骨架，返回关节控制指令字典
        """
        ctrl_dict = {}
        if landmarks_3d is None:
            return ctrl_dict
            
        R_palm = self.build_palm_frame(landmarks_3d)
        
        # ========== 处理四指 ==========
        L_jack_finger = self.finger_params.L_PM + self.finger_params.L_MN + self.finger_params.L_NT
        
        for name, idxs in self.finger_indices.items():
            mcp_idx, pip_idx, dip_idx, tip_idx = idxs
            
            # 计算 MP 手指骨骼真实总长（用于动态缩放）
            l_mp1 = np.linalg.norm(landmarks_3d[pip_idx] - landmarks_3d[mcp_idx])
            l_mp2 = np.linalg.norm(landmarks_3d[dip_idx] - landmarks_3d[pip_idx])
            l_mp3 = np.linalg.norm(landmarks_3d[tip_idx] - landmarks_3d[dip_idx])
            L_mp_finger = l_mp1 + l_mp2 + l_mp3
            scale = L_jack_finger / (L_mp_finger + 1e-6)
            
            # 指根到指尖的向量
            v_mp = landmarks_3d[tip_idx] - landmarks_3d[mcp_idx]
            
            # 转换到 Jack Palm 坐标系并缩放 (四指专用映射)
            v_jack = self.mp_to_jack_finger(v_mp, R_palm) * scale
            
            # 组装局部 Target 坐标 P_0 = (0, 0, 27.98)
            P_0 = np.array(self.finger_params.P)
            T_target = P_0 + v_jack
            
            # 调用 IK
            q1_sols, q2_sols, info = self.finger_ik.solve_q1q2(T_target)
            
            if info['success'] and len(q1_sols) > 0:
                q1, q2 = q1_sols[0], q2_sols[0]
                q3 = info['q3']
                # 更新缓存
                self.last_q[name] = [q1, q2, q3]
            else:
                # 求解失败，使用上一帧
                q1, q2, q3 = self.last_q[name]
                
            ctrl_dict[f'{name}_act_q1'] = np.radians(q1)
            ctrl_dict[f'{name}_act_q2'] = np.radians(-q2)  # 取反修正侧摆方向
            ctrl_dict[f'{name}_act_q3'] = np.radians(q3)

        # ========== 处理拇指 ==========
        cmc_idx, mcp_idx, ip_idx, tip_idx = 1, 2, 3, 4
        
        # 拇指总长计算用于缩放
        l_mp1 = np.linalg.norm(landmarks_3d[mcp_idx] - landmarks_3d[cmc_idx])
        l_mp2 = np.linalg.norm(landmarks_3d[ip_idx] - landmarks_3d[mcp_idx])
        l_mp3 = np.linalg.norm(landmarks_3d[tip_idx] - landmarks_3d[ip_idx])
        L_mp_thumb = l_mp1 + l_mp2 + l_mp3
        # L_jack_thumb = 38.0 + 38.0 + 29.0 # 近似长度 L_OA + L_AB + L_BE
        # scale_thumb = L_jack_thumb / (L_mp_thumb + 1e-6)
        L_jack_thumb = self.thumb_params.L1 + self.thumb_params.L2 + self.thumb_params.L3
        scale_thumb = (L_jack_thumb * 0.95) / (L_mp_thumb + 1e-6)

        # CMC 到 TIP 的向量
        v_mp_thumb = landmarks_3d[tip_idx] - landmarks_3d[cmc_idx]
        # 转换到 Thumb Base 局部坐标系 (大拇指专用映射)
        v_jack_palm = self.mp_to_jack_thumb(v_mp_thumb, R_palm) * scale_thumb
        P_target_thumb = self.R_thumb_base.T @ v_jack_palm
        
        # === 核心修正 ===
        # 人手大拇指的目标点在前方 (+X)，但 Jack Hand 的原生 IK 求解器工作空间在后方 (-X)
        # 因为用户在 XML 中将关节翻转了 180 度来“视觉上”纠正大拇指方向，
        # 我们必须把目标点也沿着 X 轴镜像到后面去，IK 才能在它的“旧世界”里找到合法解！
        P_target_thumb[0] = -P_target_thumb[0]
        
        # 计算约束平面法向量 n_c
        # 稳定方法：用拇指尼指节 (CMC->MCP) x (MCP->TIP) 得到弯曲平面的法向
        # 这个叉乘在拇指弯曲时方向稳定，当拇指弯曲时两节屠小为透视的叉乘最小化了共线问题
        vec_a = landmarks_3d[mcp_idx] - landmarks_3d[cmc_idx]  # CMC -> MCP
        vec_b = landmarks_3d[tip_idx] - landmarks_3d[mcp_idx]  # MCP -> TIP
        n_mp = np.cross(vec_b, vec_a)
        norm_n = np.linalg.norm(n_mp)
        if norm_n < 1e-6:
            # 共线备用：用掌心法向收敛
            n_mp = R_palm[:, 1]
        else:
            n_mp = n_mp / norm_n
        
        n_jack_palm = self.mp_to_jack_thumb(n_mp, R_palm)
        n_c_thumb = self.R_thumb_base.T @ n_jack_palm
        n_c_thumb = n_c_thumb / (np.linalg.norm(n_c_thumb) + 1e-6)
        
        # 匹配 P_target_thumb 的 X 轴镜像，保持法向量与目标点的几何一致性
        n_c_thumb[0] = -n_c_thumb[0]
        
        # 调用拇指 IK
        q_thumb, info_thumb = thumb_ik_solve(P_target_thumb, n_c_thumb, self.thumb_params)
        
        if q_thumb is not None:
            # 应用 180 度翻转修正给 q3 q4 (针对之前我们在 XML 里翻转了 thumb_q2_frame)
            # 由于真实的 Python 算法是向 -Local X 弯曲，我们保留原角度，因为我们只改了 XML，
            # 以匹配真实抓取动作，这里我们需要发送正确的控制指令以防方向错乱！
            # 实际上，传给 actuator 的角度不需要改符号，因为我们改了 xml axis 依然是 0 0 1？
            # 我们改的是 euler=0 0 180。这意味着 q3 和 q4 关节框架旋转了180度，所以对于同样的控制值，它会向上弯。
            # 因此，IK 算出来如果是 -50 度（向下），我们在 XML 里转了180度，就会变成向上！
            # 刚好负负得正，无需额外翻转符号！
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
