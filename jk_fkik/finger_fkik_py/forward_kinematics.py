"""
forward_kinematics.py - 正运动学求解

实现手指机构的正运动学计算：
- fk_T0: 已知 q3，计算在 q1=0, q2=0 时末端 T 的位姿
- fk_chain_to_T: 已知 d1, d2 变化量和 q3，计算末端 T 的位姿
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .params import FingerParams
from . import utils
from .q3_to_q4 import q3_to_q4


@dataclass
class FKInfo:
    """正运动学求解信息"""
    success: bool = False
    error: Optional[str] = None
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    q4: float = 0.0
    theta_init: float = -5.41
    theta3: float = 0.0
    theta4: float = 0.0
    T_pos: Optional[np.ndarray] = None
    T_rot: Optional[np.ndarray] = None


def fk_T0(q3: float, params: FingerParams) -> Tuple[np.ndarray, np.ndarray, FKInfo]:
    """
    串联机构正运动学：已知 q3，计算在 q1=0, q2=0 时末端 T 的位姿
    
    Args:
        q3: 关节角 q3 (度)
        params: 手指机构参数
    
    Returns:
        T_pos: T 点位置 [Tx, Ty, Tz] (mm)
        T_rot: T 点姿态矩阵 (3x3)
        info: 详细信息
    """
    info = FKInfo()
    
    # ========== 步骤 1: 求解 q4 ==========
    q4 = q3_to_q4(q3, params)
    if q4 is None:
        info.success = False
        info.error = 'q3_to_q4 求解失败：四连杆机构无解'
        return np.array([0.0, 0.0, 0.0]), np.eye(3), info
    
    info.q4 = q4
    
    # ========== 步骤 2: 初始偏置旋转（绕 x 轴 -5.41°）==========
    theta_init = -5.41  # 初始位置时的 x 轴偏置角
    Rx_init = utils.rotation_matrix_x(theta_init)
    
    # ========== 步骤 3: 计算 R 矩阵（q1、q2 旋转）==========
    q1 = 0.0
    q2 = 0.0
    
    R_q1q2 = utils.rotation_matrix_from_q1q2(q1, q2)
    
    # 总旋转：先初始偏置，再 q1/q2 旋转
    R = R_q1q2 @ Rx_init
    
    # ========== 步骤 4: 计算后续旋转矩阵 ==========
    theta3 = q3 + 5.41
    theta4 = q4 - 8.15
    
    Rx3 = utils.rotation_matrix_x(theta3)
    Rx4 = utils.rotation_matrix_x(theta4)
    
    # ========== 步骤 5: 计算 T 点位置 ==========
    P = np.array(params.P)
    PM = params.L_PM
    MN = params.L_MN
    NT = params.L_NT
    
    v_NT = np.array([0.0, 0.0, NT])
    v_MN = np.array([0.0, 0.0, MN]) + Rx4 @ v_NT
    v_PM = np.array([0.0, 0.0, PM]) + Rx3 @ v_MN
    
    T_pos = P + R @ v_PM
    
    # ========== 步骤 6: 计算 T 点姿态 ==========
    T_rot = R @ Rx3 @ Rx4
    
    # ========== 信息输出 ==========
    info.success = True
    info.q1 = q1
    info.q2 = q2
    info.q3 = q3
    info.theta_init = theta_init
    info.theta3 = theta3
    info.theta4 = theta4
    info.T_pos = T_pos
    info.T_rot = T_rot
    
    return T_pos, T_rot, info


def fk_d1d2_to_q1q2(d1_init: float, delta_d1: float, 
                    d2_init: float, delta_d2: float, 
                    params: FingerParams) -> Tuple[float, float, Dict]:
    """
    根据 d1、d2 变化量正向求解 q1、q2
    
    Args:
        d1_init: 初始 d1 值 (mm)
        delta_d1: d1 变化量 (mm)
        d2_init: 初始 d2 值 (mm)
        delta_d2: d2 变化量 (mm)
        params: 手指机构参数
    
    Returns:
        q1: 屈伸角 (度)
        q2: 外展角 (度)
        info: 详细信息字典
    """
    # ========== 加载参数 ==========
    ax = params.ax
    ay = params.ay
    bx = params.bx
    by = params.by
    bz = params.bz
    px, py, pz = params.P
    l = params.l1
    
    # 注意：q1 限位与 MATLAB 版本保持一致 [-90, 0]
    q1_min = -90.0
    q1_max = 0.0
    q2_min = -45.0
    q2_max = 45.0
    
    # ========== 计算当前 d1、d2 绝对值 ==========
    d1 = d1_init + delta_d1
    d2 = d2_init + delta_d2
    
    C1 = np.array([ax, ay, d1])
    C2 = np.array([-ax, ay, d2])
    
    # ========== 步骤 1: q2 初值估计 ==========
    sin_q2_approx = (d2 - d1) / (2 * bx)
    
    info = {'success': False, 'error': None, 'q1': None, 'q2': None}
    
    if abs(sin_q2_approx) > 1:
        info['error'] = '无解：|d2-d1| > 2*bx'
        return None, None, info
    
    q2_0 = utils.rad2deg(math.asin(sin_q2_approx))
    
    if q2_0 < q2_min or q2_0 > q2_max:
        info['error'] = f'无解：q2={q2_0:.2f}°超出限位 [{q2_min}, {q2_max}]'
        return None, None, info
    
    # ========== 步骤 2: 求解 q1 ==========
    def q1_equation(q1_deg: float) -> float:
        q1_rad = utils.deg2rad(q1_deg)
        q2_rad = utils.deg2rad(q2_0)
        
        c1 = math.cos(q1_rad)
        s1 = math.sin(q1_rad)
        c2 = math.cos(q2_rad)
        s2 = math.sin(q2_rad)
        
        K = s1 * by + c1 * bz
        
        B1x = px + c2 * bx + s2 * K
        B1y = py + c1 * by - s1 * bz
        B1z = pz - s2 * bx + c2 * K
        
        F = (B1x - C1[0])**2 + (B1y - C1[1])**2 + (B1z - C1[2])**2 - l**2
        return F
    
    # 检查端点
    F_min = q1_equation(q1_min)
    F_max = q1_equation(q1_max)
    
    q1_sol = None
    
    if abs(F_min) < 1e-8:
        q1_sol = q1_min
    elif abs(F_max) < 1e-8:
        q1_sol = q1_max
    elif F_min * F_max < 0:
        # 使用二分法求解
        result, success = utils.numerical_solve_bounded(q1_equation, q1_min, q1_max)
        if success:
            q1_sol = result
        else:
            info['error'] = '无解：q1 求解失败'
            return None, None, info
    else:
        # 尝试牛顿法
        q1_guess = (q1_min + q1_max) / 2
        result, success = utils.numerical_solve_1d(q1_equation, q1_guess)
        if success and result is not None:
            q1_sol = result
        else:
            info['error'] = '无解：q1 方程无根'
            return None, None, info
    
    if q1_sol is None:
        info['error'] = '无解：q1 求解失败'
        return None, None, info
    
    # ========== 步骤 3: 双变量精化 ==========
    def residual(q):
        q1_rad = utils.deg2rad(q[0])
        q2_rad = utils.deg2rad(q[1])
        
        c1 = math.cos(q1_rad)
        s1 = math.sin(q1_rad)
        c2 = math.cos(q2_rad)
        s2 = math.sin(q2_rad)
        
        K = s1 * by + c1 * bz
        
        B1x = px + c2 * bx + s2 * K
        B1y = py + c1 * by - s1 * bz
        B1z = pz - s2 * bx + c2 * K
        
        B2x = px - c2 * bx + s2 * K
        B2y = B1y
        B2z = pz + s2 * bx + c2 * K
        
        F1 = (B1x - C1[0])**2 + (B1y - C1[1])**2 + (B1z - C1[2])**2 - l**2
        F2 = (B2x - C2[0])**2 + (B2y - C2[1])**2 + (B2z - C2[2])**2 - l**2
        
        return np.array([F1, F2])
    
    # 使用 scipy 进行双变量优化（与 MATLAB fsolve 一致）
    q_init = [q1_sol, q2_0]
    q1, q2 = q1_sol, q2_0  # 默认值
    
    try:
        from scipy.optimize import fsolve
        q_sol, info_opt, ier, mesg = fsolve(residual, q_init, full_output=True)
        if ier == 1:  # 求解成功
            # 规范化 q1 到 [-180, 180] 范围
            q1_raw = q_sol[0]
            q1 = utils.wrap_to_pi(utils.deg2rad(q1_raw))
            q1 = utils.rad2deg(q1)
            q2 = q_sol[1]
            info['scipy_used'] = True
        else:
            # fsolve 失败，使用初始解
            q1, q2 = q1_sol, q2_0
            info['scipy_fallback'] = True
    except ImportError:
        # 如果没有 scipy，使用初始解
        q1, q2 = q1_sol, q2_0
        info['scipy_not_available'] = True
    
    # ========== 步骤 4: 限位检查 ==========
    if q1 < q1_min or q1 > q1_max or q2 < q2_min or q2 > q2_max:
        info['error'] = f'无解：结果超出限位 (q1={q1:.2f}°, q2={q2:.2f}°)'
        return None, None, info
    
    info['success'] = True
    info['q1'] = q1
    info['q2'] = q2
    
    return q1, q2, info


def fk_chain_to_T(delta_d1: float, delta_d2: float, q3: float, 
                  params: FingerParams) -> Tuple[np.ndarray, np.ndarray, FKInfo]:
    """
    串联机构正运动学：计算末端 T 的位姿
    
    Args:
        delta_d1: d1 变化量 (mm)
        delta_d2: d2 变化量 (mm)
        q3: 关节角 q3 (度)
        params: 手指机构参数
    
    Returns:
        T_pos: T 点位置 [Tx, Ty, Tz] (mm)
        T_rot: T 点姿态矩阵 (3x3)
        info: 详细信息
    """
    info = FKInfo()
    
    # ========== 步骤 1: 获取 d1、d2 初始值 ==========
    from .inverse_kinematics import ik_d1_d2
    
    d1_init, d2_init, _, _, _, _, info_init = ik_d1_d2(0.0, 0.0, params)
    
    if not info_init['success']:
        info.success = False
        info.error = '初始状态计算失败：' + info_init.get('error', '')
        return np.array([0.0, 0.0, 0.0]), np.eye(3), info
    
    # ========== 步骤 2: 求解 q1、q2 ==========
    q1, q2, info_q = fk_d1d2_to_q1q2(d1_init, delta_d1, d2_init, delta_d2, params)
    
    if not info_q['success']:
        info.success = False
        info.error = 'q1/q2 求解失败：' + info_q.get('error', '')
        return np.array([0.0, 0.0, 0.0]), np.eye(3), info
    
    # ========== 步骤 3: 求解 q4 ==========
    q4 = q3_to_q4(q3, params)
    if q4 is None:
        info.success = False
        info.error = 'q3_to_q4 求解失败：四连杆机构无解'
        return np.array([0.0, 0.0, 0.0]), np.eye(3), info
    
    # ========== 步骤 4: 初始偏置旋转 ==========
    theta_init = -5.41
    Rx_init = utils.rotation_matrix_x(theta_init)
    
    # ========== 步骤 5: 计算 R 矩阵 ==========
    R_q1q2 = utils.rotation_matrix_from_q1q2(q1, q2)
    R = R_q1q2 @ Rx_init
    
    # ========== 步骤 6: 计算后续旋转矩阵 ==========
    theta3 = q3 + 5.41
    theta4 = q4 - 8.15
    
    Rx3 = utils.rotation_matrix_x(theta3)
    Rx4 = utils.rotation_matrix_x(theta4)
    
    # ========== 步骤 7: 计算 T 点位置 ==========
    P = np.array(params.P)
    PM = params.L_PM
    MN = params.L_MN
    NT = params.L_NT
    
    v_NT = np.array([0.0, 0.0, NT])
    v_MN = np.array([0.0, 0.0, MN]) + Rx4 @ v_NT
    v_PM = np.array([0.0, 0.0, PM]) + Rx3 @ v_MN
    
    T_pos = P + R @ v_PM
    
    # ========== 步骤 8: 计算 T 点姿态 ==========
    T_rot = R @ Rx3 @ Rx4
    
    # ========== 信息输出 ==========
    info.success = True
    info.q1 = q1
    info.q2 = q2
    info.q3 = q3
    info.q4 = q4
    info.theta_init = theta_init
    info.theta3 = theta3
    info.theta4 = theta4
    info.T_pos = T_pos
    info.T_rot = T_rot
    
    return T_pos, T_rot, info


class ForwardKinematics:
    """
    正运动学求解器类
    
    提供统一的接口进行正运动学计算
    """
    
    def __init__(self, params: FingerParams):
        self.params = params
    
    def solve_T0(self, q3: float) -> Tuple[np.ndarray, np.ndarray, FKInfo]:
        """
        计算 q1=0, q2=0 时的末端位姿
        
        Args:
            q3: 关节角 q3 (度)
        
        Returns:
            T_pos, T_rot, info
        """
        return fk_T0(q3, self.params)
    
    def solve_chain(self, delta_d1: float, delta_d2: float, q3: float) -> Tuple[np.ndarray, np.ndarray, FKInfo]:
        """
        计算给定输入下的末端位姿
        
        Args:
            delta_d1: d1 变化量 (mm)
            delta_d2: d2 变化量 (mm)
            q3: 关节角 q3 (度)
        
        Returns:
            T_pos, T_rot, info
        """
        return fk_chain_to_T(delta_d1, delta_d2, q3, self.params)
