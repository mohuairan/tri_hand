"""
inverse_kinematics.py - 逆运动学求解

实现手指机构的逆运动学计算：
- ik_d1_d2: 根据 q1, q2 计算并联机构 d1, d2 的变化量
- ik_q3_from_PT: 根据 P, T 坐标求解 q3
- ik_solve_q1q2_byT0: 由旋转后 T 点坐标反推 q1, q2
- ik_q1q2_new: 已知 T 点坐标求解 q1, q2
"""

import math
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

from .params import FingerParams
from . import utils
from .q3_to_q4 import q3_to_q4
from .forward_kinematics import fk_T0


def ik_d1_d2(q1: float, q2: float, params: FingerParams) -> Tuple[
    float, float, float, float, float, float, Dict]:
    """
    根据 q1、q2 计算并联机构 d1、d2 的变化量
    
    Args:
        q1: 屈伸角 (度)
        q2: 外展角 (度)
        params: 手指机构参数
    
    Returns:
        d1_init: 初始状态 d1 值 (mm)
        d2_init: 初始状态 d2 值 (mm)
        d1_new: 旋转后 d1 值 (mm)
        d2_new: 旋转后 d2 值 (mm)
        delta_d1: d1 变化量 (mm)
        delta_d2: d2 变化量 (mm)
        info: 详细信息字典
    """
    # ========== 加载参数 ==========
    ax = params.ax
    ay = params.ay
    bx = params.bx
    by = params.by
    bz = params.bz
    px, py, pz = params.P
    l1 = params.l1
    l2 = params.l2
    
    info = {'success': False, 'error': None, 'q1': q1, 'q2': q2}
    
    # ========== 第一步：计算初始状态下的 d1、d2 ==========
    # 初始 B 点坐标 (未旋转时)
    X_B1_init = px + bx
    Y_B1_init = py + by
    Z_B1_init = pz + bz
    
    X_B2_init = px - bx
    Y_B2_init = py + by
    Z_B2_init = pz + bz
    
    # 初始水平距离平方
    H1_sq_init = (X_B1_init - ax)**2 + (Y_B1_init - ay)**2
    H2_sq_init = (X_B2_init + ax)**2 + (Y_B2_init - ay)**2
    
    # 检查可行性
    if l1**2 < H1_sq_init or l2**2 < H2_sq_init:
        info['success'] = False
        info['error'] = '初始状态不可达：杆长不足'
        return None, None, None, None, None, None, info
    
    # 求解初始 d1、d2 (取 C 在 B 上方的解)
    d1_init = Z_B1_init - math.sqrt(l1**2 - H1_sq_init)
    d2_init = Z_B2_init - math.sqrt(l2**2 - H2_sq_init)
    
    # ========== 第二步：计算旋转后的 B 点坐标 ==========
    q1_rad = utils.deg2rad(q1)
    q2_rad = utils.deg2rad(q2)
    
    c1 = math.cos(q1_rad)
    s1 = math.sin(q1_rad)
    c2 = math.cos(q2_rad)
    s2 = math.sin(q2_rad)
    
    # 计算旋转后的向量 b1' = R * [bx, by, bz]'
    b1x_prime = c2 * bx + s2 * s1 * by + s2 * c1 * bz
    b1y_prime = c1 * by - s1 * bz
    b1z_prime = -s2 * bx + c2 * s1 * by + c2 * c1 * bz
    
    # 计算旋转后的向量 b2' = R * [-bx, by, bz]'
    b2x_prime = -c2 * bx + s2 * s1 * by + s2 * c1 * bz
    b2y_prime = c1 * by - s1 * bz  # 与 b1y_prime 相同
    b2z_prime = s2 * bx + c2 * s1 * by + c2 * c1 * bz
    
    # 新 B 点坐标
    X_B1_new = px + b1x_prime
    Y_B1_new = py + b1y_prime
    Z_B1_new = pz + b1z_prime
    
    X_B2_new = px + b2x_prime
    Y_B2_new = py + b2y_prime
    Z_B2_new = pz + b2z_prime
    
    # ========== 第三步：计算新的 d1'、d2' ==========
    H1_sq_new = (X_B1_new - ax)**2 + (Y_B1_new - ay)**2
    H2_sq_new = (X_B2_new + ax)**2 + (Y_B2_new - ay)**2
    
    # 检查可行性
    if l1**2 < H1_sq_new or l2**2 < H2_sq_new:
        info['success'] = False
        info['error'] = '旋转后状态不可达：杆长不足'
        return d1_init, d2_init, None, None, None, None, info
    
    # 求解新 d1'、d2' (符号必须与初始状态一致)
    d1_new = Z_B1_new - math.sqrt(l1**2 - H1_sq_new)
    d2_new = Z_B2_new - math.sqrt(l2**2 - H2_sq_new)
    
    # ========== 第四步：计算变化量 ==========
    delta_d1 = d1_new - d1_init
    delta_d2 = d2_new - d2_init
    
    # ========== 信息输出 ==========
    info['success'] = True
    info['H1_sq_init'] = H1_sq_init
    info['H2_sq_init'] = H2_sq_init
    info['H1_sq_new'] = H1_sq_new
    info['H2_sq_new'] = H2_sq_new
    
    return d1_init, d2_init, d1_new, d2_new, delta_d1, delta_d2, info


def ik_q3_from_PT(P: np.ndarray, T: np.ndarray, params: FingerParams, 
                  q3_init: float = 0.0) -> Tuple[Optional[float], Dict]:
    """
    根据 P、T 三维坐标求解 q3
    
    Args:
        P: P 点坐标 [Px, Py, Pz] (mm)
        T: T 点坐标 [Tx, Ty, Tz] (mm)
        params: 手指机构参数
        q3_init: 迭代初值（度）
    
    Returns:
        q3: 求解角度（度）
        info: 求解状态信息字典
    """
    info = {
        'success': False,
        'error': None,
        'D_PT': None,
        'residual': None,
        'q4': None
    }
    
    # ========== 从 params 读取参数 ==========
    L_PM = params.L_PM
    L_MN = params.L_MN
    L_NT = params.L_NT
    THETA1_0 = params.theta3_init
    THETA2_0 = params.theta4_init
    tolerance = params.tolerance
    
    # ========== 目标距离 ==========
    P_arr = np.array(P)
    T_arr = np.array(T)
    D_PT = np.linalg.norm(T_arr - P_arr)
    
    info['D_PT'] = D_PT
    
    # ========== 定义残差函数 ==========
    def compute_residual(q3_deg: float) -> float:
        """计算残差"""
        # 调用耦合函数求 q4
        q4_deg = q3_to_q4(q3_deg, params)
        
        # 检查 q4 是否有效
        if q4_deg is None:
            return 1e10  # 返回大值表示不可达
        
        # 计算绝对角度
        theta1 = THETA1_0 + q3_deg
        theta2 = THETA2_0 + q4_deg
        
        # 平面链长计算（局部坐标系）
        theta1_rad = utils.deg2rad(theta1)
        theta2_rad = utils.deg2rad(theta2)
        
        x_end = L_PM + L_MN * math.cos(theta1_rad) + L_NT * math.cos(theta1_rad + theta2_rad)
        y_end = L_MN * math.sin(theta1_rad) + L_NT * math.sin(theta1_rad + theta2_rad)
        
        # 残差 = 计算长度 - 目标长度
        F = math.sqrt(x_end**2 + y_end**2) - D_PT
        return F
    
    # ========== 求解 ==========
    result, success = utils.numerical_solve_1d(compute_residual, q3_init, tolerance)
    
    if success and result is not None:
        q3 = result
        info['success'] = True
        info['residual'] = compute_residual(q3)
        info['q4'] = q3_to_q4(q3, params)
        return q3, info
    else:
        info['error'] = 'q3 数值求解失败'
        info['residual'] = None
        return None, info


def apply_rotation(v: np.ndarray, q1_rad: float, q2_rad: float) -> np.ndarray:
    """
    对向量 v 施加旋转：先 Rx(q1)，再 Ry(q2)
    
    Args:
        v: 输入向量
        q1_rad: 绕 x 轴角度（弧度）
        q2_rad: 绕 y 轴角度（弧度）
    
    Returns:
        旋转后向量
    """
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(q1_rad), -math.sin(q1_rad)],
        [0, math.sin(q1_rad), math.cos(q1_rad)]
    ])
    
    Ry = np.array([
        [math.cos(q2_rad), 0, math.sin(q2_rad)],
        [0, 1, 0],
        [-math.sin(q2_rad), 0, math.cos(q2_rad)]
    ])
    
    return Ry @ Rx @ v


def ik_solve_q1q2_byT0(P: np.ndarray, T0: np.ndarray, 
                       T_current: np.ndarray) -> Tuple[float, float, float]:
    """
    由旋转后 T 点坐标反推旋转角 q1, q2
    
    Args:
        P: 旋转中心坐标（P 点固定不动）
        T0: q1=q2=0 时 T 的坐标
        T_current: 旋转后 T 的坐标
    
    Returns:
        q1_deg: 绕 x 轴旋转角（度）
        q2_deg: 绕 y 轴旋转角（度）
        error: 重建误差
    """
    # ========== 角度范围定义（度）==========
    Q1_MIN = -90.0
    Q1_MAX = 15.0
    Q2_MIN = -45.0
    Q2_MAX = 45.0
    
    # ========== 统一转为列向量 ==========
    P = np.array(P).flatten()
    T0 = np.array(T0).flatten()
    T_current = np.array(T_current).flatten()
    
    # ========== 第 1 步：构造相对 P 点的 PT 向量 ==========
    v0 = T0 - P      # 初始 PT 向量
    vp = T_current - P  # 旋转后 PT 向量
    
    a, b, c = v0
    ap, bp, cp = vp
    
    # ========== 第 2 步：校验刚体约束 ==========
    norm0 = np.linalg.norm(v0)
    normp = np.linalg.norm(vp)
    
    if abs(norm0 - normp) / (norm0 + 1e-12) > 1e-4:
        raise ValueError(f'向量模长不一致：|v0|={norm0:.6f}, |vp|={normp:.6f}')
    
    # ========== 第 3 步：求 q1 的两个候选值 ==========
    rho = math.sqrt(b**2 + c**2)
    
    q1_candidates = []
    
    if rho < 1e-12:
        # b=c=0，q1 不可观测
        q1_candidates = [0.0]
    else:
        cos_val = bp / rho
        cos_val = max(-1.0, min(1.0, cos_val))  # 防止浮点误差越界
        phi = math.atan2(c, b)
        delta = math.acos(cos_val)
        
        # 两个候选解
        q1_candidates = [-phi + delta, -phi - delta]
    
    # ========== 第 4 步：对每个 q1 候选值，解析求 q2，并筛选范围 ==========
    valid_q1 = []
    valid_q2 = []
    valid_err = []
    
    for k, q1_raw in enumerate(q1_candidates):
        # 归一化到 (-π, π] 对应的弧度范围
        q1_rad = utils.wrap_to_pi(q1_raw)
        q1_d = utils.rad2deg(q1_rad)
        
        # 检查 q1 是否在允许范围内
        if q1_d < Q1_MIN or q1_d > Q1_MAX:
            continue
        
        # 用原始 q1（未截断）计算中间量
        ux = a
        uz = b * math.sin(q1_rad) + c * math.cos(q1_rad)
        
        # 解析求 q2
        q2_rad = math.atan2(uz, ux) - math.atan2(cp, ap)
        q2_rad = utils.wrap_to_pi(q2_rad)
        q2_d = utils.rad2deg(q2_rad)
        
        # 检查 q2 是否在允许范围内
        if q2_d < Q2_MIN or q2_d > Q2_MAX:
            continue
        
        # 计算重建误差
        v_recon = apply_rotation(v0, q1_rad, q2_rad)
        err = np.linalg.norm(v_recon - vp)
        
        valid_q1.append(q1_d)
        valid_q2.append(q2_d)
        valid_err.append(err)
    
    # ========== 第 5 步：从有效解中选误差最小的作为输出 ==========
    if not valid_err:
        raise ValueError('没有找到满足角度范围约束的有效解！')
    
    best_idx = int(np.argmin(valid_err))
    q1_deg = valid_q1[best_idx]
    q2_deg = valid_q2[best_idx]
    error = valid_err[best_idx]
    
    return q1_deg, q2_deg, error


def ik_q1q2_new(T: np.ndarray, params: FingerParams) -> Tuple[
    List[float], List[float], Dict]:
    """
    已知 T 点坐标，解析求解 q1、q2
    
    Args:
        T: 目标点坐标 [Tx, Ty, Tz]
        params: 手指机构参数
    
    Returns:
        q1_solutions: q1 解列表（度）
        q2_solutions: q2 解列表（度）
        info: 详细信息字典
    """
    info = {
        'success': False,
        'error': None,
        'q3': None,
        'q4': None,
        'num_solutions': 0
    }
    
    P = np.array(params.P)
    T_arr = np.array(T)
    
    # ========== 第一步：求解 q3、q4 ==========
    q3, info_q3 = ik_q3_from_PT(P, T_arr, params, 0)
    
    if not info_q3['success'] or q3 is None:
        info['error'] = 'q3 求解失败：' + str(info_q3.get('error', ''))
        return [], [], info
    
    q4 = q3_to_q4(q3, params)
    
    if q4 is None:
        info['error'] = 'q4 求解失败：四连杆机构无解'
        return [], [], info
    
    # ========== 第二步：计算 T0 ==========
    T0, _, info_fk = fk_T0(q3, params)
    
    if not info_fk.success:
        info['error'] = 'fk_T0 计算失败'
        return [], [], info
    
    # ========== 第三步：求解 q1, q2 ==========
    try:
        q1_deg, q2_deg, error = ik_solve_q1q2_byT0(P, T0, T_arr)
    except ValueError as e:
        info['error'] = str(e)
        return [], [], info
    
    # ========== 信息输出 ==========
    info['success'] = True
    info['q3'] = q3
    info['q4'] = q4
    info['num_solutions'] = 1
    
    return [q1_deg], [q2_deg], info


class InverseKinematics:
    """
    逆运动学求解器类
    
    提供统一的接口进行逆运动学计算
    """
    
    def __init__(self, params: FingerParams):
        self.params = params
    
    def solve_d1_d2(self, q1: float, q2: float) -> Tuple[
        float, float, float, float, float, float, Dict]:
        """
        根据 q1, q2 计算 d1, d2 变化量
        
        Args:
            q1: 屈伸角 (度)
            q2: 外展角 (度)
        
        Returns:
            d1_init, d2_init, d1_new, d2_new, delta_d1, delta_d2, info
        """
        return ik_d1_d2(q1, q2, self.params)
    
    def solve_q1q2(self, T: np.ndarray) -> Tuple[List[float], List[float], Dict]:
        """
        根据目标点 T 求解 q1, q2
        
        Args:
            T: 目标点坐标 [Tx, Ty, Tz]
        
        Returns:
            q1_solutions, q2_solutions, info
        """
        return ik_q1q2_new(T, self.params)
    
    def solve_q3(self, P: np.ndarray, T: np.ndarray, 
                 q3_init: float = 0.0) -> Tuple[Optional[float], Dict]:
        """
        根据 P, T 坐标求解 q3
        
        Args:
            P: P 点坐标
            T: T 点坐标
            q3_init: 迭代初值
        
        Returns:
            q3, info
        """
        return ik_q3_from_PT(P, T, self.params, q3_init)
