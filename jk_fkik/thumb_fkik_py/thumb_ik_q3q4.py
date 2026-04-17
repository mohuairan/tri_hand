#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拇指机构逆运动学 - q3, q4 求解

功能：在已知 q1、q2 的基础上，求解平面内两关节的角度
方法：几何法解析解（余弦定理 + 三角关系）
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

# 支持两种运行方式：相对导入（作为模块）和绝对导入（直接运行）
try:
    from .params import ThumbParams
except ImportError:
    from params import ThumbParams
import warnings


@dataclass
class IKQ3Q4Info:
    """逆解 q3/q4 信息结构体"""
    status: int = 0                        # 求解状态 (0=成功，1=无解，2=多解/警告)
    error_msg: str = ''                    # 错误/状态信息
    P_M: np.ndarray = field(default_factory=lambda: np.zeros(3))  # M 点位置
    P_N: np.ndarray = field(default_factory=lambda: np.zeros(3))  # N 点位置
    d_x: float = 0.0                       # MT 在{1}系 X 分量
    d_y: float = 0.0                       # MT 在{1}系 Y 分量
    d_z: float = 0.0                       # MT 在{1}系 Z 分量
    r: float = 0.0                         # MT 在平面内投影长度
    q3_all: List[float] = field(default_factory=list)  # 所有可行的 q3 解
    q4_all: List[float] = field(default_factory=list)  # 所有可行的 q4 解
    limit_exceeded: bool = False           # 关节限位超限标志
    exceed_joint: List[str] = field(default_factory=list)  # 超限关节名称列表


def thumb_ik_q3q4(P_target: np.ndarray, q1: float, q2: float, 
                  params: Optional[ThumbParams] = None) -> Tuple[Optional[float], Optional[float], IKQ3Q4Info]:
    """
    求解拇指机构的 q3 和 q4（平面内 2R 逆解）
    
    功能：在已知 q1、q2 的基础上，求解平面内两关节的角度
    方法：几何法解析解（余弦定理 + 三角关系）
    
    参数:
        P_target : 3x1 向量，末端目标位置 [x; y; z] (mm)
        q1       : 关节 1 角度 (度)
        q2       : 关节 2 角度 (度)
        params   : 参数对象
    
    返回:
        q3       : 关节 3 角度 (度)
        q4       : 关节 4 角度 (度)
        info     : IKQ3Q4Info 对象，包含求解状态和中间信息
    """
    if params is None:
        params = ThumbParams()
    
    # 初始化输出
    q3 = None
    q4 = None
    info = IKQ3Q4Info()
    
    # 输入验证
    if P_target is None:
        info.status = 1
        info.error_msg = '输入参数不足：需要 P_target, q1, q2'
        return q3, q4, info
    
    if q1 is None or q2 is None:
        info.status = 1
        info.error_msg = 'q1 或 q2 为 None，请先求解 q1, q2'
        return q3, q4, info
    
    # 角度预处理
    deg2rad = np.pi / 180.0
    alpha1_rad = params.alpha1 * deg2rad
    alpha2_rad = params.alpha2 * deg2rad
    alpha3_rad = params.alpha3 * deg2rad
    
    # 计算基座旋转矩阵 R_01
    c1, s1 = np.cos(np.radians(q1)), np.sin(np.radians(q1))
    c2, s2 = np.cos(np.radians(q2)), np.sin(np.radians(q2))
    ca1, sa1 = np.cos(alpha1_rad), np.sin(alpha1_rad)
    
    R_01 = np.array([
        [c2*ca1, -s2, c2*sa1],
        [c1*s2*ca1 + s1*sa1, c1*c2, c1*s2*sa1 - s1*ca1],
        [s1*s2*ca1 - c1*sa1, s1*c2, s1*s2*sa1 + c1*ca1]
    ])
    
    # 计算 M 点位置
    info.P_M = R_01 @ np.array([0, 0, params.L1])
    
    # 计算 MT 向量在{1}系中的坐标
    MT = P_target - info.P_M
    MT_local = R_01.T @ MT  # R_01.T 是逆换乘
    
    info.d_x = float(MT_local[0])  # {1}系 X 分量
    info.d_y = float(MT_local[1])  # {1}系 Y 分量 (理论应为 0)
    info.d_z = float(MT_local[2])  # {1}系 Z 分量
    
    # d_y 验证
    if abs(info.d_y) > params.dy_tolerance:
        warnings.warn(f'thumb_ik_q3q4: d_y={info.d_y:.6f}，q1/q2 可能存在误差')
    
    # MT 在平面内的投影长度
    info.r = np.sqrt(info.d_x**2 + info.d_z**2)
    
    # 检查可达性
    r_min = abs(params.L2 - params.L3)  # 最小可达距离
    r_max = params.L2 + params.L3        # 最大可达距离
    
    if info.r > r_max:
        info.status = 1
        info.error_msg = f'目标点超出最大工作空间 (距离={info.r:.2f}mm > 最大={r_max:.2f}mm)'
        return q3, q4, info
    
    if info.r < r_min:
        info.status = 1
        info.error_msg = f'目标点超出最小工作空间 (距离={info.r:.2f}mm < 最小={r_min:.2f}mm)'
        return q3, q4, info
    
    # 解析求解 q3, q4
    q3_solutions = []
    q4_solutions = []
    
    # 步骤 1: 用余弦定理求关节 4 的角度
    cos_theta4 = (info.r**2 - params.L2**2 - params.L3**2) / (2 * params.L2 * params.L3)
    cos_theta4 = np.clip(cos_theta4, -1.0, 1.0)
    
    # 两个可能的 θ4 解 (正负肘部配置)
    theta4_solutions = [np.arccos(cos_theta4), -np.arccos(cos_theta4)]
    
    # 步骤 2: 计算∠NMT
    cos_NMT = (params.L2**2 + info.r**2 - params.L3**2) / (2 * params.L2 * info.r)
    cos_NMT = np.clip(cos_NMT, -1.0, 1.0)
    angle_NMT = np.arccos(cos_NMT)
    
    # 步骤 3: 计算 MT 的方向角 β
    beta = np.arctan2(info.d_x, info.d_z)
    
    # 步骤 4: 遍历所有θ4 解，计算对应的 q3, q4
    for theta4 in theta4_solutions:
        # q4 = θ4 - α3
        q4_candidate = np.degrees(theta4) - params.alpha3
        
        # 根据θ4 的正负确定θ3 的计算方式
        if theta4 >= 0:
            theta3 = beta - angle_NMT
        else:
            theta3 = beta + angle_NMT
        
        # q3 = θ3 - α2
        q3_candidate = np.degrees(theta3) - params.alpha2
        
        # 关节限位检查
        if (params.q3_limit_min - params.limit_tolerance <= q3_candidate <= params.q3_limit_max + params.limit_tolerance and
            params.q4_limit_min - params.limit_tolerance <= q4_candidate <= params.q4_limit_max + params.limit_tolerance):
            q3_solutions.append(q3_candidate)
            q4_solutions.append(q4_candidate)
    
    # 处理求解结果
    if not q3_solutions:
        info.status = 1
        info.error_msg = '无解：所有候选解均超出关节限位'
        return q3, q4, info
    
    info.q3_all = q3_solutions
    info.q4_all = q4_solutions
    
    # 选择最优解（关节角绝对值和最小）
    if len(info.q3_all) > 1:
        info.status = 2
        info.error_msg = f'多解：共{len(info.q3_all)}组可行解，已选择最优解'
        cost = [abs(q3) + abs(q4) for q3, q4 in zip(info.q3_all, info.q4_all)]
        best_idx = cost.index(min(cost))
        q3 = info.q3_all[best_idx]
        q4 = info.q4_all[best_idx]
    else:
        info.status = 0
        info.error_msg = '求解成功'
        q3 = info.q3_all[0]
        q4 = info.q4_all[0]
    
    # 检查最终解是否超限
    info.exceed_joint = []
    if q3 < params.q3_limit_min or q3 > params.q3_limit_max:
        info.exceed_joint.append('q3')
    if q4 < params.q4_limit_min or q4 > params.q4_limit_max:
        info.exceed_joint.append('q4')
    
    if info.exceed_joint:
        info.limit_exceeded = True
        info.status = 2
        info.error_msg = f'⚠ 警告：关节 {", ".join(info.exceed_joint)} 超出限位'
    
    # 计算 N 点位置
    theta3 = (params.alpha2 + q3) * deg2rad
    v2_local = np.array([params.L2 * np.sin(theta3), 0, params.L2 * np.cos(theta3)])
    info.P_N = R_01 @ (np.array([0, 0, params.L1]) + v2_local)
    
    return q3, q4, info
