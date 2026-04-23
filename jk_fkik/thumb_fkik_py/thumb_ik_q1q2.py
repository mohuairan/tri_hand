#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拇指机构逆运动学 - q1, q2 求解

功能：根据目标位置和约束平面法向量，求解基座两个关节的角度
方法：直接从法向量解析求解 q1, q2
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

# 支持两种运行方式：相对导入（作为模块）和绝对导入（直接运行）
try:
    from .params import ThumbParams
except ImportError:
    from params import ThumbParams


@dataclass
class IKQ1Q2Info:
    """逆解 q1/q2 信息结构体"""
    status: int = 0                        # 求解状态 (0=成功，1=无解，2=多解/警告)
    error_msg: str = ''                    # 错误/状态信息
    n_c: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 输入的法向量（可能已修正）
    phi: float = 0.0                       # 约束平面夹角（中间输出，度）
    q1_all: List[float] = field(default_factory=list)  # 所有可行的 q1 解
    q2_all: List[float] = field(default_factory=list)  # 所有可行的 q2 解
    all_candidates: List[Tuple[float, float, bool]] = field(default_factory=list)  # 所有候选解
    limit_exceeded: bool = False           # 关节限位超限标志
    exceed_joint: List[str] = field(default_factory=list)  # 超限关节名称列表
    correction: Dict[str, Any] = field(default_factory=lambda: {
        'applied': False, 'nx': 0, 'ny': 0, 'nz': 0, 'method': 'none'
    })


def thumb_ik_q1q2(P_target: np.ndarray, n_c: np.ndarray, 
                  params: Optional[ThumbParams] = None) -> Tuple[Optional[float], Optional[float], IKQ1Q2Info]:
    """
    求解拇指机构的 q1 和 q2（约束平面姿态）
    
    功能：根据目标位置和约束平面法向量，求解基座两个关节的角度
    方法：直接从法向量解析求解 q1, q2
    
    参数:
        P_target : 3x1 向量，末端目标位置 [x; y; z] (mm)
        n_c      : 3x1 向量，约束平面法向量（单位向量）
                   n_c = [-sin(q2); cos(q1)*cos(q2); sin(q1)*cos(q2)]
        params   : 参数对象
    
    返回:
        q1       : 关节 1 角度 (度)
        q2       : 关节 2 角度 (度)
        info     : IKQ1Q2Info 对象，包含求解状态和中间信息
    
    数学原理:
        法向量定义：n_c = R_01 * [0; 1; 0]
        展开得：n_c = [-sin(q2); cos(q1)*cos(q2); sin(q1)*cos(q2)]
        
        因此：
        - sin(q2) = -n_c[0]
        - q2 = asin(-n_c[0])
        - q1 = atan2(n_c[2], n_c[1])
    """
    if params is None:
        params = ThumbParams()
    
    # 初始化输出
    q1 = None
    q2 = None
    info = IKQ1Q2Info()
    
    # 输入验证
    if P_target is None or n_c is None:
        info.status = 1
        info.error_msg = '输入参数不足：需要 P_target 和 n_c'
        return q1, q2, info
    
    if len(P_target) != 3:
        info.status = 1
        info.error_msg = 'P_target 必须是 3x1 向量 [x; y; z]'
        return q1, q2, info
    
    if len(n_c) != 3:
        info.status = 1
        info.error_msg = 'n_c 必须是 3x1 向量 [nx; ny; nz]'
        return q1, q2, info
    
    # 法向量预处理和验证
    nx_orig, ny_orig, nz_orig = n_c.copy()
    
    # 检查法向量是否为单位向量
    norm_nc = np.linalg.norm(n_c)
    if abs(norm_nc - 1.0) > params.tol_normal * 10:
        print(f'\n[警告] 法向量模长={norm_nc:.6f}，进行归一化处理')
        n_c = n_c / norm_nc
        info.correction['applied'] = True
        info.correction['method'] = 'normalize'
    
    # 检查法向量 X 分量
    if n_c[0] < -params.tol_normal:
        # 翻转法向，同一平面的另一个法向方向
        n_c = -n_c
        info.correction['applied'] = True
        info.correction['method'] = 'flip_normal'
    
    # 保存归一化后的法向量
    info.n_c = n_c
    nx, ny, nz = n_c
    
    # 从法向量直接求解 q1, q2
    # 步骤 1: 求解 q2
    sin_q2 = -nx
    
    # 检查 sin(q2) 是否在有效范围内
    if abs(sin_q2) > 1.0 + params.tol_normal:
        info.status = 1
        info.error_msg = f'法向量 X 分量超出范围 (nx={nx:.4f})'
        return q1, q2, info
    
    # 截断到 [-1, 1] 避免数值误差
    sin_q2 = np.clip(sin_q2, -1.0, 1.0)
    
    # q2 = asin(-n_c[0])
    q2_rad = np.arcsin(sin_q2)
    q2 = np.degrees(q2_rad)
    
    # 步骤 2: 求解 q1
    cos_q2 = np.cos(q2_rad)
    
    if abs(cos_q2) < params.tol_normal:
        info.status = 1
        info.error_msg = f'q2 接近±90°，cos(q2)={cos_q2:.6f} 过小'
        return q1, q2, info
    
    # 使用 atan2 直接求解 q1
    q1_rad = np.arctan2(nz, ny)
    q1 = np.degrees(q1_rad)
    
    # 步骤 3: 计算平面夹角 phi（中间输出）
    cos_phi = abs(np.cos(q1_rad) * np.cos(q2_rad))
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    info.phi = np.degrees(np.arccos(cos_phi))
    
    # 关节限位检查
    q1_solutions = []
    q2_solutions = []
    
    in_limit = True
    skip_reason = ''
    
    if q2 < params.q2_limit_min - params.limit_tolerance or \
       q2 > params.q2_limit_max + params.limit_tolerance:
        skip_reason = f'q2 超限 ({q2:.2f}° 不在 [{params.q2_limit_min:.2f}°, {params.q2_limit_max:.2f}°])'
        in_limit = False
    elif q1 < params.q1_limit_min - params.limit_tolerance or \
         q1 > params.q1_limit_max + params.limit_tolerance:
        skip_reason = f'q1 超限 ({q1:.2f}° 不在 [{params.q1_limit_min:.2f}°, {params.q1_limit_max:.2f}°])'
        in_limit = False
    
    # 记录候选解
    info.all_candidates = [(q1, q2, in_limit)]
    
    if in_limit:
        q1_solutions = [q1]
        q2_solutions = [q2]
    
    # 检查是否有可行解
    if not q1_solutions:
        info.status = 1
        info.error_msg = f'无解：{skip_reason}'
        return q1, q2, info
    
    # 输出结果
    info.q1_all = q1_solutions
    info.q2_all = q2_solutions
    
    # 单解情况直接输出
    q1 = info.q1_all[0]
    q2 = info.q2_all[0]
    
    # 检查最终解是否超限
    info.exceed_joint = []
    if q1 < params.q1_limit_min - params.limit_tolerance or q1 > params.q1_limit_max + params.limit_tolerance:
        info.exceed_joint.append('q1')
    if q2 < params.q2_limit_min - params.limit_tolerance or q2 > params.q2_limit_max + params.limit_tolerance:
        info.exceed_joint.append('q2')
    
    if info.exceed_joint:
        info.limit_exceeded = True
        info.status = 2
        info.error_msg = f'⚠ 警告：关节 {", ".join(info.exceed_joint)} 超出限位'
    else:
        info.error_msg = '求解成功'
    
    # 记录法向量修正信息
    if info.correction['applied'] and info.correction['method'] == 'normalize':
        info.correction['nx'] = float(nx - nx_orig)
        info.correction['ny'] = float(ny - ny_orig)
        info.correction['nz'] = float(nz - nz_orig)
    
    return q1, q2, info
