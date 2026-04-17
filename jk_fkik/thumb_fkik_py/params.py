#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拇指机构参数定义

包含拇指正逆运动学求解所需的所有结构常数和配置参数
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ThumbParams:
    """拇指机构参数类
    
    包含拇指正逆运动学求解所需的所有结构常数和配置参数
    """
    
    # ==================== 结构常数 ====================
    # 这些参数由机械设计决定，单位：mm 或 度
    
    # 固定偏角 (度)
    alpha1: float = -3.65      # 基座固定偏角
    alpha2: float = 8.22       # M 系固定偏角  
    alpha3: float = -16.72     # N 系固定偏角
    
    # 连杆长度 (mm)
    L1: float = field(default_factory=lambda: np.hypot(23.51, 1.5))  # OM 长度 ≈ 23.56 mm
    L2: float = 53.17          # MN 长度 (mm)
    L3: float = 25.57          # NT 长度 (mm)
    
    # ==================== 关节限位 (度) ====================
    # 各关节的物理运动范围限制
    
    # q1: 基座关节 1
    q1_limit_min: float = -95
    q1_limit_max: float = 0
    
    # q2: 基座关节 2
    q2_limit_min: float = -55
    q2_limit_max: float = 0
    
    # q3: 中间关节
    q3_limit_min: float = -86
    q3_limit_max: float = 0
    
    # q4: 末端关节
    q4_limit_min: float = -70
    q4_limit_max: float = 0
    
    # 关节限位检查容差 (度)
    limit_tolerance: float = 0.5
    
    # ==================== 逆解容差参数 ====================
    # 用于逆解求解时的数值容差和目标点修正范围
    
    # 位置容差 (mm)
    tol_x: float = 0.04
    tol_y: float = 0.04
    tol_z: float = 0.04
    
    # 角度容差 (度)
    tol_phi: float = 0.5
    
    # 法向量容差
    # 根据 tol_phi=0.5° 计算：sin(0.5°) ≈ 0.0087
    tol_normal: float = 0.01
    
    # 混合策略阈值
    delta_analytic_limit: float = -5000   # delta > 此值时用解析法
    delta_hard_limit: float = -80000      # delta < 此值时直接报错
    
    # ==================== 验证容差 ====================
    # 用于正逆解验证的误差阈值
    
    # 位置验证容差 (mm)
    position_tolerance: float = 0.01
    
    # 共面性验证容差
    coplanar_tolerance: float = 1e-6
    
    # d_y 验证容差 (MT 向量在{1}系中的 Y 分量)
    dy_tolerance: float = 1e-6
    
    # ==================== 平面夹角范围 ====================
    # phi 的有效范围 (度)
    phi_min: float = 0
    phi_max: float = 90
