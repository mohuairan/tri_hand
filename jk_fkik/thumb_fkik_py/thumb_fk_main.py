#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拇指机构正运动学

功能：根据给定的四个关节角度，计算末端执行器的位置和姿态
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

# 支持两种运行方式：相对导入（作为模块）和绝对导入（直接运行）
try:
    from .params import ThumbParams
except ImportError:
    from params import ThumbParams


@dataclass
class FKInfo:
    """正解信息结构体"""
    P_M: np.ndarray = field(default_factory=lambda: np.zeros(3))      # M 点位置 (3x1 向量)
    P_N: np.ndarray = field(default_factory=lambda: np.zeros(3))      # N 点位置 (3x1 向量)
    n_c: np.ndarray = field(default_factory=lambda: np.zeros(3))      # OMNT 平面法向量 (单位向量)
    phi: float = 0.0                                                   # 平面与基座 X-Z 平面夹角 (度)


def thumb_fk_main(q1: float, q2: float, q3: float, q4: float, 
                  params: Optional[ThumbParams] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FKInfo]:
    """
    计算灵巧手拇指机构的正运动学
    
    功能：根据给定的四个关节角度，计算末端执行器的位置和姿态
    
    参数:
        q1, q2, q3, q4 : 关节角度 (单位：度)
            q1: 基座关节 1 角度（绕 X 轴旋转）
            q2: 基座关节 2 角度（绕 Z 轴旋转）
            q3: 中间关节角度
            q4: 末端关节角度
        params: 参数对象，如果为 None 则使用默认参数
    
    返回:
        T_end : 4x4 齐次变换矩阵，包含位置和姿态信息
        P_end : 3x1 位置向量 [x; y; z] (mm)
        R_end : 3x3 旋转矩阵，描述末端姿态
        info  : FKInfo 对象，包含中间点和平面信息
            info.P_M      : M 点位置 (3x1 向量)
            info.P_N      : N 点位置 (3x1 向量)
            info.n_c      : OMNT 平面法向量 (单位向量)
            info.phi      : 平面与基座 X-Z 平面夹角 (度)
    
    示例:
        T, P, R, info = thumb_fk_main(-1.85, -12.13, -82.04, -63.65)
    """
    if params is None:
        params = ThumbParams()
    
    # 角度预处理 - 转换为弧度
    deg2rad = np.pi / 180.0
    th1 = q1 * deg2rad                          # q1 弧度
    th2 = q2 * deg2rad                          # q2 弧度
    th3 = (params.alpha2 + q3) * deg2rad        # 关节 3 总角度 (含固定偏角)
    th4 = (params.alpha3 + q4) * deg2rad        # 关节 4 总角度 (含固定偏角)
    th1_base = params.alpha1 * deg2rad          # 基座固定偏角弧度
    
    # 计算基座旋转矩阵 R_01
    Rx1 = np.array([
        [1, 0, 0],
        [0, np.cos(th1), -np.sin(th1)],
        [0, np.sin(th1), np.cos(th1)]
    ])
    
    Rz2 = np.array([
        [np.cos(th2), -np.sin(th2), 0],
        [np.sin(th2), np.cos(th2), 0],
        [0, 0, 1]
    ])
    
    Ry_alpha1 = np.array([
        [np.cos(th1_base), 0, np.sin(th1_base)],
        [0, 1, 0],
        [-np.sin(th1_base), 0, np.cos(th1_base)]
    ])
    
    # 复合旋转矩阵
    R_01 = Rx1 @ Rz2 @ Ry_alpha1
    
    # 计算局部平面向量 ({1}系 X-Z 平面内)
    v1 = np.array([0, 0, params.L1])                    # OM 向量
    v2_local = np.array([params.L2 * np.sin(th3), 0, params.L2 * np.cos(th3)])  # MN 向量
    v3_local = np.array([params.L3 * np.sin(th3 + th4), 0, params.L3 * np.cos(th3 + th4)])  # NT 向量
    
    # 局部坐标系中的末端位置
    P_local_1 = v1 + v2_local + v3_local
    
    # 计算全局位置
    P_end = R_01 @ P_local_1
    
    # 计算中间点位置
    P_M = R_01 @ v1
    P_N = R_01 @ (v1 + v2_local)
    
    # 计算末端姿态
    Ry_total = np.array([
        [np.cos(th3 + th4), 0, np.sin(th3 + th4)],
        [0, 1, 0],
        [-np.sin(th3 + th4), 0, np.cos(th3 + th4)]
    ])
    R_end = R_01 @ Ry_total
    
    # 构建齐次变换矩阵
    T_end = np.eye(4)
    T_end[0:3, 0:3] = R_end
    T_end[0:3, 3] = P_end
    
    # 计算 OMNT 平面法向量和夹角
    # 法向量 n_c = R_01 * [0; 1; 0]
    n_c = R_01 @ np.array([0, 1, 0])
    n_c = n_c / np.linalg.norm(n_c)  # 单位化
    
    # 平面与基座 X-Z 平面夹角 phi
    # phi = acos(n_c · y_axis) = acos(n_c[1])
    phi = np.arccos(np.abs(n_c[1])) * 180.0 / np.pi  # 转换为度
    
    # 创建信息对象
    info = FKInfo(P_M=P_M, P_N=P_N, n_c=n_c, phi=phi)
    
    return T_end, P_end, R_end, info
