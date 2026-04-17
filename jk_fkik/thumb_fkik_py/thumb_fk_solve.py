#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拇指机构正运动学求解与验证脚本

功能：用于调试和验证正运动学算法
"""

import numpy as np
from typing import Optional
from .thumb_fk_main import thumb_fk_main, FKInfo
from .params import ThumbParams


def thumb_fk_solve(q1: float, q2: float, q3: float, q4: float,
                   params: Optional[ThumbParams] = None) -> dict:
    """
    正解求解与验证
    
    功能：根据关节角度计算末端位置，并返回所有信息
    
    参数:
        q1, q2, q3, q4 : 关节角度 (度)
        params         : 参数对象
    
    返回:
        result : 包含所有结果的字典
            result['T']       : 4x4 齐次变换矩阵
            result['P']       : 末端位置 (3x1 向量)
            result['R']       : 末端姿态矩阵 (3x3)
            result['info']    : FKInfo 对象
            result['params']  : 使用的参数对象
    """
    if params is None:
        params = ThumbParams()
    
    # 调用正解函数
    T, P, R, info = thumb_fk_main(q1, q2, q3, q4, params)
    
    result = {
        'T': T,
        'P': P,
        'R': R,
        'info': info,
        'params': params
    }
    
    return result


if __name__ == '__main__':
    """测试代码"""
    print("=" * 60)
    print("拇指机构正解测试")
    print("=" * 60)
    
    # 设置关节角度
    q1, q2, q3, q4 = -24.6, -0.32, -17.43, -20.17
    
    # 调用正解
    result = thumb_fk_solve(q1, q2, q3, q4)
    
    # 输出结果
    print(f"\n关节角度：q1={q1}°, q2={q2}°, q3={q3}°, q4={q4}°")
    print(f"\n末端位置 (mm): x={result['P'][0]:.4f}, y={result['P'][1]:.4f}, z={result['P'][2]:.4f}")
    print(f"\n末端姿态矩阵:")
    print(result['R'])
    
    print(f"\n==================== 约束平面信息 ====================\n")
    print(f"OMNT 平面法向量：[{result['info'].n_c[0]:.6f}, {result['info'].n_c[1]:.6f}, {result['info'].n_c[2]:.6f}]")
    print(f"平面与基座 X-Z 平面夹角：phi = {result['info'].phi:.4f}°")
    
    print(f"\n==================== 中间点位置 ====================\n")
    print(f"M 点位置：[{result['info'].P_M[0]:.4f}, {result['info'].P_M[1]:.4f}, {result['info'].P_M[2]:.4f}] (mm)")
    print(f"N 点位置：[{result['info'].P_N[0]:.4f}, {result['info'].P_N[1]:.4f}, {result['info'].P_N[2]:.4f}] (mm)")
    
    print("\n==================== 测试完成 ====================\n")
