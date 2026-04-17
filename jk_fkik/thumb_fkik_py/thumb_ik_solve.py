#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拇指机构完整逆运动学求解

功能：根据目标位置和法向量，求解所有四个关节的角度
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# 支持两种运行方式：相对导入（作为模块）和绝对导入（直接运行）
try:
    from .params import ThumbParams
    from .thumb_ik_q1q2 import thumb_ik_q1q2, IKQ1Q2Info
    from .thumb_ik_q3q4 import thumb_ik_q3q4, IKQ3Q4Info
    from .thumb_fk_main import thumb_fk_main, FKInfo
except ImportError:
    from params import ThumbParams
    from thumb_ik_q1q2 import thumb_ik_q1q2, IKQ1Q2Info
    from thumb_ik_q3q4 import thumb_ik_q3q4, IKQ3Q4Info
    from thumb_fk_main import thumb_fk_main, FKInfo


def thumb_ik_solve(P_target: np.ndarray, n_c: np.ndarray,
                   params: Optional[ThumbParams] = None) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    """
    完整的拇指机构逆运动学求解
    
    功能：根据目标位置和法向量，求解所有四个关节的角度
    
    参数:
        P_target : 3x1 向量，末端目标位置 [x; y; z] (mm)
        n_c      : 3x1 向量，约束平面法向量（单位向量）
        params   : 参数对象
    
    返回:
        q        : [q1, q2, q3, q4] 关节角度列表 (度)，如果失败则为 None
        info     : 包含所有中间信息的字典
            info['q1q2']  : IKQ1Q2Info 对象
            info['q3q4']  : IKQ3Q4Info 对象
            info['fk']    : FKInfo 对象（正解验证结果）
            info['error'] : 错误信息（如果有）
    """
    if params is None:
        params = ThumbParams()
    
    result_info = {
        'q1q2': None,
        'q3q4': None,
        'fk': None,
        'error': None
    }
    
    # 步骤 1: 求解 q1, q2
    q1, q2, info_q1q2 = thumb_ik_q1q2(P_target, n_c, params)
    result_info['q1q2'] = info_q1q2
    
    if q1 is None or q2 is None:
        result_info['error'] = f'q1/q2 求解失败：{info_q1q2.error_msg}'
        return None, result_info
    
    # 步骤 2: 求解 q3, q4
    q3, q4, info_q3q4 = thumb_ik_q3q4(P_target, q1, q2, params)
    result_info['q3q4'] = info_q3q4
    
    if q3 is None or q4 is None:
        result_info['error'] = f'q3/q4 求解失败：{info_q3q4.error_msg}'
        return None, result_info
    
    # 步骤 3: 正解验证
    T, P, R, fk_info = thumb_fk_main(q1, q2, q3, q4, params)
    result_info['fk'] = fk_info
    
    pos_error = np.linalg.norm(P - P_target)
    if pos_error > params.position_tolerance:
        result_info['error'] = f'正解验证警告：位置误差={pos_error:.6f}mm'
    
    return [q1, q2, q3, q4], result_info
