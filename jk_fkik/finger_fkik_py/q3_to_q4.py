"""
q3_to_q4 - 四连杆机构运动学求解

根据 q3 角度计算 q4 角度，实现四连杆机构的耦合关系。
"""

import math
import numpy as np
from typing import Optional
from .params import FingerParams
from . import utils


def q3_to_q4(q3: float, params: FingerParams) -> Optional[float]:
    """
    四连杆机构运动学求解
    
    根据 q3 角度计算 q4 角度（NT 相对 MN 延长线夹角的变化量）
    
    Args:
        q3: MN 杆转动角度（度），逆时针为正，顺时针为负
        params: 手指机构参数
    
    Returns:
        q4: NT 相对 MN 延长线夹角的变化量（度）
            初始位置 q3=0 时 q4=0；两者均向负值变化（顺时针）
            如果无解，返回 None
    
    调用方式：
        q4 = q3_to_q4(q3, params)
    """
    # ========== 从 params 读取参数 ==========
    L_MN = params.L_MN
    L_NQ = params.L_NQ
    L_KQ = params.L_KQ
    MK_LEN = params.MK_LEN
    MK_ANGLE = params.MK_ANGLE
    PHI = params.PHI
    ANGLE_INIT = params.theta4_init
    
    # ========== 固定点 K 坐标 ==========
    KX = MK_LEN * math.cos(utils.deg2rad(MK_ANGLE))
    KY = MK_LEN * math.sin(utils.deg2rad(MK_ANGLE))
    
    # ========== 动点 N 坐标 ==========
    NX = L_MN * math.cos(utils.deg2rad(q3))
    NY = L_MN * math.sin(utils.deg2rad(q3))
    
    # ========== 虚拟连杆 NK ==========
    DX = KX - NX
    DY = KY - NY
    L_NK = math.sqrt(DX ** 2 + DY ** 2)
    
    # 限位检查：如果 NK 超出可行范围，返回 None
    if (L_NK < abs(L_KQ - L_NQ)) or (L_NK > L_KQ + L_NQ):
        return None  # 标记此位置无解
    
    # ========== 三角形内角 gamma = ∠QNK ==========
    THETA_NK = math.atan2(DY, DX)
    
    COS_GAMMA = (L_NQ ** 2 + L_NK ** 2 - L_KQ ** 2) / (2 * L_NQ * L_NK)
    # 数值钳位：限制在 [-1, 1] 防止数值误差
    COS_GAMMA = max(-1.0, min(1.0, COS_GAMMA))
    GAMMA = math.acos(COS_GAMMA)
    
    # ========== NQ 绝对角度（Q 在 MN 上方）==========
    THETA_NQ = THETA_NK - GAMMA
    
    # ========== NT 绝对角度 ==========
    # Q 在 NQ 上，T 在 NT 上，∠QNT=PHI（度），T 在 NQ 顺时针方向
    # 注意：PHI 是度，需要转换为弧度
    THETA_NT = THETA_NQ - utils.deg2rad(PHI)
    
    # ========== NT 相对 MN 延长线的绝对夹角 ==========
    angle_abs = THETA_NT - utils.deg2rad(q3)
    
    # ========== 规范化到 [-π, π] ==========
    angle_abs = utils.wrap_to_pi(angle_abs)
    
    # ========== 输出变化量（减去初始值）==========
    q4 = utils.rad2deg(angle_abs) - ANGLE_INIT
    
    return q4


def q3_to_q4_with_info(q3: float, params: FingerParams) -> dict:
    """
    四连杆机构运动学求解（带详细信息）
    
    Args:
        q3: MN 杆转动角度（度）
        params: 手指机构参数
    
    Returns:
        包含求解结果的字典
    """
    result = {
        'success': False,
        'q4': None,
        'error': None,
        'theta_nk': None,
        'gamma': None,
        'theta_nq': None,
        'theta_nt': None
    }
    
    q4 = q3_to_q4(q3, params)
    
    if q4 is None:
        result['error'] = '四连杆机构无解：NK 连杆超出可行范围'
        return result
    
    result['success'] = True
    result['q4'] = q4
    
    return result
