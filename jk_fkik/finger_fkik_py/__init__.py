"""
finger_fiik_py - 三指手指正逆运动学求解库

适用于树莓派 5 的轻量级实现，支持与拇指（四指机构）整合。

作者：上海钧舵
"""

from .params import FingerParams
from .forward_kinematics import ForwardKinematics
from .inverse_kinematics import InverseKinematics
from . import utils

__version__ = "1.0.0"
__all__ = [
    "FingerParams",
    "ForwardKinematics", 
    "InverseKinematics",
    "utils"
]
