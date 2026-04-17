#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拇指机构正逆运动学 Python 包
用于树莓派 5 等平台运行

作者：基于 MATLAB 版本移植
功能：计算灵巧手拇指机构的正运动学和逆运动学
"""

from .params import ThumbParams
from .thumb_fk_main import thumb_fk_main, FKInfo
from .thumb_ik_q1q2 import thumb_ik_q1q2, IKQ1Q2Info
from .thumb_ik_q3q4 import thumb_ik_q3q4, IKQ3Q4Info
from .thumb_ik_solve import thumb_ik_solve
from .thumb_fk_solve import thumb_fk_solve

__version__ = '1.0.0'
__author__ = '基于 MATLAB 版本移植'

__all__ = [
    'ThumbParams',
    'FKInfo',
    'IKQ1Q2Info',
    'IKQ3Q4Info',
    'thumb_fk_main',
    'thumb_fk_solve',
    'thumb_ik_q1q2',
    'thumb_ik_q3q4',
    'thumb_ik_solve',
]
