"""
FingerParams - 手指机构参数配置类

集中管理 MCP 关节的所有机构参数，便于传递和维护。
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import math


@dataclass
class FingerParams:
    """
    手指机构参数配置类
    
    包含所有几何参数、杆长、初始角度、关节限位等。
    使用 dataclass 提供清晰的字段定义和类型提示。
    """
    
    # ========== 几何参数 (单位：mm) ==========
    ax: float = 8.82      # 基座 X 方向半宽
    ay: float = 8.0       # 基座 Y 方向偏移
    bx: float = 9.0       # 动平台 X 方向半宽
    by: float = 5.66      # 动平台 Y 方向偏移
    bz: float = 5.66      # 动平台 Z 方向偏移
    p: float = 27.98      # 并联机构初始高度
    
    # 支链长度
    l1: float = 13.0      # 支链 1 长度 (B1-C1 连杆)
    l2: float = 13.0      # 支链 2 长度 (B2-C2 连杆)
    
    # ========== 三杆机构参数 (单位：mm) ==========
    L_PM: float = 42.39   # PM 杆长 (sqrt(42.2^2 + 4^2))
    L_MN: float = 30.0    # MN 杆长
    L_NT: float = 24.16   # NT 杆长
    
    # ========== 四连杆机构参数 ==========
    L_NQ: float = 4.0     # NQ 杆长
    L_KQ: float = 27.5    # KQ 杆长
    MK_LEN: float = 5.0   # MK 距离
    MK_ANGLE: float = -80.0  # MK 与水平线夹角（度）
    PHI: float = 141.89   # ∠QNT 固定角（度）
    
    # ========== P 点坐标 (单位：mm) ==========
    # P 点是空间固定点，可根据实际安装位置修改
    P: Tuple[float, float, float] = field(default=(0.0, 0.0, 27.98))
    
    # ========== 初始角度 (单位：度) ==========
    theta3_init: float = 5.41   # M 点初始角（MN 相对 PM 延长线）
    theta4_init: float = -8.15  # N 点初始角（NT 相对 MN 延长线）
    
    # ========== 关节搜索范围 (单位：度) ==========
    q1_min: float = -90.0   # q1(屈伸角) 最小值
    q1_max: float = 15.0    # q1(屈伸角) 最大值
    q2_min: float = -45.0   # q2(外展角) 最小值
    q2_max: float = 45.0    # q2(外展角) 最大值
    q3_min: float = -90.0   # q3 最小值
    q3_max: float = 15.0    # q3 最大值
    
    # ========== 求解容差 ==========
    tolerance: float = 1e-8
    
    # ========== 用于整合的标识 ==========
    finger_id: str = "finger"  # 手指标识，用于多指系统
    
    def __post_init__(self):
        """初始化后处理"""
        # 如果 P 是列表，转换为元组
        if isinstance(self.P, list):
            self.P = tuple(self.P)
        
        # 设置 OP 距离验证
        self.L_OP = self.p
    
    @property
    def Px(self) -> float:
        """P 点 X 坐标"""
        return self.P[0]
    
    @property
    def Py(self) -> float:
        """P 点 Y 坐标"""
        return self.P[1]
    
    @property
    def Pz(self) -> float:
        """P 点 Z 坐标"""
        return self.P[2]
    
    def get_joint_limits(self) -> dict:
        """获取关节限位字典"""
        return {
            'q1': (self.q1_min, self.q1_max),
            'q2': (self.q2_min, self.q2_max),
            'q3': (self.q3_min, self.q3_max)
        }
    
    def is_q1_valid(self, q1: float) -> bool:
        """检查 q1 是否在有效范围内"""
        return self.q1_min <= q1 <= self.q1_max
    
    def is_q2_valid(self, q2: float) -> bool:
        """检查 q2 是否在有效范围内"""
        return self.q2_min <= q2 <= self.q2_max
    
    def is_q3_valid(self, q3: float) -> bool:
        """检查 q3 是否在有效范围内"""
        return self.q3_min <= q3 <= self.q3_max
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'ax': self.ax, 'ay': self.ay,
            'bx': self.bx, 'by': self.by, 'bz': self.bz,
            'p': self.p,
            'l1': self.l1, 'l2': self.l2,
            'L_PM': self.L_PM, 'L_MN': self.L_MN, 'L_NT': self.L_NT,
            'L_NQ': self.L_NQ, 'L_KQ': self.L_KQ,
            'MK_LEN': self.MK_LEN, 'MK_ANGLE': self.MK_ANGLE,
            'PHI': self.PHI,
            'P': self.P,
            'theta3_init': self.theta3_init,
            'theta4_init': self.theta4_init,
            'q1_min': self.q1_min, 'q1_max': self.q1_max,
            'q2_min': self.q2_min, 'q2_max': self.q2_max,
            'q3_min': self.q3_min, 'q3_max': self.q3_max,
            'tolerance': self.tolerance,
            'finger_id': self.finger_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FingerParams':
        """从字典创建参数对象"""
        return cls(**data)
