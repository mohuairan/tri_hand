"""
utils - 数学工具和辅助函数

提供角度转换、旋转矩阵、数值求解等通用工具函数。
适用于树莓派 5 的轻量级实现。
"""

import math
from typing import Tuple, List, Optional
import numpy as np


# ========== 角度转换 ==========

def deg2rad(deg: float) -> float:
    """角度转弧度"""
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    """弧度转角度"""
    return rad * 180.0 / math.pi


def wrap_to_pi(angle: float) -> float:
    """将弧度值归一化到 (-π, π]"""
    angle = math.fmod(angle + math.pi, 2 * math.pi) - math.pi
    return angle


def wrap_to_180(angle: float) -> float:
    """将角度归一化到 (-180°, 180°]"""
    angle = math.fmod(angle + 180, 360) - 180
    return angle


# ========== 旋转矩阵 ==========

def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    """
    绕 X 轴旋转矩阵
    
    Args:
        angle_deg: 旋转角度（度）
    
    Returns:
        3x3 旋转矩阵
    """
    angle_rad = deg2rad(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """
    绕 Y 轴旋转矩阵
    
    Args:
        angle_deg: 旋转角度（度）
    
    Returns:
        3x3 旋转矩阵
    """
    angle_rad = deg2rad(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """
    绕 Z 轴旋转矩阵
    
    Args:
        angle_deg: 旋转角度（度）
    
    Returns:
        3x3 旋转矩阵
    """
    angle_rad = deg2rad(angle_deg)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotation_matrix_from_q1q2(q1_deg: float, q2_deg: float) -> np.ndarray:
    """
    根据 q1, q2 计算旋转矩阵（ZYX 顺序）
    
    Args:
        q1_deg: 屈伸角（度）
        q2_deg: 外展角（度）
    
    Returns:
        3x3 旋转矩阵
    """
    q1_rad = deg2rad(q1_deg)
    q2_rad = deg2rad(q2_deg)
    
    c1 = math.cos(q1_rad)
    s1 = math.sin(q1_rad)
    c2 = math.cos(q2_rad)
    s2 = math.sin(q2_rad)
    
    # R = R_q1q2 (与 MATLAB 代码一致)
    R = np.array([
        [c2, s2 * s1, s2 * c1],
        [0, c1, -s1],
        [-s2, c2 * s1, c2 * c1]
    ])
    
    return R


def euler_angles_from_rotation(R: np.ndarray) -> Tuple[float, float, float]:
    """
    从旋转矩阵提取欧拉角（ZYX 顺序）
    
    Args:
        R: 3x3 旋转矩阵
    
    Returns:
        (yaw, pitch, roll) 角度（度）
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    if sy > 1e-10:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(-R[0, 1], R[1, 1])
    
    return (rad2deg(yaw), rad2deg(pitch), rad2deg(roll))


# ========== 向量运算 ==========

def vector_norm(v: np.ndarray) -> float:
    """计算向量模长"""
    return float(np.linalg.norm(v))


def vector_normalize(v: np.ndarray) -> np.ndarray:
    """向量归一化"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError("向量模长过小，无法归一化")
    return v / norm


def vector_angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两向量夹角（度）
    
    Args:
        v1, v2: 输入向量
    
    Returns:
        夹角（度）
    """
    v1_norm = vector_normalize(v1)
    v2_norm = vector_normalize(v2)
    dot = np.dot(v1_norm, v2_norm)
    angle_rad = math.acos(np.clip(dot, -1.0, 1.0))
    return rad2deg(angle_rad)


# ========== 数值求解工具 ==========

class NewtonSolver:
    """
    牛顿法求解器
    
    用于树莓派等嵌入式设备，避免 scipy 依赖
    """
    
    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def solve(self, f, df, x0: float) -> Tuple[Optional[float], bool]:
        """
        牛顿法求解单变量方程
        
        Args:
            f: 目标函数
            df: 导数函数
            x0: 初始猜测
        
        Returns:
            (解，成功标志)
        """
        x = x0
        for i in range(self.max_iterations):
            fx = f(x)
            if abs(fx) < self.tolerance:
                return x, True
            
            dfx = df(x)
            if abs(dfx) < 1e-10:
                return None, False
            
            x = x - fx / dfx
        
        return x, abs(f(x)) < self.tolerance
    
    def solve_brentq(self, f, a: float, b: float) -> Tuple[Optional[float], bool]:
        """
        Brent 方法求解（简化版二分法）
        
        Args:
            f: 目标函数
            a, b: 搜索区间
        
        Returns:
            (解，成功标志)
        """
        fa = f(a)
        fb = f(b)
        
        if fa * fb > 0:
            return None, False
        
        if abs(fa) < self.tolerance:
            return a, True
        if abs(fb) < self.tolerance:
            return b, True
        
        for i in range(self.max_iterations):
            c = (a + b) / 2
            fc = f(c)
            
            if abs(fc) < self.tolerance:
                return c, True
            
            if fa * fc < 0:
                a = c
                fa = fc
            else:
                b = c
                fb = fc
            
            if abs(b - a) < self.tolerance:
                return c, True
        
        return None, False


def numerical_solve_1d(f, x0: float, tolerance: float = 1e-8, 
                       max_iter: int = 100) -> Tuple[Optional[float], bool]:
    """
    数值求解单变量方程（牛顿法 + 二分法混合）
    
    Args:
        f: 目标函数
        x0: 初始猜测
        tolerance: 容差
        max_iter: 最大迭代次数
    
    Returns:
        (解，成功标志)
    """
    h = 1e-6  # 数值微分步长
    
    # 首先尝试牛顿法
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tolerance:
            return x, True
        
        # 数值导数
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        
        if abs(dfx) < 1e-10:
            break  # 牛顿法失败，尝试二分法
        
        # 限制步长，防止发散
        dx = fx / dfx
        if abs(dx) > 10:
            dx = 10 if dx > 0 else -10
        
        x_new = x - dx
        
        # 检查是否收敛
        if abs(f(x_new)) < tolerance:
            return x_new, True
        
        # 如果新值更差，尝试二分法
        if abs(f(x_new)) > abs(fx) * 2:
            break
        
        x = x_new
    
    # 牛顿法失败，尝试在合理范围内搜索
    # 先在 [-90, 90] 范围内扫描寻找符号变化
    for a in range(-90, 90, 1):
        b = a + 1
        fa = f(a)
        fb = f(b)
        
        if abs(fa) < tolerance:
            return float(a), True
        if abs(fb) < tolerance:
            return float(b), True
        
        if fa * fb < 0:
            # 找到符号变化，使用二分法
            result, success = numerical_solve_bounded(f, float(a), float(b), tolerance, max_iter)
            if success:
                return result, True
    
    return None, False


def numerical_solve_bounded(f, a: float, b: float, 
                            tolerance: float = 1e-8,
                            max_iter: int = 100) -> Tuple[Optional[float], bool]:
    """
    数值求解 bounded 区间内的单变量方程（二分法）
    
    Args:
        f: 目标函数
        a, b: 搜索区间
        tolerance: 容差
        max_iter: 最大迭代次数
    
    Returns:
        (解，成功标志)
    """
    fa = f(a)
    fb = f(b)
    
    if abs(fa) < tolerance:
        return a, True
    if abs(fb) < tolerance:
        return b, True
    
    if fa * fb > 0:
        # 尝试 fsolve 风格的方法
        return numerical_solve_1d(f, (a + b) / 2, tolerance, max_iter)
    
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tolerance:
            return c, True
        
        if fa * fc < 0:
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        
        if abs(b - a) < tolerance:
            return c, True
    
    return None, False
