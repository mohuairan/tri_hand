"""
BH_FK_MAIN 三自由度灵巧手正运动学解算
"""
import math
import numpy as np
from params import params


def bh_fk_main(q, p):
    """
    正运动学计算
    
    参数:
        q: 关节角度 [q1, q2, q3] (度)
        p: 参数字典
    
    返回:
        T: 4x4 齐次变换矩阵
    """
    # 1. 角度转弧度
    q1 = math.radians(q[0])
    q2 = math.radians(q[1])
    q3 = math.radians(q[2])
    
    # 2. 计算实际关节 3 角度 (含偏置)
    theta3 = q3 + math.radians(p['offset_deg'])
    
    # 3. 三角函数预计算
    s1 = math.sin(q1)
    c1 = math.cos(q1)
    s2 = math.sin(q2)
    c2 = math.cos(q2)
    s3 = math.sin(theta3)
    c3 = math.cos(theta3)
    
    L1 = p['L1']
    L2 = p['L2']
    
    # 4. 根据机构学推导的公式
    # term_K = L1*sin(q2) + L2*sin(q2+theta3)
    #        = L1*s2 + L2*(s2*c3 + c2*s3)
    #        = s2*(L1 + L2*c3) + L2*s3*c2
    term_K = s2 * (L1 + L2 * c3) + L2 * s3 * c2
    
    x = s1 * term_K
    y = -c1 * term_K      # 负号来自 Rx 旋转矩阵
    z = L1 * c2 + L2 * (c2 * c3 - s2 * s3)  # cos(q2+theta3) = c2*c3 - s2*s3
    
    # 5. 构建齐次变换矩阵
    R_z1 = np.array([
        [c1, -s1, 0],
        [s1, c1, 0],
        [0, 0, 1]
    ])
    R_x2 = np.array([
        [1, 0, 0],
        [0, c2, -s2],
        [0, s2, c2]
    ])
    R_x3 = np.array([
        [1, 0, 0],
        [0, c3, -s3],
        [0, s3, c3]
    ])
    
    R = R_z1 @ R_x2 @ R_x3
    
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]
    
    return T


if __name__ == "__main__":
    p = params()
    
    print('=== 正运动学测试===')
    print('请输入关节角度 (度)，按回车键使用默认值')
    print('默认值：[-127.36, -43.71, 35.52]\n')
    
    # 用户输入
    q1 = input(f'请输入 q1 (默认 -127.36): ').strip()
    q2 = input(f'请输入 q2 (默认 -43.71): ').strip()
    q3 = input(f'请输入 q3 (默认 35.52): ').strip()
    
    # 解析输入，使用默认值
    q_input = [
        float(q1) if q1 else -127.36,
        float(q2) if q2 else -43.71,
        float(q3) if q3 else 35.52
    ]
    
    print('=== 正运动学测试===')
    print(f'输入关节角 (度): [{q_input[0]:.2f}, {q_input[1]:.2f}, {q_input[2]:.2f}]')
    
    T = bh_fk_main(q_input, p)
    pos = T[0:3, 3]
    
    print('计算结果:')
    print(f'   X = {pos[0]:.4f} mm')
    print(f'   Y = {pos[1]:.4f} mm  (实测：31.94 mm)')
    print(f'   Z = {pos[2]:.4f} mm  (实测：59.61 mm)')
    
    # 验证公式正确性
    q2 = math.radians(q_input[1])
    theta3 = math.radians(q_input[2] + p['offset_deg'])
    term_K_manual = math.sin(q2)*(p['L1'] + p['L2']*math.cos(theta3)) + p['L2']*math.sin(theta3)*math.cos(q2)
    y_manual = -math.cos(0) * term_K_manual
    print(f'\n公式验证：term_K = {term_K_manual:.4f}, Y = {y_manual:.4f}')
