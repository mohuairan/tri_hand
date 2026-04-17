"""
BH_IK_MAIN 三自由度灵巧手逆运动学解算
"""
import math
import numpy as np
from params import params


def remove_duplicate_solutions(valid_solutions, p):
    """
    去重函数：移除三角函数等价的重复解
    """
    if len(valid_solutions) <= 1:
        return valid_solutions
    
    unique_solutions = []
    for q in valid_solutions:
        is_duplicate = False
        
        for q_ref in unique_solutions:
            # 检查 q1 是否相差 360° 的整数倍
            q1_diff = abs(q[0] - q_ref[0])
            q1_same = (q1_diff < 0.01) or (abs(q1_diff - 360) < 0.01)
            
            # 检查 q2, q3 是否相同
            q2_same = abs(q[1] - q_ref[1]) < 0.01
            q3_same = abs(q[2] - q_ref[2]) < 0.01
            
            if q1_same and q2_same and q3_same:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_solutions.append(q)
    
    return unique_solutions


def bh_ik_main(P, p):
    """
    逆运动学计算
    
    参数:
        P: 目标位置 [x, y, z] (mm)
        p: 参数字典
    
    返回:
        q_all: 有效解 (可能是一组或多组)
        status: 状态 (1=找到有效解，0=无有效解)
        info: 包含所有解的信息字典
    """
    x = P[0]
    y = P[1]
    z = P[2]
    
    q_all = None
    status = 0
    
    # 初始化结构体
    info = {
        'all_solutions': [],
        'valid_solutions': [],
        'distance': 0,
        'num_solutions': 0,
        'num_valid': 0
    }
    
    # 容差参数
    limit_tol = 0.5
    pos_tol = 1e-4
    
    # 1. 计算距离相关变量
    r = math.sqrt(x**2 + y**2)
    D_sq = x**2 + y**2 + z**2
    D = math.sqrt(D_sq)
    info['distance'] = D
    
    # 工作空间判断
    if D > p['D_max'] + pos_tol or D < p['D_min'] - pos_tol:
        print(f'警告：目标点超出工作空间范围！D = {D:.2f}')
        return q_all, status, info
    
    # 2. 求解 q3（余弦定理）
    cos_theta3 = (D_sq - p['L1']**2 - p['L2']**2) / (2 * p['L1'] * p['L2'])
    
    if cos_theta3 > 1:
        cos_theta3 = 1
    if cos_theta3 < -1:
        cos_theta3 = -1
    
    theta3_rad_1 = math.acos(cos_theta3)
    theta3_rad_2 = -math.acos(cos_theta3)
    
    theta3_list = [theta3_rad_1, theta3_rad_2]
    
    # 3. 求解 q1
    q1_rad_base = math.atan2(x, -y)
    q1_list = [q1_rad_base, q1_rad_base + math.pi]
    
    # 4. 遍历所有组合：2 个 theta3 × 2 个 term_K 符号 = 4 组解
    solutions = []
    
    for i in range(2):
        theta3_rad = theta3_list[i]
        q3_deg = math.degrees(theta3_rad) - p['offset_deg']
        
        A = p['L1'] + p['L2'] * math.cos(theta3_rad)
        B = p['L2'] * math.sin(theta3_rad)
        denom = A**2 + B**2
        
        for sign_K in [1, -1]:
            r_signed = sign_K * r
            
            sin_q2 = (A * r_signed - B * z) / denom
            cos_q2 = (B * r_signed + A * z) / denom
            
            sin_q2 = max(min(sin_q2, 1), -1)
            cos_q2 = max(min(cos_q2, 1), -1)
            
            q2_rad = math.atan2(sin_q2, cos_q2)
            q2_deg = math.degrees(q2_rad)
            
            # 计算 q1
            if sign_K > 0:
                q1_rad = math.atan2(x, -y)
            else:
                q1_rad = math.atan2(-x, y)
            
            q1_deg = math.degrees(q1_rad)
            
            # q1 归一化到 (-180, 180]，允许 -180
            while q1_deg > 180:
                q1_deg = q1_deg - 360
            while q1_deg < -180:
                q1_deg = q1_deg + 360
            
            # 处理边界等价：-180 和 180 物理相同，统一输出 180
            if q1_deg < -179.999:
                q1_deg = 180
            
            solutions.append([q1_deg, q2_deg, q3_deg])
    
    info['all_solutions'] = solutions
    info['num_solutions'] = len(solutions)
    
    # 5. 筛选在限位内的解
    valid_solutions = []
    for q in solutions:
        # 限位检查使用容差，支持边界值
        in_limit = (q[0] >= p['limit']['q1'][0] - limit_tol and q[0] <= p['limit']['q1'][1] + limit_tol) and \
                   (q[1] >= p['limit']['q2'][0] - limit_tol and q[1] <= p['limit']['q2'][1] + limit_tol) and \
                   (q[2] >= p['limit']['q3'][0] - limit_tol and q[2] <= p['limit']['q3'][1] + limit_tol)
        
        if in_limit:
            # 截断到限位边界，但不改变有效值
            q_clamped = [
                max(min(q[0], p['limit']['q1'][1]), p['limit']['q1'][0]),
                max(min(q[1], p['limit']['q2'][1]), p['limit']['q2'][0]),
                max(min(q[2], p['limit']['q3'][1]), p['limit']['q3'][0])
            ]
            valid_solutions.append(q_clamped)
    
    info['valid_solutions'] = valid_solutions
    info['num_valid'] = len(valid_solutions)
    
    # 6. 返回结果
    if len(valid_solutions) > 0:
        # 去重：移除三角函数等价的重复解
        valid_solutions = remove_duplicate_solutions(valid_solutions, p)
        
        if len(valid_solutions) == 1:
            q_all = valid_solutions[0]
        else:
            q_all = valid_solutions
        
        status = 1
        print(f'找到 {len(valid_solutions)} 组有效解')
    else:
        print('警告：所有解均超出限位范围！')
        if len(solutions) > 0:
            q_all = solutions[0]
        status = 0
    
    return q_all, status, info


if __name__ == "__main__":
    # 测试代码
    p = params()
    P_test = [0, 50, 50]
    
    print('=== 逆运动学测试===')
    print(f'目标位置：{P_test}')
    
    q_all, status, info = bh_ik_main(P_test, p)
    
    if status == 1:
        print(f'有效解：{q_all}')
    else:
        print('无有效解')
    
    print(f'所有解数量：{info["num_solutions"]}')
    print(f'有效解数量：{info["num_valid"]}')
