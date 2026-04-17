#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灵巧手运动学统一示例
支持拇指（thumb）和四指（finger）的正逆运动学演示

用法:
    # 选择机构类型和求解类型
    python examples.py thumb fk           # 拇指正解 - 预设值
    python examples.py thumb ik           # 拇指逆解 - 预设值
    python examples.py finger fk          # 四指正解 - 预设值
    python examples.py finger ik          # 四指逆解 - 预设值
    
    # 交互式输入参数
    python examples.py thumb fk --input
    python examples.py finger ik --input
    
    # 命令行直接传参
    python examples.py thumb fk --q1 -24.6 --q2 -0.32 --q3 -17.43 --q4 -20.17
    python examples.py finger ik --Tx -3.69 --Ty 93.91 --Tz 46.59
"""

import sys
import argparse
import numpy as np


# ========== 拇指示例 ==========

def run_thumb_fk(args):
    """拇指正运动学示例"""
    from thumb_fkik_py import thumb_fk_main, ThumbParams
    
    params = ThumbParams()
    
    # 获取输入参数
    if args.input:
        print("\n=== 拇指正运动学 ===")
        q1 = float(input("请输入关节角度 q1 (度，默认 -24.6): ") or -24.6)
        q2 = float(input("请输入关节角度 q2 (度，默认 -0.32): ") or -0.32)
        q3 = float(input("请输入关节角度 q3 (度，默认 -17.43): ") or -17.43)
        q4 = float(input("请输入关节角度 q4 (度，默认 -20.17): ") or -20.17)
    elif args.q1 is not None:
        q1, q2, q3, q4 = args.q1, args.q2, args.q3, args.q4
    else:
        # 预设值
        q1, q2, q3, q4 = -24.6, -0.32, -17.43, -20.17
    
    print(f"\n输入关节角度：q1={q1}°, q2={q2}°, q3={q3}°, q4={q4}°")
    
    # 调用正解函数
    T, P, R, info = thumb_fk_main(q1, q2, q3, q4, params)
    
    # 输出结果
    print(f"\n=== 计算结果 ===")
    print(f"末端位置 P = [{P[0]:.4f}, {P[1]:.4f}, {P[2]:.4f}] mm")
    print(f"\n姿态矩阵 R:")
    print(f"  [{R[0,0]:.4f}  {R[0,1]:.4f}  {R[0,2]:.4f}]")
    print(f"  [{R[1,0]:.4f}  {R[1,1]:.4f}  {R[1,2]:.4f}]")
    print(f"  [{R[2,0]:.4f}  {R[2,1]:.4f}  {R[2,2]:.4f}]")
    print(f"\n约束平面信息:")
    print(f"  OMNT 平面法向量 n_c = [{info.n_c[0]:.6f}, {info.n_c[1]:.6f}, {info.n_c[2]:.6f}]")
    print(f"  平面与基座 X-Z 平面夹角 phi = {info.phi:.4f}°")
    print(f"\n中间点位置:")
    print(f"  M 点位置 = [{info.P_M[0]:.4f}, {info.P_M[1]:.4f}, {info.P_M[2]:.4f}] mm")
    print(f"  N 点位置 = [{info.P_N[0]:.4f}, {info.P_N[1]:.4f}, {info.P_N[2]:.4f}] mm")
    
    return T, P, R, info


def run_thumb_ik(args):
    """拇指逆运动学示例"""
    from thumb_fkik_py import thumb_ik_solve, thumb_fk_main, ThumbParams
    
    params = ThumbParams()
    
    # 获取输入参数
    if args.input:
        print("\n=== 拇指逆运动学 ===")
        print("请输入目标位置坐标 (mm):")
        Tx = float(input("  Tx (默认 -32.84): ") or -32.84)
        Ty = float(input("  Ty (默认 38.41): ") or 38.41)
        Tz = float(input("  Tz (默认 83.46): ") or 83.46)
        print("请输入约束平面法向量 (单位向量):")
        nx = float(input("  nx (默认 0.006): ") or 0.006)
        ny = float(input("  ny (默认 0.909): ") or 0.909)
        nz = float(input("  nz (默认 -0.416): ") or -0.416)
    elif args.Tx is not None:
        Tx, Ty, Tz = args.Tx, args.Ty, args.Tz
        nx = args.nx if args.nx is not None else 0.006
        ny = args.ny if args.ny is not None else 0.909
        nz = args.nz if args.nz is not None else -0.416
    else:
        # 预设值
        Tx, Ty, Tz = -32.84, 38.41, 83.46
        nx, ny, nz = 0.006, 0.909, -0.416
    
    P_target = np.array([Tx, Ty, Tz])
    n_c = np.array([nx, ny, nz])
    
    # 法向量归一化
    n_c = n_c / np.linalg.norm(n_c)
    
    print(f"\n输入:")
    print(f"  目标位置 P_target = [{Tx:.4f}, {Ty:.4f}, {Tz:.4f}] mm")
    print(f"  法向量 n_c = [{nx:.6f}, {ny:.6f}, {nz:.6f}]")
    
    # 调用逆解函数
    q, solve_info = thumb_ik_solve(P_target, n_c, params)
    
    if q is not None:
        print(f"\n=== 逆解结果 ===")
        print(f"  q1 = {q[0]:.4f}°")
        print(f"  q2 = {q[1]:.4f}°")
        print(f"  q3 = {q[2]:.4f}°")
        print(f"  q4 = {q[3]:.4f}°")
        
        # 验证：重新运行正解
        T, P_fk, R, fk_info = thumb_fk_main(q[0], q[1], q[2], q[3], params)
        n_error = np.linalg.norm(fk_info.n_c - n_c)
        p_error = np.linalg.norm(P_fk - P_target)
        print(f"\n=== 验证结果 ===")
        print(f"  位置误差 = {p_error:.6f} mm")
        print(f"  法向量误差 = {n_error:.6f}")
        if solve_info.get('error'):
            print(f"  警告：{solve_info['error']}")
    else:
        print(f"\n逆解失败：{solve_info.get('error', '未知错误')}")
    
    return q, solve_info


# ========== 四指示例 ==========

def run_finger_fk(args):
    """四指正运动学示例"""
    from finger_fkik_py import ForwardKinematics, FingerParams
    
    params = FingerParams()
    fk = ForwardKinematics(params)
    
    # 获取输入参数
    if args.input:
        print("\n=== 四指正运动学 ===")
        print("请输入输入参数:")
        delta_d1 = float(input("  d1 变化量 (mm，默认 -0.644): ") or -0.644)
        delta_d2 = float(input("  d2 变化量 (mm，默认 -0.644): ") or -0.644)
        q3 = float(input("  q3 关节角 (度，默认 -52.64): ") or -52.64)
    elif args.d1 is not None:
        delta_d1, delta_d2, q3 = args.d1, args.d2, args.q3
    else:
        # 预设值
        delta_d1, delta_d2, q3 = -0.644, -0.644, -52.64
    
    print(f"\n输入参数:")
    print(f"  delta_d1 = {delta_d1:.3f} mm")
    print(f"  delta_d2 = {delta_d2:.3f} mm")
    print(f"  q3 = {q3:.4f}°")
    
    # 求解
    T_pos, T_rot, info = fk.solve_chain(delta_d1, delta_d2, q3)
    
    if info.success:
        print(f"\n=== 求解成功 ===")
        print(f"\n中间变量:")
        print(f"  q1 = {info.q1:.4f}°")
        print(f"  q2 = {info.q2:.4f}°")
        print(f"  q3 = {info.q3:.4f}°")
        print(f"  q4 = {info.q4:.4f}°")
        print(f"\n末端 T 点位姿:")
        print(f"  位置 T = [{T_pos[0]:.3f}, {T_pos[1]:.3f}, {T_pos[2]:.3f}] mm")
        print(f"\n  姿态矩阵 R:")
        print(f"    [{T_rot[0,0]:.4f}  {T_rot[0,1]:.4f}  {T_rot[0,2]:.4f}]")
        print(f"    [{T_rot[1,0]:.4f}  {T_rot[1,1]:.4f}  {T_rot[1,2]:.4f}]")
        print(f"    [{T_rot[2,0]:.4f}  {T_rot[2,1]:.4f}  {T_rot[2,2]:.4f}]")
    else:
        print(f"\n求解失败：{info.error}")
    
    return T_pos, T_rot, info


def run_finger_ik(args):
    """四指逆运动学示例"""
    from finger_fkik_py import InverseKinematics, FingerParams
    import numpy as np
    
    params = FingerParams()
    ik = InverseKinematics(params)
    
    # 获取输入参数
    if args.input:
        print("\n=== 四指逆运动学 ===")
        print("请输入目标位置坐标 (mm):")
        Tx = float(input("  Tx (默认 -3.69): ") or -3.69)
        Ty = float(input("  Ty (默认 93.91): ") or 93.91)
        Tz = float(input("  Tz (默认 46.59): ") or 46.59)
    elif args.Tx is not None:
        Tx, Ty, Tz = args.Tx, args.Ty, args.Tz
    else:
        # 预设值
        Tx, Ty, Tz = -3.69, 93.91, 46.59
    
    T_target = np.array([Tx, Ty, Tz])
    
    print(f"\n输入:")
    print(f"  目标点 T = [{Tx:.2f}, {Ty:.2f}, {Tz:.2f}] mm")
    
    # 求解 q1, q2
    q1_solutions, q2_solutions, info = ik.solve_q1q2(T_target)
    
    if info['success']:
        print(f"\n=== 求解成功 ===")
        print(f"找到 {info['num_solutions']} 组解")
        print(f"\n中间变量:")
        print(f"  q3 = {info['q3']:.4f}°")
        print(f"  q4 = {info['q4']:.4f}°")
        print(f"\n计算结果:")
        
        for i, (q1, q2) in enumerate(zip(q1_solutions, q2_solutions)):
            print(f"\n解 {i+1}:")
            print(f"  q1 = {q1:.4f}°")
            print(f"  q2 = {q2:.4f}°")
            
            # 计算对应的 d1, d2 变化量
            from finger_fkik_py.inverse_kinematics import ik_d1_d2
            d1_init, d2_init, d1_new, d2_new, delta_d1, delta_d2, d_info = ik_d1_d2(q1, q2, params)
            
            if d_info['success']:
                print(f"  Δd1 = {delta_d1:.3f} mm")
                print(f"  Δd2 = {delta_d2:.3f} mm")
    else:
        print(f"\n求解失败：{info['error']}")
    
    return q1_solutions, q2_solutions, info


# ========== 主函数 ==========

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='灵巧手运动学统一示例 - 支持拇指和四指的正逆运动学演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 拇指正解
  python examples.py thumb fk
  python examples.py thumb fk --q1 -24.6 --q2 -0.32 --q3 -17.43 --q4 -20.17
  python examples.py thumb fk --input

  # 拇指逆解
  python examples.py thumb ik
  python examples.py thumb ik --Tx -32.84 --Ty 38.41 --Tz 83.46 --nx 0.006 --ny 0.909 --nz -0.416
  python examples.py thumb ik --input

  # 四指正解
  python examples.py finger fk
  python examples.py finger fk --d1 -0.644 --d2 -0.644 --q3 -52.64
  python examples.py finger fk --input

  # 四指逆解
  python examples.py finger ik
  python examples.py finger ik --Tx -3.69 --Ty 93.91 --Tz 46.59
  python examples.py finger ik --input
        '''
    )
    
    # 位置参数
    parser.add_argument('hand', choices=['thumb', 'finger'],
                        help='选择机构类型：thumb(拇指) 或 finger(四指)')
    parser.add_argument('mode', choices=['fk', 'ik'],
                        help='求解类型：fk(正运动学) 或 ik(逆运动学)')
    
    # 通用选项
    parser.add_argument('--input', '-i', action='store_true',
                        help='交互式输入参数')
    
    # 关节角度参数
    parser.add_argument('--q1', type=float, default=None,
                        help='关节角度 q1 (度)')
    parser.add_argument('--q2', type=float, default=None,
                        help='关节角度 q2 (度)')
    parser.add_argument('--q3', type=float, default=None,
                        help='关节角度 q3 (度)')
    parser.add_argument('--q4', type=float, default=None,
                        help='关节角度 q4 (度)')
    
    # 位置参数
    parser.add_argument('--Tx', type=float, default=None,
                        help='目标位置 X 坐标 (mm)')
    parser.add_argument('--Ty', type=float, default=None,
                        help='目标位置 Y 坐标 (mm)')
    parser.add_argument('--Tz', type=float, default=None,
                        help='目标位置 Z 坐标 (mm)')
    
    # 四指特有参数
    parser.add_argument('--d1', type=float, default=None,
                        help='d1 变化量 (mm)')
    parser.add_argument('--d2', type=float, default=None,
                        help='d2 变化量 (mm)')
    
    # 拇指特有参数（法向量）
    parser.add_argument('--nx', type=float, default=None,
                        help='约束平面法向量 X 分量')
    parser.add_argument('--ny', type=float, default=None,
                        help='约束平面法向量 Y 分量')
    parser.add_argument('--nz', type=float, default=None,
                        help='约束平面法向量 Z 分量')
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("灵巧手运动学示例")
    print("=" * 60)
    
    if args.hand == 'thumb':
        if args.mode == 'fk':
            run_thumb_fk(args)
        else:
            run_thumb_ik(args)
    else:  # finger
        if args.mode == 'fk':
            run_finger_fk(args)
        else:
            run_finger_ik(args)
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
