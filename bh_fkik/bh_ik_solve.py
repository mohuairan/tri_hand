"""
BH_IK_SOLVE 逆运动学独立测试脚本
"""
import math
import numpy as np
from params import params
from bh_fk_solve import bh_fk_main
from bh_ik_main import bh_ik_main


def main():
    # 1. 加载参数
    p = params()
    
    print('=== 逆运动学独立测试（实测校准版）===')
    print('请输入目标位置坐标 (mm)，按回车键使用默认值')
    print('默认值：[-21.3643, 16.3106, 78.1069]\n')
    
    # 用户输入
    px = input(f'请输入 X (默认 -21.3643): ').strip()
    py = input(f'请输入 Y (默认 16.3106): ').strip()
    pz = input(f'请输入 Z (默认 78.1069): ').strip()
    
    # 解析输入，使用默认值
    P_input = [
        float(px) if px else -21.3643,
        float(py) if py else 16.3106,
        float(pz) if pz else 78.1069
    ]
    
    print('=== 逆运动学独立测试（实测校准版）===')
    print(f'输入目标位置 (mm): [{P_input[0]:.4f}, {P_input[1]:.4f}, {P_input[2]:.4f}]')
    print(f'目标点距离原点：{np.linalg.norm(P_input):.4f} mm (工作空间：{p["D_min"]:.2f} ~ {p["D_max"]:.2f} mm)\n')
    
    # 2. 调用逆解函数
    q_all, status, info = bh_ik_main(P_input, p)
    
    # 3. 输出所有解
    print('--- 所有可能的逆解 ---')
    for i in range(info['num_solutions']):
        q = info['all_solutions'][i]
        print(f'解 {i+1}: q = [{q[0]:7.2f}°, {q[1]:7.2f}°, {q[2]:7.2f}°]', end='')
        
        in_limit = (q[0] >= p['limit']['q1'][0] and q[0] <= p['limit']['q1'][1]) and \
                   (q[1] >= p['limit']['q2'][0] and q[1] <= p['limit']['q2'][1]) and \
                   (q[2] >= p['limit']['q3'][0] and q[2] <= p['limit']['q3'][1])
        
        if in_limit:
            print(' [OK] (在限位内)')
        else:
            print(' [X] (超限)')
        
        # 验证位置
        T_test = bh_fk_main(q, p)
        P_test = T_test[0:3, 3]
        err_test = np.linalg.norm(P_test - P_input)
        print(f'       位置：[{P_test[0]:.4f}, {P_test[1]:.4f}, {P_test[2]:.4f}], 误差：{err_test:.6f} mm')
    
    print()
    
    # 4. 输出有效解
    if status == 1:
        print('=== 有效解（在限位内）===')
        if isinstance(q_all[0], (int, float)):
            # 单个解
            num_valid = 1
            q_all = [q_all]
        else:
            num_valid = len(q_all)
        
        for i in range(num_valid):
            q = q_all[i]
            print(f'有效解 {i+1}: q = [{q[0]:.4f}°, {q[1]:.4f}°, {q[2]:.4f}°]')
            
            T_verify = bh_fk_main(q, p)
            P_verify = T_verify[0:3, 3]
            err_pos = np.linalg.norm(P_verify - P_input)
            print(f'           位置复现误差：{err_pos:.6f} mm\n')
    else:
        print('✗ 无有效解')


if __name__ == "__main__":
    main()
