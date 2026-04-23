#!/usr/bin/env python3
"""检查 q3→q4 的实际映射曲线"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finger_fkik_py import FingerParams
from finger_fkik_py.q3_to_q4 import q3_to_q4

params = FingerParams()
print(f"q3 范围: [{params.q3_min}, {params.q3_max}]")

# 细密采样
for q3_d in np.linspace(-90, 15, 22):
    q4_d = q3_to_q4(q3_d, params)
    if q4_d is not None:
        print(f"  q3={q3_d:7.2f}°  →  q4={q4_d:8.2f}°")
    else:
        print(f"  q3={q3_d:7.2f}°  →  无解")

# 实际工作范围内的拟合 (只用有意义的范围)
q3_range = np.linspace(-90, 15, 2000)
q3_valid, q4_valid = [], []
for q3_d in q3_range:
    q4_d = q3_to_q4(q3_d, params)
    if q4_d is not None and abs(q4_d) < 200:  # 过滤异常值
        q3_valid.append(q3_d)
        q4_valid.append(q4_d)

q3_v = np.array(q3_valid)
q4_v = np.array(q4_valid)
print(f"\n过滤后有效点: {len(q3_v)}")
print(f"q3: [{q3_v.min():.2f}, {q3_v.max():.2f}]°")
print(f"q4: [{q4_v.min():.2f}, {q4_v.max():.2f}]°")

# 弧度拟合
q3_r = np.deg2rad(q3_v)
q4_r = np.deg2rad(q4_v)

for order in [3, 4, 5]:
    c = np.polyfit(q3_r, q4_r, order)
    pred = np.polyval(c, q3_r)
    err = np.rad2deg(np.max(np.abs(pred - q4_r)))
    rms = np.rad2deg(np.sqrt(np.mean((pred - q4_r)**2)))
    polycoef = c[::-1]
    print(f"\n--- {order}阶 ---  最大误差: {err:.4f}°  RMS: {rms:.4f}°")
    print(f"  polycoef: {' '.join(f'{x:.8f}' for x in polycoef)}")
