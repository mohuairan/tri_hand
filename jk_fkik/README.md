# jk_fkik - 灵巧手运动学求解库

灵巧手拇指和四指正逆运动学 Python 求解库，适用于树莓派 5 等嵌入式平台。

## 项目结构

```
jk_fkik/
├── finger_fkik_py/          # 四指机构运动学包
│   ├── __init__.py          # 包初始化
│   ├── params.py            # 参数配置类 (FingerParams)
│   ├── utils.py             # 数学工具函数
│   ├── q3_to_q4.py          # 四连杆耦合计算
│   ├── forward_kinematics.py # 正运动学求解
│   └── inverse_kinematics.py # 逆运动学求解
│
├── thumb_fkik_py/           # 拇指机构运动学包
│   ├── __init__.py          # 包初始化
│   ├── params.py            # 参数配置类 (ThumbParams)
│   ├── thumb_fk_main.py     # 正运动学核心
│   ├── thumb_fk_solve.py    # 正解封装
│   ├── thumb_ik_q1q2.py     # 逆解 q1/q2
│   ├── thumb_ik_q3q4.py     # 逆解 q3/q4
│   └── thumb_ik_solve.py    # 完整逆解
│
├── examples.py              # 统一示例脚本
├── setup.py                 # 安装脚本
└── requirements.txt         # 依赖列表
```

## 快速开始

### Windows 系统安装指南

#### 步骤 1：确认 Python 已安装

打开命令提示符（cmd）或 PowerShell，输入：

```bash
py --version
```

或

```bash
python --version
```

如果显示版本号（如 `Python 3.x.x`），说明 Python 已安装。

如果显示"不是内部或外部命令"，请：
1. 访问 https://www.python.org/downloads/ 下载 Python
2. 安装时务必勾选 **"Add Python to PATH"** 选项
3. 安装完成后重启终端

#### 步骤 2：使用 py 命令安装依赖

Windows 系统推荐使用 `py` 命令（Python 启动器）：

```bash
py -m pip install -r requirements.txt
```

#### 步骤 3：安装库

```bash
py -m pip install -e .
```

### Linux/macOS 系统

```bash
# 安装依赖
pip3 install -r requirements.txt

# 安装库
pip3 install -e .
```

### 备用安装方式

如果上述命令失败，可尝试：

```bash
# 方式 1：使用 python -m pip
python -m pip install -r requirements.txt
python -m pip install -e .

# 方式 2：使用 pip3
pip3 install -r requirements.txt
pip3 install -e .
```

### 3. 运行示例

**重要：请先切换到项目目录**

```bash
# 切换到项目目录（首次运行时执行一次即可）
cd C:\Users\admin\Desktop\上海钧舵\三指手\jk_fkik
```

然后在项目目录中运行：

```bash
# 方式 1：使用 py 命令（推荐）
py examples.py thumb fk           # 拇指正解
py examples.py thumb ik           # 拇指逆解
py examples.py finger fk          # 四指正解
py examples.py finger ik          # 四指逆解

# 交互式输入参数
py examples.py thumb fk --input
py examples.py finger ik --input

# 命令行直接传参
py examples.py thumb fk --q1 -24.6 --q2 -0.32 --q3 -17.43 --q4 -20.17
py examples.py finger ik --Tx -3.69 --Ty 93.91 --Tz 46.59

# 方式 2：使用 python -m 命令
python examples.py thumb fk
```

## 功能特性

- ✅ **拇指机构**：4 自由度运动学求解
- ✅ **四指机构**：3 自由度运动学求解
- ✅ **正运动学**：已知关节角求末端位姿
- ✅ **逆运动学**：已知目标位姿求关节角
- ✅ **参数化配置**：支持自定义机构参数
- ✅ **轻量级依赖**：仅需 NumPy，适用于嵌入式设备

## API 参考

### 拇指机构 (thumb_fkik_py)

```python
from thumb_fkik_py import thumb_fk_main, thumb_ik_solve, ThumbParams
import numpy as np

# 创建参数对象
params = ThumbParams()

# 正解：已知关节角求末端位置
q1, q2, q3, q4 = -24.6, -0.32, -17.43, -20.17
T, P, R, info = thumb_fk_main(q1, q2, q3, q4, params)
print(f"末端位置：{P}")

# 逆解：已知末端位置和法向量求关节角
P_target = np.array([-32.84, 38.41, 83.46])
n_c = np.array([0.006, 0.909, -0.416])
q, info = thumb_ik_solve(P_target, n_c, params)
if q is not None:
    print(f"关节角：q1={q[0]}, q2={q[1]}, q3={q[2]}, q4={q[3]}")
```

### 四指机构 (finger_fkik_py)

```python
from finger_fkik_py import ForwardKinematics, InverseKinematics, FingerParams
import numpy as np

# 创建参数对象
params = FingerParams()

# 正解
fk = ForwardKinematics(params)
delta_d1, delta_d2, q3 = -0.644, -0.644, -52.64
T_pos, T_rot, info = fk.solve_chain(delta_d1, delta_d2, q3)
print(f"末端位置：{T_pos}")

# 逆解
ik = InverseKinematics(params)
T_target = np.array([-3.69, 93.91, 46.59])
q1_solutions, q2_solutions, info = ik.solve_q1q2(T_target)
if info['success']:
    print(f"q1={q1_solutions[0]}, q2={q2_solutions[0]}")
```

## 参数说明

### 拇指机构参数

| 参数 | 含义 | 单位 |
|------|------|------|
| q1, q2, q3, q4 | 关节角度 | 度 |
| Tx, Ty, Tz | 目标位置坐标 | mm |
| nx, ny, nz | 约束平面法向量 | 单位向量 |

### 四指机构参数

| 参数 | 含义 | 单位 |
|------|------|------|
| q1, q2, q3 | 关节角度 | 度 |
| d1, d2 | 电机位移变化量 | mm |
| Tx, Ty, Tz | 目标位置坐标 | mm |

## 系统要求

- Python 3.7+
- NumPy >= 1.20.0

### 可选依赖

- SciPy >= 1.7.0（用于更精确的数值求解）

## 注意事项

1. **角度单位**：所有角度输入输出均为度（deg），内部计算自动转换为弧度
2. **坐标单位**：所有长度单位均为毫米（mm）
3. **关节限位**：求解结果会自动检查关节限位
4. **拇指/四指独立**：两部分使用不同的参数和算法，相互独立，但安装时作为一个整体

## 许可证

上海钧舵 - 内部使用
