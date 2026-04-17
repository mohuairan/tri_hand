# fkik_solve - 灵巧手正逆运动学求解项目

本项目包含多款灵巧手的正运动学（Forward Kinematics）和逆运动学（Inverse Kinematics）解算实现，提供 Python 和 MATLAB 两种版本。

## 项目结构

```
fkik_solve/
├── bh_fkik/                     # 巴赫三自由度灵巧手运动学
│   ├── params.py               # 运动学参数定义
│   ├── bh_fk_solve.py          # 正运动学解算
│   ├── bh_ik_main.py           # 逆运动学核心函数
│   ├── bh_ik_solve.py          # 逆运动学测试
│   ├── README.md               # 详细说明
│   └── 巴赫正逆运动学 matlab/   # MATLAB 版本实现
│
└── jk_fkik/                     # 杰克灵巧手运动学库
    ├── finger_fkik_py/         # 四指机构运动学包
    ├── thumb_fkik_py/          # 拇指机构运动学包
    ├── examples.py             # 使用示例
    ├── setup.py                # 安装脚本
    └── 杰克正逆运动学 matlab/   # MATLAB 版本实现
        ├── 拇指正逆运动学/      # 拇指机构
        └── 三指正逆运动学/      # 三指机构
```

## 子项目说明

### 1. bh_fkik - 巴赫三自由度灵巧手

三自由度灵巧手的正逆运动学 Python 实现。

**主要功能：**
- 正运动学：已知关节角度 [q1, q2, q3]，求末端位置 (x, y, z)
- 逆运动学：已知目标位置 (x, y, z)，求解所有可能的关节角度组合

**运动学参数：**
| 参数 | 值 | 说明 |
|------|-----|------|
| L1 | 50.0 mm | ON 连杆长度 |
| L2 | 42.66 mm | NT 连杆长度 |
| offset_deg | 18.55° | 关节 3 的偏置角度 |

**快速使用：**
```python
# 正运动学
from params import params
from bh_fk_solve import bh_fk_main
p = params()
q = [-127.36, -43.71, 35.52]
T = bh_fk_main(q, p)

# 逆运动学
from bh_ik_main import bh_ik_main
P = [-21.3643, 16.3106, 78.1069]
q_all, status, info = bh_ik_main(P, p)
```

详细文档请参阅 [bh_fkik/README.md](bh_fkik/README.md)

---

### 2. jk_fkik - 杰克灵巧手运动学库

灵巧手拇指和四指正逆运动学 Python 求解库，适用于树莓派 5 等嵌入式平台。

**主要功能：**
- ✅ **拇指机构**：4 自由度运动学求解
- ✅ **四指机构**：3 自由度运动学求解
- ✅ **正运动学**：已知关节角求末端位姿
- ✅ **逆运动学**：已知目标位姿求关节角

**安装方法：**
```bash
# Windows 系统
py -m pip install -r requirements.txt
py -m pip install -e .

# Linux/macOS 系统
pip3 install -r requirements.txt
pip3 install -e .
```

**使用示例：**
```python
# 拇指机构
from thumb_fkik_py import thumb_fk_main, thumb_ik_solve, ThumbParams
params = ThumbParams()
T, P, R, info = thumb_fk_main(-24.6, -0.32, -17.43, -20.17, params)

# 四指机构
from finger_fkik_py import ForwardKinematics, InverseKinematics, FingerParams
params = FingerParams()
fk = ForwardKinematics(params)
T_pos, T_rot, info = fk.solve_chain(-0.644, -0.644, -52.64)
```

详细文档请参阅 [jk_fkik/README.md](jk_fkik/README.md)

---

## 通用说明

### 单位规范
- **角度单位**：所有角度输入输出均为**度 (degree)**，内部计算使用弧度
- **长度单位**：所有长度参数均为**毫米 (mm)**

### 坐标系定义
- Z 轴向上
- X-Y 平面为水平面

### 依赖要求
- Python 3.6+
- NumPy

### 运行测试
各子项目均提供测试脚本，可直接运行：

**巴赫项目：**
```bash
# 首次运行需安装依赖
cd bh_fkik
py -m pip install -r requirements.txt  # 安装 numpy

# 运行测试
py bh_fk_solve.py    # 正运动学测试 (支持交互式输入)
py bh_ik_solve.py    # 逆运动学测试
```

**杰克项目：**
```bash
cd jk_fkik
# 基础测试
py examples.py thumb fk  # 拇指正解
py examples.py thumb ik  # 拇指逆解
py examples.py finger fk # 四指正解
py examples.py finger ik # 四指逆解

# 交互式输入参数（推荐）
py examples.py thumb fk --input    # 交互式输入关节角度
py examples.py thumb ik --input    # 交互式输入目标位置
py examples.py finger fk --input   # 交互式输入关节角度
py examples.py finger ik --input   # 交互式输入目标位置

# 命令行直接传参
py examples.py thumb fk --q1 -24.6 --q2 -0.32 --q3 -17.43 --q4 -20.17
py examples.py finger fk --d1 -0.644 --d2 -0.644 --q3 -52.64
py examples.py thumb ik --Tx -32.84 --Ty 38.41 --Tz 83.46
py examples.py finger ik --Tx -3.69 --Ty 93.91 --Tz 46.59
```

## 版本信息

- 原始版本：MATLAB (2024)
- Python 移植版本：1.0 (2026)

## 许可证

上海钧舵 - 内部使用
