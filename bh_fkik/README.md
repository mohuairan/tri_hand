# 巴赫三自由度灵巧手正逆运动学

本项目包含三自由度灵巧手的正运动学（Forward Kinematics）和逆运动学（Inverse Kinematics）解算的 Python 实现。

## 文件结构

```
.
├── params.py           # 运动学参数定义（连杆长度、关节限位等）
├── bh_fk_solve.py      # 正运动学解算及测试脚本
├── bh_ik_main.py       # 逆运动学核心解算函数
├── bh_ik_solve.py      # 逆运动学测试脚本
├── README.md           # 本说明文件
└── requirements.txt    # Python 依赖
```

## 依赖

- Python 3.6+
- NumPy

### 安装依赖

**Windows 系统：**
```bash
# 方法 1：使用 py 命令
py -m pip install numpy

# 方法 2：如果已安装 pip
pip install -r requirements.txt
```

**Linux/Mac 系统：**
```bash
pip3 install -r requirements.txt
```

## 运动学参数

| 参数 | 值 | 说明 |
|------|-----|------|
| L1 | 50.0 mm | ON 连杆长度 |
| L2 | 42.66 mm | NT 连杆长度 |
| offset_deg | 18.55° | 关节 3 的偏置角度 |

### 关节限位（度）

| 关节 | 范围 |
|------|------|
| q1 | [-180°, 180°] |
| q2 | [-80°, 45°] |
| q3 | [0°, 80°] |

### 工作空间

- 最大工作半径：D_max = L1 + L2 = 92.66 mm
- 最小工作半径：D_min ≈ 23.5 mm（取决于关节 3 最大角度时的折叠状态）

## 使用方法

### 正运动学（已知关节角度，求末端位置）

```python
from params import params
from bh_fk_solve import bh_fk_main

# 加载参数
p = params()

# 输入关节角度 [q1, q2, q3] (度)
q = [-127.36, -43.71, 35.52]

# 计算 4x4 齐次变换矩阵
T = bh_fk_main(q, p)

# 获取末端位置 (x, y, z)
pos = T[0:3, 3]
print(f"位置：X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f} mm")
```

运行测试脚本：
```bash
python bh_fk_solve.py
```

### 逆运动学（已知末端位置，求关节角度）

```python
from params import params
from bh_ik_main import bh_ik_main

# 加载参数
p = params()

# 输入目标位置 [x, y, z] (mm)
P = [-21.3643, 16.3106, 78.1069]

# 求解逆运动学
q_all, status, info = bh_ik_main(P, p)

if status == 1:
    print(f"找到有效解：{q_all}")
else:
    print("无有效解（可能超出工作空间或限位）")

# 查看所有解信息
print(f"所有解数量：{info['num_solutions']}")
print(f"有效解数量：{info['num_valid']}")
```

运行测试脚本：
```bash
python bh_ik_solve.py
```

## API 参考

### params()
返回包含所有运动学参数的字典。

**返回:**
- `L1`: 连杆 1 长度 (mm)
- `L2`: 连杆 2 长度 (mm)
- `offset_deg`: 关节 3 偏置角度 (度)
- `limit`: 关节限位字典
- `D_max`: 最大工作半径
- `D_min`: 最小工作半径
- `tol`: 数值容差

### bh_fk_main(q, p)
正运动学解算。

**参数:**
- `q`: 关节角度列表 [q1, q2, q3] (度)
- `p`: 参数字典

**返回:**
- `T`: 4x4 齐次变换矩阵 (numpy.ndarray)

### bh_ik_main(P, p)
逆运动学解算。

**参数:**
- `P`: 目标位置列表 [x, y, z] (mm)
- `p`: 参数字典

**返回:**
- `q_all`: 有效解（单个解列表或解的列表）
- `status`: 状态码 (1=找到有效解，0=无有效解)
- `info`: 包含详细信息的字典
  - `all_solutions`: 所有可能的解
  - `valid_solutions`: 限位内的有效解
  - `num_solutions`: 总解数
  - `num_valid`: 有效解数
  - `distance`: 目标点距离原点的距离

## 算法说明

### 正运动学
基于 D-H 参数法，通过齐次变换矩阵计算末端执行器位置：
1. 将关节角度转换为弧度
2. 计算各连杆的旋转矩阵
3. 连乘得到总旋转矩阵
4. 根据几何关系计算末端位置

### 逆运动学
采用解析法求解：
1. 计算目标点距离，判断是否在工作空间内
2. 使用余弦定理求解关节 3 角度（2 组解）
3. 求解关节 1 角度（2 组解）
4. 遍历所有组合（共 4 组解）
5. 筛选满足关节限位的解
6. 去除三角函数等价的重复解

## 注意事项

1. 角度单位：输入/输出均为**度 (degree)**，内部计算使用弧度
2. 长度单位：所有长度参数均为**毫米 (mm)**
3. 坐标系定义：
   - Z 轴向上
   - X-Y 平面为水平面
4. 逆解可能有多组，函数会返回所有在限位内的有效解
5. 当目标点超出工作空间时，返回 `status=0`

## 测试数据

### 正运动学测试
- 输入：`q = [-127.36°, -43.71°, 35.52°]`
- 预期输出：`X ≈ -21.36 mm, Y ≈ 16.31 mm, Z ≈ 78.11 mm`

### 逆运动学测试
- 输入：`P = [-21.3643, 16.3106, 78.1069] mm`
- 预期输出：多组解，其中有效解应能精确复现目标位置

## 版本

- 原始版本：MATLAB (2024)
- Python 移植版本：1.0 (2026)

## 作者

基于巴赫灵巧手运动学分析移植
