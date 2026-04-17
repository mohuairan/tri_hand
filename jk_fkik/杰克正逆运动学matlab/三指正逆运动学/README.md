# 三指手 MCP 关节正逆运动学工具箱

## 项目概述

本项目是 ILDA 手机械手 MCP（掌指关节）的运动学分析与仿真工具箱，使用 MATLAB 编写。该机构采用**并联 - 串联混合驱动**设计，包含一个 2-DOF 并联机构驱动基指节，以及一个四连杆耦合的三杆机构驱动末指节。

## 机构特点

- **并联机构部分**：由两个电机驱动（d1, d2），通过支链控制动平台的屈伸（q1）和外展（q2）运动
- **串联机构部分**：由独立电机驱动 q3，通过四连杆机构耦合驱动 q4，形成三杆机构（PM-MN-NT）
- **关节耦合**：q3 和 q4 通过四连杆机构存在运动学耦合关系

---

## 文件结构

```
三指正逆运动学/
├── 核心参数与函数
│   ├── params.m              # 机构参数配置文件
│   └── q3_to_q4.m            # q3-q4 四连杆耦合函数
│
├── 正运动学 (FK) 文件
│   ├── fk_solve.m            # 正运动学求解主程序
│   ├── fk_T0.m               # 串联部分正解（q1=q2=0 时）
│   ├── fk_chain_to_T.m       # 完整正运动学求解
│   ├── fk_d1d2_to_q1q2.m     # d1,d2 → q1,q2 正解
│   ├── fk_generate_points.m  # 正学生成所有点坐标
│   ├── fk_visualize_3d.m     # 3D 可视化与 MP4 生成
│   ├── fk_visualize_3d_traj.m # 轨迹跟踪可视化
│   ├── fk_visualize_gif.m    # GIF 动画生成
│   └── fk_ik_verify.m        # 正逆解精度验证
│
├── 逆运动学 (IK) 文件
│   ├── ik_solve.m            # 逆运动学求解主程序
│   ├── ik_d1_d2.m            # q1,q2 → d1,d2 反解
│   ├── ik_q1q2_new.m         # T 点 → q1,q2 解析求解
│   ├── ik_q3_from_PT.m       # P,T 坐标 → q3 求解
│   ├── ik_solve_q1q2_byT0.m  # 由 T0 和 T 反推 q1,q2
│   ├── ik_generate_points.m  # 逆学生成所有点坐标
│   ├── ik_solve_batch.m      # 批量逆解
│   ├── ik_trajectory_plan.m  # 轨迹规划（正方形/圆形）
│   └── ik_visualize_gif.m    # 逆学 GIF 生成
│
├── 数据文件 (.mat)
│   ├── ik_trajectory_data.mat     # 轨迹规划数据
│   ├── ik_traj_square_seq.mat     # 正方形轨迹控制量
│   ├── ik_traj_circle_seq.mat     # 圆形轨迹控制量
│   ├── fk_ik_verify_result.mat    # 验证结果数据
│   └── fk_visualize_3d_params.mat # 3D 可视化参数
│
└── 输出文件
    ├── fk_animation_3d.mp4       # 3D 动画视频
    ├── fk_traj_square.mp4        # 正方形轨迹视频
    ├── fk_traj_circle.mp4        # 圆形轨迹视频
    ├── fk_ik_verify_result.png   # 验证结果图
    ├── ik_trajectory_plan.png    # 轨迹规划图
    └── ik_trajectory_circle_ctrl.png # 圆形控制量图
```

---

## 文件依赖关系图

### 正运动学 (FK) 依赖关系

```
fk_solve.m (正运动学主程序)
├── params.m (参数文件)
└── fk_chain_to_T.m (由 delta_d1, delta_d2, q3 求末端 T 位姿)
    ├── q3_to_q4.m (q3-q4 四连杆耦合)
    ├── ik_d1_d2.m (求 d1、d2 初始值)
    └── fk_d1d2_to_q1q2.m (由 d1,d2 求 q1,q2)
```

### 正运动学可视化依赖

```
fk_generate_points.m (生成所有点坐标)
├── params.m (参数文件)
├── ik_d1_d2.m (求 d1,d2 初始值)
└── fk_chain_to_T.m (核心 FK 函数)
    ├── q3_to_q4.m
    ├── ik_d1_d2.m
    └── fk_d1d2_to_q1q2.m

fk_visualize_3d.m (3D 动画生成)
└── fk_generate_points.m

fk_visualize_3d_traj.m (轨迹跟踪可视化)
├── params.m
└── fk_generate_points.m

fk_ik_verify.m (正逆解精度验证)
├── params.m
├── fk_chain_to_T.m
├── ik_q1q2_new.m (核心 IK 函数)
└── ik_d1_d2.m
```

### 逆运动学 (IK) 依赖关系

```
ik_solve.m (逆运动学主程序)
├── params.m (参数文件)
└── ik_q1q2_new.m (由 T 点求 q1,q2,q3,q4)
    ├── ik_q3_from_PT.m (由 P,T 坐标求 q3)
    │   ├── params.m
    │   └── q3_to_q4.m
    ├── q3_to_q4.m (q3-q4 耦合)
    ├── fk_T0.m (q1=q2=0 时求 T0)
    │   ├── params.m
    │   └── q3_to_q4.m
    └── ik_solve_q1q2_byT0.m (由 T0 和 T 反推 q1,q2)

ik_d1_d2.m (由 q1,q2 求 d1,d2 变化量)
└── params.m
```

### 逆运动学可视化依赖

```
ik_generate_points.m (逆学生成所有点坐标)
├── params.m
├── ik_q1q2_new.m (核心 IK 函数)
└── ik_d1_d2.m

ik_solve_batch.m (批量逆解)
├── params.m
├── ik_q1q2_new.m
└── ik_d1_d2.m

ik_trajectory_plan.m (轨迹规划)
├── params.m
├── ik_q1q2_new.m
└── ik_d1_d2.m

ik_visualize_gif.m (逆学 GIF 生成)
├── params.m
└── ik_generate_points.m
```

### 完整依赖树 (从顶层到底层)

```
┌──────────────────────────────────────────────────────────────┐
│                       params.m                                │
│                   (核心参数文件 - 被所有文件依赖)              │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────┐                           ┌───────────────┐
│  q3_to_q4.m   │                           │  ik_d1_d2.m   │
│ (q3-q4 耦合)   │                           │(q1,q2→d1,d2)  │
└───────────────┘                           └───────────────┘
        │                                           │
        │                    ┌──────────────────────┘
        │                    │
        ▼                    ▼
┌─────────────────────────────────┐
│         fk_chain_to_T.m         │
│      (正运动学核心函数)          │
│  输入：delta_d1, delta_d2, q3   │
│  输出：T_pos, T_rot, info       │
└─────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│fk_generate_ │ │fk_visualize_│ │  fk_ik_     │
│  points.m   │ │  3d_traj.m  │ │  verify.m   │
└─────────────┘ └─────────────┘ └─────────────┘
        │
        ▼
┌───────────────────┐
│ fk_visualize_3d.m │
└───────────────────┘


┌─────────────────────────────────────────┐
│            fk_T0.m                      │
│    (q1=q2=0 时串联部分正解)              │
│         输入：q3                        │
│         输出：T0, info                  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│       ik_solve_q1q2_byT0.m              │
│    (由 T0 和当前 T 反推 q1,q2)            │
│         输入：P, T0, T_current          │
│         输出：q1, q2, error             │
└─────────────────────────────────────────┘
                    │
                    ▲
        ┌───────────┴───────────┐
        │                       │
        ▼                       │
┌───────────────────┐          │
│  ik_q3_from_PT.m  │──────────┘
│   (由 P,T 求 q3)    │
│    输入：P, T      │
│    输出：q3, info  │
└───────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         ik_q1q2_new.m                   │
│      (逆运动学核心函数)                  │
│  输入：T (目标点)                        │
│  输出：q1, q2, info(含 q3,q4)           │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ik_generate_ │ │ik_trajectory│ │ik_solve_    │
│  points.m   │ │  _plan.m    │ │  batch.m    │
└─────────────┘ └─────────────┘ └─────────────┘
        │
        ▼
┌───────────────────┐
│ik_visualize_gif.m │
└───────────────────┘
```

---

## 核心文件详解

### 1. params.m - 参数配置文件

**作用**：定义机构的所有几何参数和运动学参数

**主要参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ax, ay` | 基座 X/Y 方向尺寸 | 8.82, 8 mm |
| `bx, by, bz` | 动平台尺寸 | 9, 5.66, 5.66 mm |
| `l1, l2` | 支链连杆长度 | 13 mm |
| `L_PM, L_MN, L_NT` | 三杆机构杆长 | 42.39, 30, 24.16 mm |
| `L_NQ, L_KQ` | 四连杆参数 | 4.0, 27.5 mm |
| `q1_min, q1_max` | q1 关节限位 | -90° ~ 15° |
| `q2_min, q2_max` | q2 关节限位 | -45° ~ 45° |

**依赖**：无（基础参数文件）

---

### 2. q3_to_q4.m - 四连杆耦合函数

**作用**：根据 q3 角度计算 q4 角度，实现四连杆机构的运动学耦合

**输入**：`q3` - MN 杆转动角度（度）

**输出**：`q4` - NT 相对 MN 延长线夹角的变化量（度）

**依赖**：`params.m`

---

### 正运动学 (Forward Kinematics) 文件组

#### 3. fk_chain_to_T.m - 完整正运动学求解

**作用**：核心 FK 函数，根据输入 (delta_d1, delta_d2, q3) 计算末端 T 的位姿

**工作流程**：
1. 调用 `ik_d1_d2` 获取初始 d1, d2 值
2. 调用 `fk_d1d2_to_q1q2` 求解 q1, q2
3. 调用 `q3_to_q4` 求解 q4
4. 通过旋转矩阵链计算 T 点位置和姿态

**输入**：
- `delta_d1` - d1 变化量 (mm)
- `delta_d2` - d2 变化量 (mm)
- `q3` - 关节角 q3 (度)
- `params` - 参数结构体

**输出**：
- `T_pos` - T 点位置 [Tx, Ty, Tz]
- `T_rot` - T 点姿态矩阵 (3x3)
- `info` - 包含 q1, q2, q3, q4 等信息

**依赖**：`params.m`, `q3_to_q4.m`, `fk_d1d2_to_q1q2.m`, `ik_d1_d2.m`

---

#### 4. fk_d1d2_to_q1q2.m - 并联机构正解

**作用**：根据 d1, d2 的变化量求解 q1（屈伸角）和 q2（外展角）

**算法**：
1. q2 初值估计：通过 `sin_q2_approx = (d2-d1)/(2*bx)` 
2. q1 求解：使用 `fzero` 求解非线性方程
3. 双变量精化：使用 `fsolve` 同时优化 q1, q2

**依赖**：`params.m`

---

#### 5. fk_T0.m - 串联部分正解

**作用**：在 q1=0, q2=0 的假设下，仅根据 q3 计算末端 T 的位姿

**用途**：被 `ik_q1q2_new.m` 调用，用于分离求解串联部分

**依赖**：`params.m`, `q3_to_q4.m`

---

#### 6. fk_generate_points.m - 正学生成所有点坐标

**作用**：批量生成运动过程中所有关键点（O, P, A1, A2, B1, B2, C1, C2, M, N, T）的坐标

**输入**：
- `delta_d1_seq` - d1 变化量序列
- `delta_d2_seq` - d2 变化量序列
- `q3_seq` - q3 角度序列

**输出**：
- `all_points` - 所有点坐标结构体数组
- `all_info` - 求解信息（关节角、电机位移等）

**依赖**：`params.m`, `ik_d1_d2.m`, `fk_chain_to_T.m`

---

#### 7. fk_visualize_3d.m - 3D 可视化与视频生成

**作用**：生成机构运动的 3D 动画视频（MP4 格式）

**运动阶段**：
1. 仅 d2 变化
2. 仅 d1 变化
3. 仅 q3 变化
4. d1+d2 同步变化
5. 三者协同变化

**输出**：
- `fk_animation_3d.mp4` - 3D 动画视频
- `fk_visualize_3d_params.mat` - 可视化参数

**依赖**：`params.m`, `fk_generate_points.m`

---

#### 8. fk_visualize_3d_traj.m - 轨迹跟踪可视化

**作用**：加载 IK 规划的轨迹控制量，用 FK 生成机构沿指定轨迹运动的视频

**输入**：
- `ik_traj_square_seq.mat` - 正方形轨迹控制量
- `ik_traj_circle_seq.mat` - 圆形轨迹控制量
- `ik_trajectory_data.mat` - 目标轨迹数据

**输出**：
- `fk_traj_square.mp4` - 正方形轨迹视频
- `fk_traj_circle.mp4` - 圆形轨迹视频

**依赖**：`params.m`, `fk_generate_points.m`

---

#### 9. fk_ik_verify.m - 正逆解精度验证

**作用**：验证 FK 和 IK 的一致性

**原理**：
1. 输入控制量 (d1, d2, q3) → FK 求解末端 T
2. 从 T 通过 IK 反求 (d1, d2, q3)
3. 比较输入与恢复值的误差

**输出**：
- `fk_ik_verify_result.png` - 误差分析图
- `fk_ik_verify_result.mat` - 验证结果数据

**依赖**：`params.m`, `fk_chain_to_T.m`, `ik_q1q2_new.m`, `ik_d1_d2.m`

---

### 逆运动学 (Inverse Kinematics) 文件组

#### 10. ik_q1q2_new.m - T 点 → q1,q2 解析求解

**作用**：核心 IK 函数，已知末端 T 点坐标，求解 q1, q2, q3, q4

**工作流程**：
1. 调用 `ik_q3_from_PT` 求解 q3
2. 调用 `q3_to_q4` 求解 q4
3. 调用 `fk_T0` 计算 q1=q2=0 时的 T0
4. 调用 `ik_solve_q1q2_byT0` 从 T0 和 T 反推 q1, q2

**输入**：`T` - 目标点坐标 [Tx, Ty, Tz]

**输出**：
- `q1_solutions`, `q2_solutions` - q1, q2 的解
- `info` - 包含 q3, q4, 求解状态等

**依赖**：`params.m`, `q3_to_q4.m`, `ik_q3_from_PT.m`, `fk_T0.m`, `ik_solve_q1q2_byT0.m`

---

#### 11. ik_q3_from_PT.m - P,T 坐标 → q3 求解

**作用**：根据 P 点和 T 点的空间坐标求解 q3

**算法**：使用 `fsolve` 数值求解，使计算长度与目标距离匹配

**依赖**：`params.m`, `q3_to_q4.m`

---

#### 12. ik_solve_q1q2_byT0.m - 由 T0 和 T 反推 q1,q2

**作用**：已知初始 T0 和旋转后 T，解析求解旋转角 q1, q2

**算法**：
1. 构造相对 P 点的向量
2. 求解 q1 的两个候选值
3. 对每个 q1 候选值解析求 q2
4. 筛选满足角度范围的解

**依赖**：无（纯数学计算）

---

#### 13. ik_d1_d2.m - q1,q2 → d1,d2 反解

**作用**：根据 q1, q2 计算并联机构 d1, d2 的变化量

**工作流程**：
1. 计算初始状态 d1_init, d2_init
2. 计算旋转后的 B 点坐标
3. 求解新的 d1_new, d2_new
4. 计算变化量 delta_d1, delta_d2

**依赖**：`params.m`

---

#### 14. ik_generate_points.m - 逆学生成所有点坐标

**作用**：根据末端 T 点序列，批量求解所有中间点坐标

**输入**：`T_sequence` - N×3 目标位置矩阵

**输出**：
- `all_points` - 所有点坐标
- `all_info` - 求解信息

**依赖**：`params.m`, `ik_q1q2_new.m`, `ik_d1_d2.m`

---

#### 15. ik_trajectory_plan.m - 轨迹规划

**作用**：规划末端 T 点沿正方形和圆形轨迹运动，求解对应的控制量

**功能**：
- 生成 3D 空间中的正方形和圆形轨迹
- 对每个轨迹点求解 IK 得到 (d1, d2, q3)
- 保存轨迹数据供可视化使用

**输出**：
- `ik_trajectory_data.mat` - 完整轨迹数据
- `ik_traj_square_seq.mat` - 正方形控制量
- `ik_traj_circle_seq.mat` - 圆形控制量
- `ik_trajectory_plan.png` - 轨迹规划图

**依赖**：`params.m`, `ik_q1q2_new.m`, `ik_d1_d2.m`

---

#### 16. ik_solve_batch.m - 批量逆解

**作用**：从正解生成的 T 点序列批量求解 IK

**输入**：`T_pose_reference.mat`（由 fk_generate_points 生成）

**依赖**：`params.m`, `ik_q1q2_new.m`, `ik_d1_d2.m`

---

#### 17. ik_visualize_gif.m - 逆学 GIF 生成

**作用**：根据 T 点序列生成逆运动学 GIF 动画

**输入**：`T_pose_reference.mat`

**输出**：`ik_animation.gif`

**依赖**：`params.m`, `ik_generate_points.m`

---

## 使用示例

### 正运动学求解

```matlab
% 运行 fk_solve.m
clear; clc;

p = params();
delta_d1 = -0.644;   % d1 变化量
delta_d2 = -0.644;   % d2 变化量
q3 = -52.64;         % q3 角度

[T_pos, T_rot, info] = fk_chain_to_T(delta_d1, delta_d2, q3, p);

fprintf('末端位置：T = [%.3f, %.3f, %.3f] mm\n', T_pos);
fprintf('关节角：q1=%.2f°, q2=%.2f°, q3=%.2f°, q4=%.2f°\n', ...
    info.q1, info.q2, info.q3, info.q4);
```

### 逆运动学求解

```matlab
% 运行 ik_solve.m
clear; clc;

p = params();
T = [-3.69, 93.91, 46.59];   % 目标点坐标

[q1_sol, q2_sol, info] = ik_q1q2_new(T, p);

fprintf('关节角：q1=%.2f°, q2=%.2f°, q3=%.2f°, q4=%.2f°\n', ...
    q1_sol, q2_sol, info.q3, info.q4);
```

### 轨迹规划与可视化

```matlab
% 1. 运行轨迹规划
ik_trajectory_plan;   % 生成正方形和圆形轨迹控制量

% 2. 运行 FK 轨迹可视化
fk_visualize_3d_traj; % 生成轨迹跟踪视频

% 3. 运行精度验证
fk_ik_verify;         % 验证 FK/IK 一致性
```

---

## 关键坐标点说明

| 点 | 说明 |
|----|------|
| O | 坐标系原点 |
| P | MCP 中心，并联机构与串联机构的连接点 |
| A1, A2 | 电机 1、2 的基座安装点 |
| B1, B2 | 动平台上的铰链点 |
| C1, C2 | 电机驱动的滑块端点 |
| M | PM 杆末端 |
| N | MN 杆末端 |
| T | 末端执行点 |

---

## 关节定义

| 关节 | 说明 | 范围 |
|------|------|------|
| q1 | 屈伸角（绕 X 轴） | -90° ~ 15° |
| q2 | 外展角（绕 Y 轴） | -45° ~ 45° |
| q3 | 三杆机构驱动角 | -90° ~ 15° |
| q4 | 四连杆从动角 | 由 q3 耦合决定 |

---

## 运行环境

- MATLAB R2016b 或更高版本
- 需要 Optimization Toolbox（用于 fsolve, fzero）

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `fk_animation_3d.mp4` | 正运动学 3D 动画 |
| `fk_traj_square.mp4` | 正方形轨迹跟踪视频 |
| `fk_traj_circle.mp4` | 圆形轨迹跟踪视频 |
| `fk_ik_verify_result.png` | 正逆解误差分析图 |
| `ik_trajectory_plan.png` | 轨迹规划与控制量曲线 |
| `ik_trajectory_circle_ctrl.png` | 圆形轨迹控制量详情 |

---

## 作者与版本

- 项目：ILDA 手机械手 MCP 关节运动学分析
- 版本：1.0
- 语言：MATLAB
