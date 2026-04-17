# 拇指机构逆运动学可视化模块

**版本：2.0**  
**更新日期：2026-04-10**

---

## 📁 文件说明

### 核心代码文件（5 个）

| 文件名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `thumb_vis_joint_traj.m` | 关节轨迹生成 + 正解计算 | 无 | `vis_joint_traj_data.mat` |
| `thumb_vis_ik_solve.m` | 批量逆解求解 + 误差分析 | `vis_joint_traj_data.mat` | `vis_ik_solve_data.mat` |
| `thumb_vis_generate_points.m` | 可视化坐标生成 | `vis_ik_solve_data.mat` | `vis_points_data.mat` |
| `thumb_vis_animation_3d.m` | 3D 动画视频生成 | `vis_points_data.mat` 等 | `.avi` 视频 + `.mat` 参数 |
| `thumb_vis_diagnose_failures.m` | 失败帧诊断分析 | `vis_ik_solve_data.mat` | `vis_failure_data.mat` + 诊断图 |

### 参数文件（1 个）

| 文件名 | 功能 |
|--------|------|
| `vis_params.m` | 统一管理所有可视化配置参数 |

### 数据文件（自动生成）

| 文件名 | 说明 | 可删除 |
|--------|------|--------|
| `vis_joint_traj_data.mat` | 关节轨迹 + 正解 T/phi | ❌ 否 |
| `vis_ik_solve_data.mat` | 逆解结果 + 误差数据 | ❌ 否 |
| `vis_points_data.mat` | 可视化坐标（O/M/N/T） | ❌ 否 |
| `thumb_ik_vis_params.mat` | 完整可视化参数备份 | ✅ 是 |
| `vis_failure_data.mat` | 失败帧诊断数据 | ✅ 是 |

### 结果文件

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `vis_joint_traj_preview.png` | 图片 | 关节轨迹预览图 |
| `vis_ik_error_analysis.png` | 图片 | 逆解误差分析图 |
| `vis_failure_diagnosis.png` | 图片 | 失败帧诊断图 |
| `thumb_ik_animation_3d.avi` | 视频 | 3D 机构运动动画 |

---

## 🚀 快速开始

### 完整流程（推荐）

```matlab
cd visualization

% 步骤 1: 生成关节轨迹（正解计算 T、phi）
thumb_vis_joint_traj

% 步骤 2: 逆解求解 + 误差分析
thumb_vis_ik_solve

% 步骤 3: 生成可视化坐标
thumb_vis_generate_points

% 步骤 4: 生成 3D 动画视频
thumb_vis_animation_3d

% 步骤 5: 播放视频
implay('thumb_ik_animation_3d.avi')
```

### 失败诊断（可选）

如果逆解失败率 > 5%，运行诊断工具：

```matlab
thumb_vis_diagnose_failures
```

---

## 📋 参数配置

### 修改可视化参数

所有可视化参数已统一在 `vis_params.m` 中管理，包括：

- **视频参数**：文件名、格式、帧率
- **图形窗口**：大小、背景色、可见性
- **颜色定义**：连杆颜色、关键点颜色、坐标轴颜色
- **线宽和点大小**：各连杆线宽、标记大小
- **三维视角**：初始视角、旋转设置
- **坐标轴范围**：显示范围、网格设置
- **字体设置**：字号、字体名称
- **信息框设置**：各信息框位置、颜色
- **运动阶段**：五阶段运动规划参数

### 示例：修改窗口大小

编辑 `vis_params.m`：

```matlab
p.fig_width = 1920;    % 改为 1920 像素
p.fig_height = 1080;   % 改为 1080 像素
```

### 示例：修改连杆颜色

编辑 `vis_params.m`：

```matlab
p.color_OM = [1.0, 0.0, 0.0];  % OM 连杆改为红色
```

---

## 📊 运动阶段说明

可视化动画包含五个运动阶段：

| 阶段 | 描述 | 关节变化 | 帧数 |
|------|------|----------|------|
| 1 | 仅 q2 变化 | q2: 0→-20→0 | 60 |
| 2 | 仅 q1 变化 | q1: 0→-15→0 | 60 |
| 3 | 仅 q3+q4 变化 | q3: 0→-60→0, q4: 0→-40→0 | 60 |
| 4 | q1+q2 同步 | q1: 0→-15→0, q2: 0→-20→0 | 60 |
| 5 | 四关节协同 | 四关节按不同频率变化 | 120 |

**总帧数：360 帧**

---

## 🔧 故障排除

### 问题：运行时报错"请先运行 xxx.m 生成数据"

**解决方案**：按照流程顺序依次运行脚本，确保上一步的数据文件已生成。

### 问题：逆解失败率高

**解决方案**：
1. 运行 `thumb_vis_diagnose_failures.m` 分析失败原因
2. 检查失败帧的 T 坐标和 phi 值分布
3. 查看是否超出工作空间或关节限位

### 问题：视频无法播放

**解决方案**：
- 视频格式为 AVI（Motion JPEG 编码）
- 使用系统自带的 Windows Media Player 或 VLC 播放器
- 如仍无法播放，尝试安装相应的解码器

---

## 📝 依赖关系

```
thumb_vis_joint_traj.m → thumb_fk_main.m → params.m
                         vis_params.m

thumb_vis_ik_solve.m → thumb_ik_q1q2.m → params.m
                       thumb_ik_q3q4.m → params.m

thumb_vis_generate_points.m → thumb_fk_main.m → params.m

thumb_vis_animation_3d.m → vis_params.m
                           params.m

thumb_vis_diagnose_failures.m → (仅数据分析，无核心依赖)
```

---

## 📌 注意事项

1. **只读调用**：所有可视化脚本只读调用核心算法文件，不修改正逆解代码
2. **参数一致性**：修改 `params.m` 会影响正逆解计算，修改 `vis_params.m` 仅影响可视化效果
3. **数据文件**：`.mat` 数据文件请勿随意删除，除非重新生成
4. **运行目录**：请在 `visualization` 文件夹内运行脚本，或确保当前目录正确

---

## 📄 许可证

本模块与主项目使用相同许可证。
