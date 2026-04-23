#!/usr/bin/env python3
"""
杰克灵巧手 MuJoCo 仿真测试脚本

功能:
  1. 加载并验证 MJCF 模型
  2. 打印模型信息 (关节、自由度、驱动器)
  3. 如果有 mujoco viewer, 启动可视化
  4. 执行简单的抓取姿态测试

用法:
  py test_mujoco_model.py           # 验证模型 + 打印信息
  py test_mujoco_model.py --view    # 启动 MuJoCo 可视化
"""

import os
import sys
import argparse
import tempfile
import numpy as np

# ========== MuJoCo 插件加载 Windows 安全策略绕过 ==========
# Windows WDAC/AppLocker 可能阻止从 conda 环境路径加载 DLL。
# 将 MUJOCO_PLUGIN_DIR 指向空目录，跳过插件加载（核心功能不受影响）。
_mujoco_plugin_dir = os.path.join(tempfile.gettempdir(), 'mujoco_empty_plugins')
os.makedirs(_mujoco_plugin_dir, exist_ok=True)
os.environ['MUJOCO_PLUGIN_DIR'] = _mujoco_plugin_dir

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'mujoco_model', 'jack_hand.xml')


def load_and_validate():
    """加载模型并打印基本信息"""
    try:
        import mujoco
    except OSError as e:
        if '4551' in str(e) or '应用程序控制策略' in str(e):
            print("错误: Windows 安全策略阻止了 MuJoCo DLL 加载")
            print("请尝试以下解决方案之一:")
            print("  1. 以管理员身份运行终端")
            print("  2. 将 conda 环境路径添加到 AppLocker 白名单")
            print(f"     路径: {sys.prefix}")
        else:
            print(f"错误: MuJoCo 加载失败: {e}")
        return None, None
    except ImportError:
        print("错误: 未安装 mujoco 包")
        print("请运行: pip install mujoco")
        return None, None
    
    print(f"MuJoCo 版本: {mujoco.__version__}")
    print(f"模型文件: {MODEL_PATH}")
    
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        return None, None
    
    print(f"\n{'='*60}")
    print(f"模型加载成功!")
    print(f"{'='*60}")
    print(f"  自由度 (nq):     {model.nq}")
    print(f"  速度维度 (nv):   {model.nv}")
    print(f"  体数 (nbody):    {model.nbody}")
    print(f"  关节数 (njnt):   {model.njnt}")
    print(f"  几何体数 (ngeom): {model.ngeom}")
    print(f"  驱动器数 (nu):   {model.nu}")
    print(f"  传感器数 (nsensor): {model.nsensor}")
    print(f"  约束数 (neq):    {model.neq}")
    
    # 打印关节信息
    print(f"\n--- 关节列表 ---")
    for i in range(model.njnt):
        name = model.joint(i).name
        jnt_type = ['free', 'ball', 'slide', 'hinge'][model.jnt_type[i]]
        if model.jnt_limited[i]:
            lo = model.jnt_range[i, 0]
            hi = model.jnt_range[i, 1]
            print(f"  [{i:2d}] {name:25s}  type={jnt_type:6s}  range=[{np.degrees(lo):7.2f}°, {np.degrees(hi):7.2f}°]")
        else:
            print(f"  [{i:2d}] {name:25s}  type={jnt_type:6s}  unlimited")
    
    # 打印驱动器信息
    print(f"\n--- 驱动器列表 ---")
    for i in range(model.nu):
        name = model.actuator(i).name
        lo = model.actuator_ctrlrange[i, 0]
        hi = model.actuator_ctrlrange[i, 1]
        print(f"  [{i:2d}] {name:25s}  range=[{np.degrees(lo):7.2f}°, {np.degrees(hi):7.2f}°]")
    
    # 打印等式约束
    print(f"\n--- 等式约束 ---")
    for i in range(model.neq):
        eq_type = ['connect', 'weld', 'joint', 'tendon', 'distance'][model.eq_type[i]]
        print(f"  [{i}] type={eq_type}")
    
    return model, data


def run_fk_test(model, data):
    """正运动学测试: 设置关节角并读取指尖位置"""
    import mujoco
    
    print(f"\n{'='*60}")
    print(f"正运动学测试")
    print(f"{'='*60}")
    
    # 测试 1: 零位
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    tips = ['thumb_tip_site', 'index_tip_site', 'middle_tip_site',
            'ring_tip_site', 'little_tip_site']
    tip_names = ['拇指', '食指', '中指', '环指', '小指']
    
    print("\n[零位状态] 指尖位置 (mm):")
    for name, cname in zip(tip_names, tips):
        site_id = model.site(cname).id
        pos = data.site_xpos[site_id] * 1000  # m → mm
        print(f"  {name}: [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")
    
    # 测试 2: 拇指弯曲
    mujoco.mj_resetData(model, data)
    data.ctrl[0] = np.radians(-24.6)   # thumb_q1
    data.ctrl[1] = np.radians(-0.32)   # thumb_q2
    data.ctrl[2] = np.radians(-17.43)  # thumb_q3
    data.ctrl[3] = np.radians(-20.17)  # thumb_q4
    
    # 运行一段时间让位置控制器收敛
    for _ in range(500):
        mujoco.mj_step(model, data)
    
    mujoco.mj_forward(model, data)
    
    print("\n[拇指测试] q1=-24.6°, q2=-0.32°, q3=-17.43°, q4=-20.17°")
    site_id = model.site('thumb_tip_site').id
    pos = data.site_xpos[site_id] * 1000
    print(f"  拇指指尖: [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}] mm")
    
    # 测试 3: 全手握拳
    mujoco.mj_resetData(model, data)
    # 所有手指 q1 弯曲, q3 弯曲
    for i in range(model.nu):
        name = model.actuator(i).name
        if 'q1' in name and 'thumb' not in name:
            data.ctrl[i] = np.radians(-60)
        elif 'q3' in name and 'thumb' not in name:
            data.ctrl[i] = np.radians(-50)
        elif name == 'thumb_act_q1':
            data.ctrl[i] = np.radians(-40)
        elif name == 'thumb_act_q3':
            data.ctrl[i] = np.radians(-50)
        elif name == 'thumb_act_q4':
            data.ctrl[i] = np.radians(-30)
    
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    
    print("\n[握拳姿态] 指尖位置 (mm):")
    for name, cname in zip(tip_names, tips):
        site_id = model.site(cname).id
        pos = data.site_xpos[site_id] * 1000
        print(f"  {name}: [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")


def launch_viewer(model, data):
    """启动 MuJoCo 可视化"""
    try:
        import mujoco.viewer
        print("\n启动 MuJoCo Viewer...")
        print("操作提示: 鼠标左键旋转, 右键平移, 滚轮缩放")
        print("按 Ctrl+Q 或关闭窗口退出\n")
        mujoco.viewer.launch(model, data)
    except ImportError:
        print("mujoco.viewer 不可用，尝试 mujoco-python-viewer...")
        try:
            import mujoco_viewer
            viewer = mujoco_viewer.MujocoViewer(model, data)
            while viewer.is_alive:
                import mujoco
                mujoco.mj_step(model, data)
                viewer.render()
            viewer.close()
        except ImportError:
            print("无法启动可视化。请安装: py -m pip install mujoco")


def main():
    parser = argparse.ArgumentParser(description='杰克灵巧手 MuJoCo 模型测试')
    parser.add_argument('--view', action='store_true', help='启动可视化')
    args = parser.parse_args()
    
    model, data = load_and_validate()
    if model is None:
        return 1
    
    run_fk_test(model, data)
    
    if args.view:
        launch_viewer(model, data)
    else:
        print(f"\n提示: 运行 'py test_mujoco_model.py --view' 启动可视化")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
