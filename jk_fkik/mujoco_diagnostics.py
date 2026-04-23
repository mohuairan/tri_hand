import mujoco
import numpy as np

class MuJoCoProbe:
    """
    MuJoCo 探针工具类。
    用于在动态仿真（如手掌自由掉落或翻滚）过程中，获取稳定且精确的局部相对坐标。
    """
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # 默认关闭重力，防止由于手掌 free 关节导致疯狂加速掉落
        self.model.opt.gravity[:] = 0

    def step(self, steps=500):
        """执行物理步进，使 PD 控制器达到稳定状态"""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_actuator_ctrl(self, act_dict: dict):
        """
        设置致动器目标角度
        :param act_dict: dict, 格式为 {'thumb_act_q1': np.radians(-40), ...}
        """
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            if name in act_dict:
                self.data.ctrl[i] = act_dict[name]

    def get_relative_position(self, site_name: str, base_body_name: str = 'palm') -> np.ndarray:
        """
        获取 site 相对于指定 base 刚体的相对局部坐标。
        利用矩阵乘法剥离基座的全局翻滚干扰。
        :param site_name: 目标 site 的名称，如 'thumb_tip_site'
        :param base_body_name: 参考刚体名称，默认 'palm'
        :return: (3,) numpy array, 单位：毫米 (mm)
        """
        site_id = self.model.site(site_name).id
        base_id = self.model.body(base_body_name).id
        
        tip_global = self.data.site_xpos[site_id]
        base_global = self.data.xpos[base_id]
        base_mat = self.data.xmat[base_id].reshape(3, 3)
        
        # 将全局差值转换到基座局部坐标系
        tip_local = base_mat.T @ (tip_global - base_global)
        return tip_local * 1000.0


class KinematicsValidator:
    """
    运动学交叉验证器。
    用于对比 Python 解析解/数值解 与 MuJoCo 物理仿真最终稳态位姿的误差。
    """
    def __init__(self, xml_path: str):
        self.probe = MuJoCoProbe(xml_path)
    
    def validate_pose(self, act_ctrls: dict, target_site: str, fk_theoretical_pos: np.ndarray, tolerance_mm=2.0) -> bool:
        """
        验证指定致动器控制指令下，物理仿真的末端位置与 FK 理论位置的误差。
        :param act_ctrls: 致动器控制指令字典
        :param target_site: 要测量的指尖 site
        :param fk_theoretical_pos: Python FK 算出的期望局部坐标 (mm)
        :param tolerance_mm: 容忍误差范围 (mm)
        """
        mujoco.mj_resetData(self.probe.model, self.probe.data)
        self.probe.set_actuator_ctrl(act_ctrls)
        self.probe.step(steps=800)  # 等待 PD 稳定
        
        mjcf_pos = self.probe.get_relative_position(target_site, 'palm')
        error = np.linalg.norm(mjcf_pos - fk_theoretical_pos)
        
        print(f"--- 交叉验证: {target_site} ---")
        print(f"指令集  : {act_ctrls}")
        print(f"FK 理论 : {np.round(fk_theoretical_pos, 2)} mm")
        print(f"MJCF 仿真: {np.round(mjcf_pos, 2)} mm")
        print(f"绝对误差: {error:.2f} mm")
        
        if error > tolerance_mm:
            print(f"[FAILED] 误差 {error:.2f}mm 超过阈值 {tolerance_mm}mm !")
            return False
        print(f"[PASSED] 误差 {error:.2f}mm 在合理范围内。")
        return True


def eval_coupling_error(q3_range_deg: tuple, q4_func_true, q4_func_poly, num_samples=1000):
    """
    四指并联机构 q3-q4 耦合拟合误差评估器。
    用于评估多项式替换真实三角几何解析解所引入的精度损失。
    """
    q3_vals = np.linspace(q3_range_deg[0], q3_range_deg[1], num_samples)
    max_err = 0.0
    worst_q3 = 0.0
    
    for q3 in q3_vals:
        q4_true = q4_func_true(q3)
        q4_poly = q4_func_poly(q3)
        err = abs(q4_true - q4_poly)
        if err > max_err:
            max_err = err
            worst_q3 = q3
            
    print(f"--- q3-q4 多项式耦合误差评估 ---")
    print(f"评估范围: {q3_range_deg} 度")
    print(f"最大关节角误差: {max_err:.4f} 度 (发生于 q3 = {worst_q3:.2f} 度)")
    return max_err

if __name__ == "__main__":
    print("Mujoco 诊断工具库已就绪。")
    print("您可以直接导入 MuJoCoProbe, KinematicsValidator 等类用于后续自动化测试。")
