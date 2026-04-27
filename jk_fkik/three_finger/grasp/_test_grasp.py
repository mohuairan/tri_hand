"""Headless grasp test with detailed phase diagnostics."""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from grasp_planner import GraspPlanner
from grasp_controller import GraspController


def collect_ids(model):
    return {
        'object': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target_object'),
        'palm': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'palm'),
        'thumb_tip': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thumb_tip'),
        'index_tip': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'index_tip'),
        'middle_tip': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'middle_tip'),
    }


def print_snapshot(label, data, ids, controller, obj_ref):
    obj_pos = data.xpos[ids['object']].copy()
    palm_pos = data.xpos[ids['palm']].copy()
    thumb_pos = data.xpos[ids['thumb_tip']].copy()
    index_pos = data.xpos[ids['index_tip']].copy()
    middle_pos = data.xpos[ids['middle_tip']].copy()
    touch = controller._get_touch_forces()
    object_contacts, geom_hits = controller._get_object_contact_info()

    delta = obj_pos - obj_ref
    lateral_shift = np.linalg.norm(delta[:2])

    print(f"\n=== {label} ===")
    print(f"  Object: {np.round(obj_pos, 5)}")
    print(f"  Palm:   {np.round(palm_pos, 5)}")
    print(f"  Delta:  {np.round(delta, 5)}  lateral={lateral_shift:.5f} m")
    print(f"  Touch:  T={touch[0]:.4f} I={touch[1]:.4f} M={touch[2]:.4f}")
    print(f"  Object contacts: {object_contacts}")
    print(f"  Contact geoms:   {geom_hits}")
    print(
        "  Tip->object center: "
        f"T={np.linalg.norm(thumb_pos - obj_pos):.5f} "
        f"I={np.linalg.norm(index_pos - obj_pos):.5f} "
        f"M={np.linalg.norm(middle_pos - obj_pos):.5f} m"
    )


xml_path = os.path.join(os.path.dirname(__file__), '..', 'mujoco_model', 'scene.xml')
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
ids = collect_ids(model)
obj_start = data.xpos[ids['object']].copy()

print("=== Initial state ===")
print(f"  Palm:   {np.round(data.xpos[ids['palm']].copy(), 5)}")
print(f"  Object: {np.round(obj_start, 5)}")

planner = GraspPlanner(model)
obj_info = planner.estimate_object_from_state(data)
grasp_plan = planner.plan_grasp(obj_info)

print("\n=== Grasp plan ===")
print(f"  Cost:    {grasp_plan['opt_cost']:.6f}")
print(f"  Success: {grasp_plan['optimizer_success']}")
print(f"  Wrist:   {np.round(grasp_plan['wrist_target'], 4)}")
print(f"  Quality: {grasp_plan['quality']}")

controller = GraspController(model, data)

ok = controller.execute_approach(grasp_plan, n_steps=80)
print_snapshot("After Approach", data, ids, controller, obj_start)
print(f"  Approach ok: {ok}")

ok = controller.execute_close(grasp_plan=grasp_plan, force_threshold=0.2,
                              close_speed=0.005, max_steps=500)
print_snapshot("After Close", data, ids, controller, obj_start)
print(f"  Close ok: {ok}")

pre_lift_obj_z = data.xpos[ids['object']][2]
ok = controller.execute_lift(height=0.05, n_steps=60)
print_snapshot("After Lift", data, ids, controller, obj_start)
print(f"  Lift ok: {ok}")
print(f"  Lift gain: {data.xpos[ids['object']][2] - pre_lift_obj_z:.5f} m")

if data.xpos[ids['object']][2] - pre_lift_obj_z > 0.015:
    print("\n>>> GRASP SUCCESS <<<")
else:
    print("\n>>> GRASP FAILED <<<")
