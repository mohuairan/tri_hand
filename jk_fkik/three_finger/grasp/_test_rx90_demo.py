"""Visual test for the grasp planner/controller with phase diagnostics."""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

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
    delta = obj_pos - obj_ref
    touch = controller._get_touch_forces()
    object_contacts, geom_hits = controller._get_object_contact_info()
    lateral_shift = np.linalg.norm(delta[:2])
    print(f"\n[{label}]")
    print(f"  Object: {np.round(obj_pos, 5)}")
    print(f"  Delta:  {np.round(delta, 5)}  lateral={lateral_shift:.5f} m")
    print(f"  Touch:  T={touch[0]:.4f} I={touch[1]:.4f} M={touch[2]:.4f}")
    print(f"  Object contacts: {object_contacts}")
    print(f"  Contact geoms: {geom_hits}")


xml_path = os.path.join(os.path.dirname(__file__), '..', 'mujoco_model', 'scene.xml')
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
ids = collect_ids(model)
obj_start = data.xpos[ids['object']].copy()

palm_mat = data.xmat[ids['palm']].reshape(3, 3)
print(f"Palm pos:    {np.round(data.xpos[ids['palm']].copy(), 5)}")
print(f"Palm Y-axis: {np.round(palm_mat[:, 1], 5)}")
print(f"Palm Z-axis: {np.round(palm_mat[:, 2], 5)}")
print(f"Object pos:  {np.round(obj_start, 5)}")

planner = GraspPlanner(model)
obj_info = planner.estimate_object_from_state(data)
print(f"\nObject center: {np.round(obj_info['center'], 5)}")
print(f"Object radius: {np.max(obj_info['size']):.5f}")

print("\nOptimizing...")
grasp_plan = planner.plan_grasp(obj_info)
print(f"Cost: {grasp_plan['opt_cost']:.6f}")
print(f"Success: {grasp_plan['optimizer_success']}")
print(f"Wrist target: {np.round(grasp_plan['wrist_target'], 4)}")
print(f"Quality: {grasp_plan['quality']}")
for fname in ['thumb', 'index', 'middle']:
    print(f"{fname} close: {np.round(grasp_plan['finger_close'][fname], 2)}")

x = np.array([
    grasp_plan['wrist_target'][0], grasp_plan['wrist_target'][1], grasp_plan['wrist_target'][2],
    grasp_plan['wrist_target'][3], grasp_plan['wrist_target'][4], grasp_plan['wrist_target'][5],
    grasp_plan['finger_close']['thumb'][0], grasp_plan['finger_close']['thumb'][1],
    grasp_plan['finger_close']['thumb'][2], grasp_plan['finger_close']['thumb'][3],
    grasp_plan['finger_close']['index'][0], grasp_plan['finger_close']['index'][1],
    grasp_plan['finger_close']['index'][2],
    grasp_plan['finger_close']['middle'][0], grasp_plan['finger_close']['middle'][1],
    grasp_plan['finger_close']['middle'][2],
], dtype=float)
state = planner._set_config_fk(x)
center = obj_info['center']
radius = np.max(obj_info['size'])
for fname in ['thumb', 'index', 'middle']:
    pos = state['tips'][fname]['pos']
    print(f"{fname} tip: {np.round(pos, 5)}  surface offset: {np.linalg.norm(pos - center) - radius:.5f}")

print("\n" + "=" * 60)
print("Visual demo")
print("=" * 60)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
controller = GraspController(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    controller._set_wrist_ctrl(np.zeros(6))
    for fname in ['thumb', 'index', 'middle']:
        controller._set_finger_ctrl(fname, np.zeros(len(controller._finger_act_ids[fname])))
    for _ in range(50):
        mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.5)

    print("\n[Approach] ...")
    ok = controller.execute_approach(grasp_plan, n_steps=80, viewer=viewer)
    print_snapshot("After Approach", data, ids, controller, obj_start)
    print(f"  OK: {ok}")

    time.sleep(0.3)

    print("\n[Close] ...")
    ok = controller.execute_close(grasp_plan=grasp_plan, force_threshold=0.2,
                                  close_speed=0.005, max_steps=500, viewer=viewer)
    print_snapshot("After Close", data, ids, controller, obj_start)
    print(f"  OK: {ok}")

    time.sleep(0.3)

    print("\n[Lift] ...")
    ok = controller.execute_lift(height=0.05, n_steps=60, viewer=viewer)
    print_snapshot("After Lift", data, ids, controller, obj_start)
    print(f"  OK: {ok}")

    if ok:
        print("\n>>> GRASP SUCCESS <<<")
    else:
        print("\n>>> GRASP FAILED <<<")

    print("\nClose viewer to exit...")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.02)
