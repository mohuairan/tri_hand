"""Interactive grasp demo with detailed phase diagnostics."""

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


def _collect_ids(model):
    return {
        'object': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target_object'),
        'palm': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'palm'),
        'thumb_tip': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thumb_tip'),
        'index_tip': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'index_tip'),
        'middle_tip': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'middle_tip'),
    }


def _print_plan_summary(grasp_plan):
    print("\n[Plan] Grasp planner output")
    print(f"  Optimizer cost: {grasp_plan['opt_cost']:.6f}")
    print(f"  Optimizer success: {grasp_plan['optimizer_success']}")
    print(f"  Optimizer message: {grasp_plan['optimizer_message']}")
    print(f"  Wrist target: {np.round(grasp_plan['wrist_target'], 4)}")
    for name in ('thumb', 'index', 'middle'):
        print(f"  {name} pregrasp: {np.round(grasp_plan['finger_pregrasp'][name], 2)}")
    for name in ('thumb', 'index', 'middle'):
        print(f"  {name} close:    {np.round(grasp_plan['finger_close'][name], 2)}")

    quality = grasp_plan.get('quality', {})
    if quality:
        print("  Quality:")
        print(f"    surface_error={quality['surface_error']:.6f}")
        print(f"    normal_error={quality['normal_error']:.6f}")
        print(f"    enclosure_error={quality['enclosure_error']:.6f}")
        print(f"    spread_error={quality['spread_error']:.6f}")
        print(f"    palm_error={quality['palm_error']:.6f}")
        print(f"    regularization_error={quality['regularization_error']:.6f}")
        print(f"    palm_local_center={np.round(quality['palm_local_center'], 4)}")
        print(f"    surface_offsets={quality['surface_offsets']}")


def _print_snapshot(label, data, ids, controller, obj_ref):
    obj_pos = data.xpos[ids['object']].copy()
    palm_pos = data.xpos[ids['palm']].copy()
    thumb_pos = data.xpos[ids['thumb_tip']].copy()
    index_pos = data.xpos[ids['index_tip']].copy()
    middle_pos = data.xpos[ids['middle_tip']].copy()
    touch = controller._get_touch_forces()
    object_contacts, geom_hits = controller._get_object_contact_info()

    delta = obj_pos - obj_ref
    lateral_shift = np.linalg.norm(delta[:2])
    tip_dists = {
        'thumb': float(np.linalg.norm(thumb_pos - obj_pos)),
        'index': float(np.linalg.norm(index_pos - obj_pos)),
        'middle': float(np.linalg.norm(middle_pos - obj_pos)),
    }

    print(f"\n[{label}]")
    print(f"  Object pos: {np.round(obj_pos, 5)}")
    print(f"  Palm pos:   {np.round(palm_pos, 5)}")
    print(f"  Object delta: {np.round(delta, 5)}  lateral={lateral_shift:.5f} m")
    print(f"  Touch: thumb={touch[0]:.4f} index={touch[1]:.4f} middle={touch[2]:.4f} N")
    print(f"  Object contacts: {object_contacts}")
    print(f"  Contact geoms: {geom_hits}")
    print(
        "  Tip->object center: "
        f"thumb={tip_dists['thumb']:.5f} "
        f"index={tip_dists['index']:.5f} "
        f"middle={tip_dists['middle']:.5f} m"
    )

    if lateral_shift > 0.01:
        print("  Warning: object has already shifted laterally by more than 1 cm.")
    if obj_pos[2] < obj_ref[2] - 0.01:
        print("  Warning: object height dropped noticeably from the start pose.")


def main():
    xml_path = os.path.join(os.path.dirname(__file__), '..', 'mujoco_model', 'scene.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = [0, 0, -9.81]
    mujoco.mj_forward(model, data)

    ids = _collect_ids(model)
    obj_start = data.xpos[ids['object']].copy()

    print("=" * 64)
    print("Three-finger grasp demo with diagnostics")
    print("=" * 64)
    print(f"Initial object position: {np.round(obj_start, 5)}")
    print(f"Initial palm position:   {np.round(data.xpos[ids['palm']].copy(), 5)}")

    planner = GraspPlanner(model)
    obj_info = planner.estimate_object_from_state(data)
    print(f"Detected object center: {np.round(obj_info['center'], 5)}")
    print(f"Detected object size:   {np.round(obj_info['size'], 5)}")

    grasp_plan = planner.plan_grasp(obj_info)
    _print_plan_summary(grasp_plan)

    controller = GraspController(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        controller._set_wrist_ctrl(np.zeros(6))
        for name in ('thumb', 'index', 'middle'):
            controller._set_finger_ctrl(name, np.zeros(len(controller._finger_act_ids[name])))
        for _ in range(50):
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.5)

        print("\n[Run] Executing approach")
        ok = controller.execute_approach(grasp_plan, n_steps=80, viewer=viewer)
        _print_snapshot("After Approach", data, ids, controller, obj_start)
        print(f"  Approach ok: {ok}")
        if not ok:
            return

        time.sleep(0.5)

        print("\n[Run] Executing close")
        ok = controller.execute_close(
            grasp_plan=grasp_plan,
            force_threshold=0.2,
            close_speed=0.015,
            max_steps=400,
            viewer=viewer,
        )
        _print_snapshot("After Close", data, ids, controller, obj_start)
        print(f"  Close ok: {ok}")

        time.sleep(0.5)

        print("\n[Run] Executing lift")
        success = controller.execute_lift(height=0.05, n_steps=60, viewer=viewer)
        _print_snapshot("After Lift", data, ids, controller, obj_start)
        print(f"  Lift success: {success}")

        if success:
            print("\n>>> GRASP SUCCESS <<<")
        else:
            print("\n>>> GRASP FAILED <<<")

        print("\nViewer remains open for inspection.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)


if __name__ == '__main__':
    main()
