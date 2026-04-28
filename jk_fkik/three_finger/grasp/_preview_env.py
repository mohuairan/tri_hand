"""Open a MuJoCo viewer on the current fixed-pregrasp grasp environment."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from grasp_rl_env import JackHandStateEnv

_DEFAULT_PREVIEW_OBJECT = "box_large"


def _self_collision_pairs(env: JackHandStateEnv):
    pairs = []
    model = env.base_env.model
    data = env.base_env.data
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        if "obj_" in (n1 or "") or "obj_" in (n2 or ""):
            continue
        pairs.append((n1, n2, float(contact.dist)))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-type",
        type=str,
        default=_DEFAULT_PREVIEW_OBJECT,
        choices=["sphere", "sphere_small", "cylinder", "box", "box_large"],
    )
    parser.add_argument("--simulate", action="store_true", help="Advance physics during preview")
    args = parser.parse_args()

    env = JackHandStateEnv(render_mode="human", object_type=args.object_type)
    try:
        obs, info = env.reset(seed=0)
        print("Preview reset info:")
        print(f"  obj_height={info['obj_height']:.6f}")
        print(f"  embedded={info['object_embedded']}")
        print(f"  off_table={info['object_off_table']}")
        print(f"  n_contacts={info['n_contacts']}")
        print(f"  object_type={info['object_type']}")
        base = env.base_env
        bottom_z = info["obj_height"] - base._current_object_vertical_extent()
        gap = bottom_z - base._table_surface_z()
        self_pairs = _self_collision_pairs(env)
        print(f"  bottom_z={bottom_z:.6f}")
        print(f"  table_surface_z={base._table_surface_z():.6f}")
        print(f"  table_gap={gap:.6f}")
        print(f"  total_ncon={int(base.data.ncon)}")
        print(f"  self_collision_count={len(self_pairs)}")
        for name1, name2, dist in self_pairs:
            print(f"    self_collision: {name1} vs {name2} dist={dist:.6f}")

        for _ in range(20000):
            if args.simulate:
                zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
                _, _, terminated, truncated, _ = env.step(zero_action)
                if terminated or truncated:
                    obs, info = env.reset(seed=0)
            env.render()
            time.sleep(0.02)
    finally:
        env.close()


if __name__ == "__main__":
    main()
