"""
grasp_planner.py - grasp planning for the three-finger hand.

The old planner optimized only fingertip-to-center distance. This version
keeps the same public API shape but upgrades the objective toward enclosure:
surface contact, contact spread, palm-relative placement, normal alignment,
and mild posture regularization.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize


class GraspPlanner:
    """Heuristic grasp planner driven by MuJoCo forward kinematics."""

    _WRIST_JOINTS = [
        'wrist_x', 'wrist_y', 'wrist_z',
        'wrist_rx', 'wrist_ry', 'wrist_rz',
    ]
    _FINGER_JOINTS = [
        'thumb_q1', 'thumb_q2', 'thumb_q3', 'thumb_q4',
        'index_q1', 'index_q2', 'index_q3',
        'middle_q1', 'middle_q2', 'middle_q3',
    ]
    _TIP_NAMES = ('thumb', 'index', 'middle')

    def __init__(self, model):
        self.model = model
        self._plan_data = mujoco.MjData(model)
        self._cache_ids()

    def _cache_ids(self):
        m = self.model
        self._palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'palm')
        self._thumb_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'thumb_base')
        self._index_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'index_base')
        self._middle_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'middle_base')
        self._thumb_tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'thumb_tip')
        self._index_tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'index_tip')
        self._middle_tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'middle_tip')
        self._thumb_nt_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'thumb_NT')
        self._index_nt_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'index_NT')
        self._middle_nt_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'middle_NT')

        self._jnt_qposadr = {}
        for name in self._WRIST_JOINTS + self._FINGER_JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._jnt_qposadr[name] = m.jnt_qposadr[jid]

    @staticmethod
    def _safe_unit(v, eps=1e-8):
        n = np.linalg.norm(v)
        if n < eps:
            return np.zeros_like(v)
        return v / n

    def _set_config_fk(self, x):
        d = self._plan_data
        mujoco.mj_resetData(self.model, d)

        d.qpos[self._jnt_qposadr['wrist_x']] = x[0]
        d.qpos[self._jnt_qposadr['wrist_y']] = x[1]
        d.qpos[self._jnt_qposadr['wrist_z']] = x[2]
        d.qpos[self._jnt_qposadr['wrist_rx']] = x[3]
        d.qpos[self._jnt_qposadr['wrist_ry']] = x[4]
        d.qpos[self._jnt_qposadr['wrist_rz']] = x[5]

        d.qpos[self._jnt_qposadr['thumb_q1']] = np.radians(x[6])
        d.qpos[self._jnt_qposadr['thumb_q2']] = np.radians(x[7])
        d.qpos[self._jnt_qposadr['thumb_q3']] = np.radians(x[8])
        d.qpos[self._jnt_qposadr['thumb_q4']] = np.radians(x[9])

        d.qpos[self._jnt_qposadr['index_q1']] = np.radians(x[10])
        d.qpos[self._jnt_qposadr['index_q2']] = np.radians(x[11])
        d.qpos[self._jnt_qposadr['index_q3']] = np.radians(x[12])

        d.qpos[self._jnt_qposadr['middle_q1']] = np.radians(x[13])
        d.qpos[self._jnt_qposadr['middle_q2']] = np.radians(x[14])
        d.qpos[self._jnt_qposadr['middle_q3']] = np.radians(x[15])

        mujoco.mj_forward(self.model, d)

        return {
            'palm_pos': d.xpos[self._palm_id].copy(),
            'palm_mat': d.xmat[self._palm_id].reshape(3, 3).copy(),
            'bases': {
                'thumb': d.xpos[self._thumb_base_id].copy(),
                'index': d.xpos[self._index_base_id].copy(),
                'middle': d.xpos[self._middle_base_id].copy(),
            },
            'links': {
                'thumb_nt': d.geom_xpos[self._thumb_nt_geom_id].copy(),
                'index_nt': d.geom_xpos[self._index_nt_geom_id].copy(),
                'middle_nt': d.geom_xpos[self._middle_nt_geom_id].copy(),
            },
            'tips': {
                'thumb': {
                    'pos': d.xpos[self._thumb_tip_id].copy(),
                    'mat': d.xmat[self._thumb_tip_id].reshape(3, 3).copy(),
                },
                'index': {
                    'pos': d.xpos[self._index_tip_id].copy(),
                    'mat': d.xmat[self._index_tip_id].reshape(3, 3).copy(),
                },
                'middle': {
                    'pos': d.xpos[self._middle_tip_id].copy(),
                    'mat': d.xmat[self._middle_tip_id].reshape(3, 3).copy(),
                },
            },
        }

    def _surface_distance_and_normal(self, point, obj_info):
        center = obj_info['center']
        geom_type = obj_info.get('geom_type', mujoco.mjtGeom.mjGEOM_SPHERE)
        size = np.asarray(obj_info['size'], dtype=float)

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = float(size[0])
            radial = point - center
            return np.linalg.norm(radial) - radius, self._safe_unit(radial)

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            q = point - center
            abs_q = np.abs(q)
            ext = size
            outside = np.maximum(abs_q - ext, 0.0)
            outside_dist = np.linalg.norm(outside)
            inside_dist = np.min(ext - abs_q)
            if outside_dist > 1e-8:
                normal = self._safe_unit(q * (abs_q > ext))
                if np.linalg.norm(normal) < 1e-8:
                    normal = self._safe_unit(q)
                return outside_dist, normal
            axis = int(np.argmin(ext - abs_q))
            normal = np.zeros(3)
            normal[axis] = 1.0 if q[axis] >= 0 else -1.0
            return -inside_dist, normal

        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            q = point - center
            radius = float(size[0])
            half_h = float(size[2]) if size.shape[0] > 2 else float(size[1])
            radial_xy = np.linalg.norm(q[:2])
            dr = radial_xy - radius
            dz = abs(q[2]) - half_h
            outside = np.array([max(dr, 0.0), max(dz, 0.0)])
            outside_dist = np.linalg.norm(outside)
            if outside_dist > 1e-8:
                if dr >= dz and radial_xy > 1e-8:
                    normal = np.array([q[0] / radial_xy, q[1] / radial_xy, 0.0])
                else:
                    normal = np.array([0.0, 0.0, 1.0 if q[2] >= 0 else -1.0])
                return outside_dist, normal
            if dr > dz and radial_xy > 1e-8:
                return dr, np.array([q[0] / radial_xy, q[1] / radial_xy, 0.0])
            return dz, np.array([0.0, 0.0, 1.0 if q[2] >= 0 else -1.0])

        radius = float(np.max(size))
        radial = point - center
        return np.linalg.norm(radial) - radius, self._safe_unit(radial)

    def estimate_object_from_state(self, data):
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
        obj_pos = data.xpos[obj_id].copy()

        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'obj_geom')
        geom_type = self.model.geom_type[geom_id]
        geom_size = self.model.geom_size[geom_id].copy()

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = geom_size[0]
            size_3d = np.array([radius, radius, radius])
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            size_3d = geom_size.copy()
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            size_3d = np.array([geom_size[0], geom_size[0], geom_size[1]])
        else:
            size_3d = geom_size.copy()

        return {
            'center': obj_pos,
            'size': size_3d,
            'geom_type': geom_type,
        }

    def estimate_object_from_depth(self, depth, camera_K, camera_pos, camera_mat,
                                   depth_threshold=0.5):
        mask = (depth > 0.001) & (depth < depth_threshold)
        if np.sum(mask) < 20:
            return None

        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]

        vs, us = np.where(mask)
        zs = depth[mask]
        xs = (us - cx) * zs / fx
        ys = (vs - cy) * zs / fy
        pts_cam = np.stack([xs, ys, zs], axis=-1)
        pts_world = (camera_mat @ pts_cam.T).T + camera_pos

        center = np.mean(pts_world, axis=0)
        pts_centered = pts_world - center
        cov = pts_centered.T @ pts_centered / len(pts_centered)
        eigvals, _ = np.linalg.eigh(cov)
        half_extents = np.sqrt(eigvals[::-1]) * 2.0

        return {
            'center': center,
            'size': half_extents,
        }

    def _evaluate_grasp_cost(self, x, obj_info):
        state = self._set_config_fk(x)
        center = obj_info['center']
        tips = state['tips']
        tip_positions = np.array([tips[name]['pos'] for name in self._TIP_NAMES])

        surface_error = 0.0
        normal_error = 0.0
        surface_offsets = {}
        contact_normals = {}
        surface_weights = {'thumb': 3.0, 'index': 2.0, 'middle': 2.0}
        desired_clearance = 0.002
        for name in self._TIP_NAMES:
            tip_pos = tips[name]['pos']
            tip_axis = tips[name]['mat'][:, 2]
            offset, obj_normal = self._surface_distance_and_normal(tip_pos, obj_info)
            target_dir = self._safe_unit(center - tip_pos)

            signed_gap = offset - desired_clearance
            if signed_gap >= 0.0:
                gap_penalty = signed_gap ** 2
            else:
                # Strongly reject solutions that place fingertips inside the object.
                gap_penalty = 60.0 * (signed_gap ** 2) + 0.10 * (-signed_gap)

            surface_error += surface_weights[name] * gap_penalty
            normal_error += surface_weights[name] * (1.0 - np.dot(tip_axis, target_dir)) ** 2
            surface_offsets[name] = float(offset)
            contact_normals[name] = obj_normal

        centroid = np.mean(tip_positions, axis=0)
        enclosure_error = np.sum((centroid - center) ** 2)

        spread_error = 0.0
        radial_dirs = [self._safe_unit(tips[name]['pos'] - center) for name in self._TIP_NAMES]
        target_dot = np.cos(np.radians(80.0))
        for i in range(len(radial_dirs)):
            for j in range(i + 1, len(radial_dirs)):
                dot = float(np.dot(radial_dirs[i], radial_dirs[j]))
                spread_error += max(dot - target_dot, 0.0) ** 2

        palm_local = state['palm_mat'].T @ (center - state['palm_pos'])
        palm_target = np.array([-0.012, 0.0, 0.060])
        palm_weights = np.array([4.0, 1.0, 6.0])
        palm_error = np.sum(((palm_local - palm_target) * palm_weights) ** 2)
        if palm_local[2] < 0.015:
            palm_error += (0.015 - palm_local[2]) ** 2 * 25.0

        nominal = np.array([
            0.0527, -0.0162, -0.1263, 0.0490, 0.4023, 0.2584,
            -71.08, 0.0, -28.46, -20.15,
            -45.52, 0.73, -32.56,
            -43.76, -11.37, -36.88,
        ], dtype=float)
        scales = np.array([
            0.08, 0.08, 0.06, 0.30, 0.25, 0.25,
            20, 15, 15, 12,
            18, 10, 15,
            18, 10, 15,
        ], dtype=float)
        reg_error = np.sum(((x - nominal) / scales) ** 2)
        reg_error += ((x[10] - x[13]) / 90.0) ** 2
        reg_error += ((x[11] - x[14]) / 20.0) ** 2
        reg_error += ((x[12] - x[15]) / 90.0) ** 2

        total = (
            4.0 * surface_error +
            1.2 * normal_error +
            0.8 * enclosure_error +
            0.8 * spread_error +
            0.12 * palm_error +
            0.08 * reg_error
        )

        quality = {
            'surface_error': float(surface_error),
            'normal_error': float(normal_error),
            'enclosure_error': float(enclosure_error),
            'spread_error': float(spread_error),
            'palm_error': float(palm_error),
            'regularization_error': float(reg_error),
            'surface_offsets': surface_offsets,
            'palm_local_center': palm_local,
        }
        return float(total), quality, state, contact_normals

    def plan_grasp(self, obj_info):
        obj_center = obj_info['center']
        obj_radius = np.max(obj_info['size'])

        def cost(x):
            total, _, _, _ = self._evaluate_grasp_cost(x, obj_info)
            return total

        seeds = [
            np.array([
                0.0527, -0.0162, -0.1263, 0.0490, 0.4023, 0.2584,
                -71.08, 0.0, -28.46, -20.15,
                -45.52, 0.73, -32.56,
                -43.76, -11.37, -36.88,
            ], dtype=float),
            np.array([
                -0.01, -0.03, -0.10, 0.0, 0.0, 0.0,
                -60, -20, -45, -25,
                -55, 0, -40,
                -55, 0, -40,
            ], dtype=float),
        ]

        bounds = [
            (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.0),
            (-0.50, 0.50), (-0.50, 0.50), (-0.80, 0.80),
            (-95, -20), (-55, 0), (-86, 0), (-70, 0),
            (-90, -10), (-20, 20), (-90, 0),
            (-90, -10), (-20, 20), (-90, 0),
        ]

        best_res = None
        for x0 in seeds:
            res = minimize(
                cost, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if best_res is None:
                best_res = res
                continue

            new_rank = (0 if res.success else 1, float(res.fun))
            best_rank = (0 if best_res.success else 1, float(best_res.fun))
            if new_rank < best_rank:
                best_res = res

        res = best_res
        x = res.x
        _, quality, state, contact_normals = self._evaluate_grasp_cost(x, obj_info)

        wrist_target = np.array([x[0], x[1], x[2], x[3], x[4], x[5]])
        finger_close = {
            'thumb': np.array([x[6], x[7], x[8], x[9]]),
            'index': np.array([x[10], x[11], x[12]]),
            'middle': np.array([x[13], x[14], x[15]]),
        }
        finger_pregrasp = {name: angles * 0.35 for name, angles in finger_close.items()}

        return {
            'wrist_target': wrist_target,
            'finger_pregrasp': finger_pregrasp,
            'finger_close': finger_close,
            'obj_center': obj_center,
            'obj_radius': obj_radius,
            'opt_cost': float(res.fun),
            'optimizer_success': bool(res.success),
            'optimizer_message': str(res.message),
            'optimizer_seed_count': len(seeds),
            'quality': quality,
            'contact_targets': {
                name: {
                    'position': state['tips'][name]['pos'],
                    'normal': contact_normals[name],
                }
                for name in self._TIP_NAMES
            },
        }
