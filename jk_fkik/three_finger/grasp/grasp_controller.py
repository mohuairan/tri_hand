"""
grasp_controller.py - execution controller for the three-finger grasp demo.

This controller is intentionally conservative:
- keep the fingers open while the wrist is still translating/downward moving
- finish pregrasp curling at a fixed wrist pose
- close thumb first to establish opposition
- freeze any finger that has already touched, instead of continuing to shove
"""

import numpy as np
import mujoco
import time


def _minimum_jerk(t):
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


_WRIST_JOINTS = ['wrist_x', 'wrist_y', 'wrist_z',
                 'wrist_rx', 'wrist_ry', 'wrist_rz']
_WRIST_ACTS = ['wrist_act_x', 'wrist_act_y', 'wrist_act_z',
               'wrist_act_rx', 'wrist_act_ry', 'wrist_act_rz']

_FINGER_CONFIG = {
    'thumb': {
        'joints': ['thumb_q1', 'thumb_q2', 'thumb_q3', 'thumb_q4'],
        'acts': ['thumb_act_q1', 'thumb_act_q2', 'thumb_act_q3', 'thumb_act_q4'],
    },
    'index': {
        'joints': ['index_q1', 'index_q2', 'index_q3'],
        'acts': ['index_act_q1', 'index_act_q2', 'index_act_q3'],
    },
    'middle': {
        'joints': ['middle_q1', 'middle_q2', 'middle_q3'],
        'acts': ['middle_act_q1', 'middle_act_q2', 'middle_act_q3'],
    },
}

_TOUCH_SENSORS = ['thumb_touch', 'index_touch', 'middle_touch']
_CLOSE_JOINTS = {
    'thumb': ['thumb_q1', 'thumb_q3', 'thumb_q4'],
    'index': ['index_q1', 'index_q3'],
    'middle': ['middle_q1', 'middle_q3'],
}
_FINGER_NAMES = ['thumb', 'index', 'middle']


class GraspController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._cache_ids()

    def _cache_ids(self):
        m = self.model
        self._wrist_act_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in _WRIST_ACTS
        ]
        self._wrist_jnt_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in _WRIST_JOINTS
        ]

        self._finger_act_ids = {}
        self._finger_jnt_ids = {}
        for fname, cfg in _FINGER_CONFIG.items():
            self._finger_act_ids[fname] = [
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in cfg['acts']
            ]
            self._finger_jnt_ids[fname] = [
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in cfg['joints']
            ]

        self._touch_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
            for n in _TOUCH_SENSORS
        ]

        self._close_act_ids = {}
        for fname, jnames in _CLOSE_JOINTS.items():
            self._close_act_ids[fname] = [
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                  n.replace('_q', '_act_q'))
                for n in jnames
            ]

        obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
        self._obj_jnt_adr = m.body_jntadr[obj_body]
        self._obj_qpos_adr = m.jnt_qposadr[self._obj_jnt_adr]
        self._obj_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'obj_geom')

        self._geom_to_finger = {}
        for gid in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            if name.startswith('thumb_'):
                self._geom_to_finger[gid] = 'thumb'
            elif name.startswith('index_'):
                self._geom_to_finger[gid] = 'index'
            elif name.startswith('middle_'):
                self._geom_to_finger[gid] = 'middle'

    def _get_touch_forces(self):
        forces = np.zeros(3)
        for i, sid in enumerate(self._touch_ids):
            adr = self.model.sensor_adr[sid]
            forces[i] = self.data.sensordata[adr]
        return forces

    def _get_object_contact_info(self):
        contacts = {fname: False for fname in _FINGER_NAMES}
        geom_hits = {fname: [] for fname in _FINGER_NAMES}

        for ic in range(self.data.ncon):
            contact = self.data.contact[ic]
            if contact.geom1 == self._obj_geom_id:
                other = contact.geom2
            elif contact.geom2 == self._obj_geom_id:
                other = contact.geom1
            else:
                continue

            fname = self._geom_to_finger.get(other)
            if fname is None:
                continue

            contacts[fname] = True
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
            if gname is not None:
                geom_hits[fname].append(gname)

        return contacts, geom_hits

    def _set_wrist_ctrl(self, target):
        for i, aid in enumerate(self._wrist_act_ids):
            lo = self.model.actuator_ctrlrange[aid, 0]
            hi = self.model.actuator_ctrlrange[aid, 1]
            self.data.ctrl[aid] = np.clip(target[i], lo, hi)

    def _set_finger_ctrl(self, finger_name, angles_rad):
        aids = self._finger_act_ids[finger_name]
        for i, aid in enumerate(aids):
            lo = self.model.actuator_ctrlrange[aid, 0]
            hi = self.model.actuator_ctrlrange[aid, 1]
            self.data.ctrl[aid] = np.clip(angles_rad[i], lo, hi)

    def _get_finger_ctrl(self, finger_name):
        aids = self._finger_act_ids[finger_name]
        return np.array([self.data.ctrl[aid] for aid in aids], dtype=float)

    def _get_wrist_pos(self):
        pos = np.zeros(6)
        for i, jid in enumerate(self._wrist_jnt_ids):
            qadr = self.model.jnt_qposadr[jid]
            pos[i] = self.data.qpos[qadr]
        return pos

    def _step_sim(self, n=5):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def _sync_viewer(self, viewer, dt=0.02):
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep(dt)

    @staticmethod
    def _blend(start, target, s):
        return (1.0 - s) * start + s * target

    def _freeze_on_contact(self, target_cmds, contact_threshold):
        forces = self._get_touch_forces()
        object_contacts, _ = self._get_object_contact_info()
        for fi, fname in enumerate(_FINGER_NAMES):
            if forces[fi] >= contact_threshold or object_contacts[fname]:
                target_cmds[fname] = self._get_finger_ctrl(fname)
        return forces, target_cmds

    def execute_approach(self, grasp_plan, n_steps=80, viewer=None):
        wrist_target = grasp_plan['wrist_target']
        finger_pregrasp = grasp_plan['finger_pregrasp']

        wrist_start = self._get_wrist_pos()
        wrist_above = wrist_target.copy()
        wrist_above[2] = wrist_start[2]

        wrist_precontact = wrist_target.copy()
        wrist_precontact[2] = min(wrist_start[2], wrist_target[2] + 0.025)

        open_cmd = {
            fname: np.zeros(len(self._finger_act_ids[fname]), dtype=float)
            for fname in _FINGER_NAMES
        }
        light_pregrasp = {
            fname: np.radians(angles_deg) * 0.18
            for fname, angles_deg in finger_pregrasp.items()
        }
        full_pregrasp = {
            fname: np.radians(angles_deg)
            for fname, angles_deg in finger_pregrasp.items()
        }

        phases = [
            (40, wrist_start, wrist_above, open_cmd, open_cmd, 10, 0.02),
            (max(20, n_steps // 2), wrist_above, wrist_precontact,
             open_cmd, light_pregrasp, 10, 0.02),
            (max(20, n_steps // 2), wrist_precontact, wrist_target,
             light_pregrasp, light_pregrasp, 10, 0.02),
        ]

        for n_phase, wrist_a, wrist_b, finger_a, finger_b, sim_steps, dt in phases:
            t = np.linspace(0.0, 1.0, n_phase)
            s = _minimum_jerk(t)
            for i in range(n_phase):
                if viewer is not None and not viewer.is_running():
                    return False
                self._set_wrist_ctrl(self._blend(wrist_a, wrist_b, s[i]))
                for fname in _FINGER_NAMES:
                    cmd = self._blend(finger_a[fname], finger_b[fname], s[i])
                    self._set_finger_ctrl(fname, cmd)
                self._step_sim(sim_steps)
                self._sync_viewer(viewer, dt=dt)

        # Curl into pregrasp only after the wrist has stopped translating.
        start_ctrl = {fname: self._get_finger_ctrl(fname) for fname in _FINGER_NAMES}
        t = np.linspace(0.0, 1.0, 35)
        s = _minimum_jerk(t)
        for i in range(len(t)):
            if viewer is not None and not viewer.is_running():
                return False
            self._set_wrist_ctrl(wrist_target)
            target_cmds = {
                fname: self._blend(start_ctrl[fname], full_pregrasp[fname], s[i])
                for fname in _FINGER_NAMES
            }
            _, target_cmds = self._freeze_on_contact(target_cmds, contact_threshold=0.03)
            for fname in _FINGER_NAMES:
                self._set_finger_ctrl(fname, target_cmds[fname])
            self._step_sim(8)
            self._sync_viewer(viewer, dt=0.015)

        return True

    def execute_close(self, grasp_plan=None, force_threshold=0.3,
                      close_speed=0.005, max_steps=500, viewer=None):
        if grasp_plan is not None:
            finger_close = grasp_plan['finger_close']
            close_targets = {
                fname: np.radians(finger_close[fname])
                for fname in _FINGER_NAMES
            }
            start_ctrl = {fname: self._get_finger_ctrl(fname) for fname in _FINGER_NAMES}

            # Phase A: thumb establishes opposition first.
            t = np.linspace(0.0, 1.0, 30)
            s = _minimum_jerk(t)
            for i in range(len(t)):
                if viewer is not None and not viewer.is_running():
                    return False
                target_cmds = {
                    'thumb': self._blend(start_ctrl['thumb'], close_targets['thumb'], s[i]),
                    'index': start_ctrl['index'].copy(),
                    'middle': start_ctrl['middle'].copy(),
                }
                _, target_cmds = self._freeze_on_contact(target_cmds, contact_threshold=0.03)
                for fname in _FINGER_NAMES:
                    self._set_finger_ctrl(fname, target_cmds[fname])
                self._step_sim(6)
                self._sync_viewer(viewer, dt=0.012)

            phase_b_start = {fname: self._get_finger_ctrl(fname) for fname in _FINGER_NAMES}
            # Phase B: index and middle close after the thumb has taken the lower side.
            for i in range(len(t)):
                if viewer is not None and not viewer.is_running():
                    return False
                target_cmds = {
                    'thumb': phase_b_start['thumb'].copy(),
                    'index': self._blend(phase_b_start['index'], close_targets['index'], s[i]),
                    'middle': self._blend(phase_b_start['middle'], close_targets['middle'], s[i]),
                }
                _, target_cmds = self._freeze_on_contact(target_cmds, contact_threshold=0.03)
                for fname in _FINGER_NAMES:
                    self._set_finger_ctrl(fname, target_cmds[fname])
                self._step_sim(6)
                self._sync_viewer(viewer, dt=0.012)

            phase_c_start = {fname: self._get_finger_ctrl(fname) for fname in _FINGER_NAMES}
            # Phase C: small settle to the exact close targets.
            t = np.linspace(0.0, 1.0, 20)
            s = _minimum_jerk(t)
            for i in range(len(t)):
                if viewer is not None and not viewer.is_running():
                    return False
                target_cmds = {
                    fname: self._blend(phase_c_start[fname], close_targets[fname], s[i])
                    for fname in _FINGER_NAMES
                }
                _, target_cmds = self._freeze_on_contact(target_cmds, contact_threshold=0.04)
                for fname in _FINGER_NAMES:
                    self._set_finger_ctrl(fname, target_cmds[fname])
                self._step_sim(6)
                self._sync_viewer(viewer, dt=0.012)

            forces = self._get_touch_forces()
            object_contacts, _ = self._get_object_contact_info()
            if all((f >= force_threshold) or object_contacts[fname]
                   for f, fname in zip(forces, _FINGER_NAMES)):
                return True

        finger_done = [False, False, False]
        incremental_speed = close_speed * 0.5

        for _ in range(max_steps):
            if viewer is not None and not viewer.is_running():
                return False

            forces = self._get_touch_forces()
            object_contacts, _ = self._get_object_contact_info()

            for fi, fname in enumerate(_FINGER_NAMES):
                if finger_done[fi]:
                    continue

                if forces[fi] >= force_threshold or object_contacts[fname]:
                    finger_done[fi] = True
                    continue

                for aid in self._close_act_ids[fname]:
                    lo = self.model.actuator_ctrlrange[aid, 0]
                    self.data.ctrl[aid] = max(self.data.ctrl[aid] - incremental_speed, lo)

            self._step_sim(5)
            self._sync_viewer(viewer, dt=0.01)

            if all(finger_done):
                return True

        n_contact = sum(1 for done in finger_done if done)
        return n_contact >= 2

    def execute_lift(self, height=0.05, n_steps=40, viewer=None):
        obj_adr = self._obj_qpos_adr
        initial_obj_z = self.data.qpos[obj_adr + 2]

        wrist_z_aid = self._wrist_act_ids[2]
        wrist_z_start = self.data.ctrl[wrist_z_aid]
        wrist_z_target = wrist_z_start + height
        hold_close_speed = 0.0015

        # Small preload before lifting so the grasp is not right on the edge of slip.
        for _ in range(20):
            if viewer is not None and not viewer.is_running():
                return False
            for fname in _FINGER_NAMES:
                for aid in self._close_act_ids[fname]:
                    lo = self.model.actuator_ctrlrange[aid, 0]
                    self.data.ctrl[aid] = max(self.data.ctrl[aid] - hold_close_speed, lo)
            self._step_sim(4)
            self._sync_viewer(viewer, dt=0.01)

        t = np.linspace(0.0, 1.0, n_steps)
        s = _minimum_jerk(t)

        for i in range(n_steps):
            if viewer is not None and not viewer.is_running():
                return False

            target_z = self._blend(wrist_z_start, wrist_z_target, s[i])
            lo = self.model.actuator_ctrlrange[wrist_z_aid, 0]
            hi = self.model.actuator_ctrlrange[wrist_z_aid, 1]
            self.data.ctrl[wrist_z_aid] = np.clip(target_z, lo, hi)

            for fname in _FINGER_NAMES:
                for aid in self._close_act_ids[fname]:
                    lo = self.model.actuator_ctrlrange[aid, 0]
                    self.data.ctrl[aid] = max(self.data.ctrl[aid] - hold_close_speed, lo)

            self._step_sim(10)
            self._sync_viewer(viewer)

        final_obj_z = self.data.qpos[obj_adr + 2]
        height_gain = final_obj_z - initial_obj_z
        return height_gain > height * 0.3

    def execute_full_grasp(self, grasp_plan, viewer=None):
        result = {'approach': False, 'close': False, 'lift': False, 'success': False}

        result['approach'] = self.execute_approach(grasp_plan, viewer=viewer)
        if not result['approach']:
            return result

        if viewer is not None:
            time.sleep(0.3)

        result['close'] = self.execute_close(grasp_plan=grasp_plan, viewer=viewer)
        if not result['close']:
            return result

        if viewer is not None:
            time.sleep(0.3)

        result['lift'] = self.execute_lift(viewer=viewer)
        result['success'] = result['lift']

        return result
