"""Check pregrasp tip heights at various wrist heights."""
import numpy as np
import mujoco
import os

xml = os.path.join(os.path.dirname(__file__), '..', 'mujoco_model', 'jack_hand_3f.xml')
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)

palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'palm')
thumb_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thumb_tip')
index_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'index_tip')
middle_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'middle_tip')


def set_joint(name, val_deg):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = np.radians(val_deg)


def set_wrist(x=0, y=0, z=0):
    for n, v in [('wrist_x', x), ('wrist_y', y), ('wrist_z', z)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        data.qpos[model.jnt_qposadr[jid]] = v


# Pregrasp angles from planner
pregrasp = {
    'thumb': (-47, 0, -24, -11),
    'index': (-28, 0, -20),
    'middle': (-30, 0, -23),
}

print('=== Pregrasp tip heights as wrist descends ===')
print('Table top at z=0.250, Object at z=0.265, Object radius=0.015')
print()
for wz in np.arange(0, -0.16, -0.01):
    mujoco.mj_resetData(model, data)
    set_wrist(x=-0.0325, y=-0.0015, z=wz)
    set_joint('thumb_q1', pregrasp['thumb'][0])
    set_joint('thumb_q2', pregrasp['thumb'][1])
    set_joint('thumb_q3', pregrasp['thumb'][2])
    set_joint('thumb_q4', pregrasp['thumb'][3])
    set_joint('index_q1', pregrasp['index'][0])
    set_joint('index_q3', pregrasp['index'][2])
    set_joint('middle_q1', pregrasp['middle'][0])
    set_joint('middle_q3', pregrasp['middle'][2])
    mujoco.mj_forward(model, data)
    pz = data.xpos[palm_id][2]
    tz = data.xpos[thumb_tip_id][2]
    iz = data.xpos[index_tip_id][2]
    mz = data.xpos[middle_tip_id][2]
    min_z = min(tz, iz, mz)
    marker = ' <-- COLLISION' if min_z < 0.250 else ''
    print(f'  wz={wz:+.2f}  palm_z={pz:.3f}  tips_z=[T={tz:.3f}, I={iz:.3f}, M={mz:.3f}]  min={min_z:.3f}{marker}')

# Also check with straight fingers
print('\n=== Straight fingers as wrist descends ===')
for wz in np.arange(0, -0.16, -0.01):
    mujoco.mj_resetData(model, data)
    set_wrist(x=-0.0325, y=-0.0015, z=wz)
    mujoco.mj_forward(model, data)
    pz = data.xpos[palm_id][2]
    tz = data.xpos[thumb_tip_id][2]
    iz = data.xpos[index_tip_id][2]
    mz = data.xpos[middle_tip_id][2]
    min_z = min(tz, iz, mz)
    marker = ' <-- COLLISION' if min_z < 0.250 else ''
    print(f'  wz={wz:+.2f}  palm_z={pz:.3f}  tips_z=[T={tz:.3f}, I={iz:.3f}, M={mz:.3f}]  min={min_z:.3f}{marker}')

# Close config tip heights
close = {
    'thumb': (-95, 0, -48.5, -23),
    'index': (-57, 0, -41),
    'middle': (-61, 0, -46),
}
print('\n=== Close config tip heights as wrist descends ===')
for wz in np.arange(0, -0.16, -0.01):
    mujoco.mj_resetData(model, data)
    set_wrist(x=-0.0325, y=-0.0015, z=wz)
    set_joint('thumb_q1', close['thumb'][0])
    set_joint('thumb_q2', close['thumb'][1])
    set_joint('thumb_q3', close['thumb'][2])
    set_joint('thumb_q4', close['thumb'][3])
    set_joint('index_q1', close['index'][0])
    set_joint('index_q3', close['index'][2])
    set_joint('middle_q1', close['middle'][0])
    set_joint('middle_q3', close['middle'][2])
    mujoco.mj_forward(model, data)
    pz = data.xpos[palm_id][2]
    tz = data.xpos[thumb_tip_id][2]
    iz = data.xpos[index_tip_id][2]
    mz = data.xpos[middle_tip_id][2]
    ty = data.xpos[thumb_tip_id][1]
    iy = data.xpos[index_tip_id][1]
    my = data.xpos[middle_tip_id][1]
    print(f'  wz={wz:+.2f}  palm_z={pz:.3f}  tips_z=[T={tz:.3f}, I={iz:.3f}, M={mz:.3f}]  tips_y=[T={ty:.3f}, I={iy:.3f}, M={my:.3f}]')
