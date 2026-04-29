# Three-Finger Grasp

This directory contains the MuJoCo-based three-finger grasping environment, SAC training code, evaluation scripts, and helper tools used for the current RL experiments.

## Main files

- `grasp_env.py`  
  Base MuJoCo environment. Defines object reset, reward, success logic, tray/table geometry handling, and contact statistics.

- `grasp_rl_env.py`  
  RL-facing wrapper. Builds the low-dimensional observation and action interface used by SAC.

- `sac_agent.py`  
  PyTorch SAC implementation, replay buffer, entropy/alpha schedule support, and critic stabilization logic.

- `train_grasp_sac.py`  
  Main SAC training entrypoint. Supports JSON config loading, TensorBoard logging, checkpointing, and staged alpha control.

- `eval_grasp_sac.py`  
  Deterministic checkpoint evaluation. Supports live rendering, slow playback, and optional video recording.

- `grasp_planner.py`  
  Geometric grasp planner used for pregrasp analysis and non-RL grasp generation.

- `grasp_controller.py`  
  Heuristic staged controller for approach / close / lift execution.

- `grasp_demo.py`  
  Manual demo script for inspecting planner and controller behavior.

- `_preview_env.py`  
  Fast scene preview for checking object placement, self-collision, and pregrasp pose.

- `_remote_tb_tunnel.py`  
  Local SSH forwarding helper for exposing remote TensorBoard on the local machine.

- `config_box_large_v4.json`  
  Main training config used for the current `box_large` experiments.

- `launch_tensorboard.ps1`  
  Convenience script to start local TensorBoard on Windows.

## Artifacts

- `released_checkpoints/`  
  Curated checkpoints intended to be kept in version control.

- `checkpoints_*`  
  Local experiment outputs. These are ignored by git and should be treated as transient training artifacts.

- `tb_runs/`  
  Local TensorBoard event files. Ignored by git.

- `logs/`  
  Runtime logs and local helper state. Ignored by git.

## Common commands

### 1. Preview the current scene

```powershell
D:\conda_envs\fkik_solve\python.exe D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\_preview_env.py
```

### 2. Train locally with the main config

```powershell
D:\conda_envs\fkik_solve\python.exe D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\train_grasp_sac.py --config D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\config_box_large_v4.json
```

### 3. Start local TensorBoard

```powershell
powershell -ExecutionPolicy Bypass -File D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\launch_tensorboard.ps1
```

Open:

```text
http://localhost:6006
```

### 4. Evaluate a local checkpoint with live rendering

```powershell
D:\conda_envs\fkik_solve\python.exe D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\eval_grasp_sac.py --checkpoint D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\released_checkpoints\box_large_successhold_v4_best_20260429\sac_best.pt --episodes 3 --render --slowdown 4.0
```

### 5. Record evaluation video

```powershell
D:\conda_envs\fkik_solve\python.exe D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\eval_grasp_sac.py --checkpoint D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\released_checkpoints\box_large_successhold_v4_best_20260429\sac_best.pt --episodes 5 --render --slowdown 1.0 --record-dir D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\recordings_eval_normal
```

### 6. Forward remote TensorBoard to local machine

```powershell
D:\conda_envs\fkik_solve\python.exe D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\_remote_tb_tunnel.py
```

Open:

```text
http://127.0.0.1:16006
```

## Current main experiment

- Object: `box_large`
- Wrist rotation: locked
- Alpha schedule: success-rate-driven staged schedule
- Current curated checkpoint:
  - `released_checkpoints/box_large_successhold_v4_best_20260429/sac_best.pt`
