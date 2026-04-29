# Curated Checkpoint: box_large_successhold_v4_best_20260429

This folder stores a manually selected checkpoint from the local `box_large_successhold_v4_final_20260428_214724` training run.

## File

- `sac_best.pt`

## Source run

- Original directory:
  - `jk_fkik/three_finger/grasp/checkpoints_sac_box_large_successhold_v4_final/box_large_successhold_v4_final_20260428_214724`

## Snapshot metrics

- `step = 290000`
- `success_rate = 0.8`
- `success_rate_ema = 0.2115`
- `alpha_phase = 3`
- `alpha = 0.03`
- `height_gain_mean = 0.0224`
- `contacts_mean = 3.0`
- `lateral_drift_mean = 0.0360`
- `spin_rate_mean = 16.30`
- `critic_loss = 53.63`
- `q1_mean = 187.74`
- `q2_mean = 187.41`

## Evaluation example

```powershell
D:\conda_envs\fkik_solve\python.exe D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\eval_grasp_sac.py --checkpoint D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\released_checkpoints\box_large_successhold_v4_best_20260429\sac_best.pt --episodes 5 --render --slowdown 4.0
```
