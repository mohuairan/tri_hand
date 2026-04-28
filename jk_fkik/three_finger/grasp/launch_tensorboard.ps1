param(
    [string]$LogDir = "D:\jodell\project\fkik_solve\jk_fkik\three_finger\grasp\tb_runs",
    [int]$Port = 6006
)

$python = "D:\conda_envs\fkik_solve\python.exe"

Write-Host "Launching TensorBoard..."
Write-Host "  LogDir: $LogDir"
Write-Host "  Port:   $Port"

& $python -m tensorboard.main --logdir $LogDir --port $Port
