Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================"

Write-Host ""
Write-Host "--- Python Processes ---" -ForegroundColor Yellow
$procs = Get-Process python* -ErrorAction SilentlyContinue
if ($procs) {
    foreach ($p in $procs) {
        $mem = "{0:N0} MB" -f ($p.WorkingSet64 / 1MB)
        $cpu = "{0:N0} sec" -f $p.CPU
        Write-Host "  PID $($p.Id)  CPU=$cpu  Mem=$mem  Start=$($p.StartTime)"
    }
} else {
    Write-Host "  None" -ForegroundColor Red
}

Write-Host ""
Write-Host "--- GPU ---" -ForegroundColor Yellow
$gpu = nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>$null
if ($gpu) { $gpu } else { Write-Host "  nvidia-smi not available" -ForegroundColor Red }

Write-Host ""
Write-Host "--- Checkpoints ---" -ForegroundColor Yellow
$roots = @(
    "CARE-E2E-Fusion\results_real",
    "项目_1_MoTo-CARE\results_real",
    "项目_2_PathwayMorph-OT\results_real"
)
foreach ($r in $roots) {
    $path = Join-Path $PSScriptRoot $r
    $name = ($r.Split('\')[0]).PadRight(18)
    if (-not (Test-Path $path)) {
        Write-Host "  $name no dir"
        continue
    }
    $sub = Get-ChildItem $path -Directory | Sort LastWriteTime -Desc | Select -First 1
    if (-not $sub) {
        Write-Host "  $name empty"
        continue
    }
    $pts = Get-ChildItem $sub.FullName -Filter "fold_*_best.pt" | Sort Name
    $lasts = Get-ChildItem $sub.FullName -Filter "fold_*_last.pt" | Sort LastWriteTime -Desc | Select -First 1
    $bestStr = if ($pts) { ($pts | ForEach-Object { $_.Name -replace 'fold_(\d)_.*','F$1' }) -join ' ' } else { '-' }
    $lastAge = if ($lasts) { "{0:N0} min" -f ((Get-Date) - $lasts.LastWriteTime).TotalMinutes } else { '-' }
    Write-Host "  $name best=[$bestStr]  last=<$lastAge>"
}
Write-Host ""
