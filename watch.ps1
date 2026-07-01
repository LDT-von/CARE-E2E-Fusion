<#
.SYNOPSIS
  一次性打印训练状态的监看脚本
  用法: powershell -File watch.ps1
#>

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$projects = @(
    @{Name="CARE-E2E-Fusion";  ResDir="$root\CARE-E2E-Fusion\results_real"},
    @{Name="MoTo-CARE";        ResDir="$root\项目_1_MoTo-CARE\results_real"},
    @{Name="PathwayMorph-OT";  ResDir="$root\项目_2_PathwayMorph-OT\results_real"}
)

Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "============================================================="
Write-Host ""

# GPU
Write-Host "--- GPU ---" -ForegroundColor Yellow
$nvidia = nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>$null
if ($nvidia) { $nvidia | ForEach-Object { Write-Host "  $_" } } else { Write-Host "  n/a" }
Write-Host ""

# Python 进程
Write-Host "--- Python 进程 ---" -ForegroundColor Yellow
$procs = Get-Process python* -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -eq "" } |
    Select-Object Id, @{N="Mem(MB)";E={[math]::Round($_.WorkingSet64/1MB,0)}}, StartTime
if ($procs) { $procs | Format-Table | Out-String | Write-Host } else { Write-Host "  无" }
Write-Host ""

# Checkpoint 文件
Write-Host "--- 最新 Checkpoint ---" -ForegroundColor Yellow
foreach ($p in $projects) {
    $name = $p.Name.PadRight(18)
    if (-not (Test-Path $p.ResDir)) { Write-Host "  ${name} 无结果目录"; continue }
    $sub = Get-ChildItem $p.ResDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $sub) { Write-Host "  ${name} 无实验子目录"; continue }
    $pts = Get-ChildItem $sub.FullName -Filter "fold_*_best.pt" | Sort-Object Name
    $lasts = Get-ChildItem $sub.FullName -Filter "fold_*_last.pt" | Sort-Object Name
    $bestInfo = if ($pts) { ($pts | ForEach-Object { $_.Name -replace 'fold_(\d)_.*','Fold$1' }) -join ' ' } else { "无" }
    $lastInfo = if ($lasts) { ($lasts | ForEach-Object { 
        $n = $_.Name -replace 'fold_(\d)_.*','$1'
        $age = [math]::Round(((Get-Date) - $_.LastWriteTime).TotalMinutes, 0)
        "Fold${n}(${age}min前)"
    }) -join ' ' } else { "无" }
    Write-Host "  ${name} best: $bestInfo | last: $lastInfo | $($sub.Name)"
}
Write-Host ""
