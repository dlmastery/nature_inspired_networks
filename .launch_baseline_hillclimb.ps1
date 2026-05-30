$ErrorActionPreference = 'Continue'
Set-Location C:\Users\evija\sacgeometry
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/baseline_hillclimb_$ts.log"
"=== baseline_resnet20 hill-climb launch $ts ===" | Tee-Object -FilePath $logFile
"=== Required for fair Δ comparison vs hill-climbed leaders (BLOCKER #13) ===" | Tee-Object -FilePath $logFile -Append

# Create the ideas/00_baseline_resnet20 scaffold if missing
if (-not (Test-Path "ideas\00_baseline_resnet20")) {
    New-Item -ItemType Directory -Path "ideas\00_baseline_resnet20\dashboard" -Force | Out-Null
    "# Baseline ResNet-20 hill-climb (calibration)" | Out-File -FilePath "ideas\00_baseline_resnet20\README.md" -Encoding utf8
}

& .\.venv\Scripts\python -u scripts\run_hillclimb.py `
    --tag baseline_resnet20 `
    --idea 00_baseline_resnet20 `
    --config configs\cifar100_phase4.yaml `
    --algorithm coordinate `
    --budget 25 `
    --epochs 30 `
    --seeds 0 1 2 `
    --root experiments 2>&1 | Tee-Object -FilePath $logFile -Append

"=== baseline_resnet20 hill-climb done at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
