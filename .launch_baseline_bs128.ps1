$ErrorActionPreference = 'Continue'
Set-Location C:\Users\evija\sacgeometry
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/baseline_bs128_$ts.log"
"=== baseline_resnet20 at bs=128 launch $ts (R2 BS confound fix) ===" | Tee-Object -FilePath $logFile
"=== 7 seeds at default lr=0.001 wd=0.0005 bs=128 AdamW (R2 Q2) ===" | Tee-Object -FilePath $logFile -Append
& .\.venv\Scripts\python -u scripts\run_hillclimb.py --tag baseline_resnet20 --idea 00_baseline_resnet20 --config configs\cifar100_phase4.yaml --algorithm grid --budget 16 --lr 0.001 --wd 0.0005 --batch 128 --seeds 0 1 2 3 4 5 6 --optimizer AdamW --epochs 30 --root experiments 2>&1 | Tee-Object -FilePath $logFile -Append
"=== done at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
