$ErrorActionPreference = 'Continue'
Set-Location C:\Users\evija\sacgeometry
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/baseline_bs128_lr3em3_seeds_$ts.log"
"=== baseline @ bs=128 lr=3e-3 wd=5e-4 seeds 1,2 launch $ts ===" | Tee-Object -FilePath $logFile
"=== Closes the iso-tuned baseline comparison loop ===" | Tee-Object -FilePath $logFile -Append
& .\.venv\Scripts\python -u scripts\run_hillclimb.py --tag baseline_resnet20 --idea 00_baseline_resnet20 --config configs\cifar100_phase4.yaml --algorithm grid --budget 4 --lr 0.003 --wd 0.0005 --batch 128 --seeds 1 2 --optimizer AdamW --epochs 30 --root experiments 2>&1 | Tee-Object -FilePath $logFile -Append
"=== done at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
