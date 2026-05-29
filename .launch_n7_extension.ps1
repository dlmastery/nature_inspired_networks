$ErrorActionPreference = 'Continue'
Set-Location C:\Users\evija\sacgeometry
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:PYTORCH_ENABLE_MPS_FALLBACK = "1"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/n7_extension_$ts.log"
"=== Launching n=7 extension at $ts ===" | Tee-Object -FilePath $logFile
"=== Tags: baseline_resnet20 sg_only_phi_budget pair_gm_pdw slot_act_sine ===" | Tee-Object -FilePath $logFile -Append
"=== Seeds: 3 4 5 6 (extending existing 0 1 2 to reach n=7) ===" | Tee-Object -FilePath $logFile -Append
"=== Expected: 16 runs x ~47 min ~= 12.5 GPU hours ===" | Tee-Object -FilePath $logFile -Append
& .\.venv\Scripts\python -u scripts\run_sweep.py `
    --config configs\cifar100_phase4.yaml `
    --seeds 3 4 5 6 `
    --skip-existing `
    --only baseline_resnet20 sg_only_phi_budget pair_gm_pdw slot_act_sine `
    --root experiments 2>&1 | Tee-Object -FilePath $logFile -Append
"=== n=7 extension done at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
