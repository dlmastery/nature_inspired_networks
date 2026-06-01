$ErrorActionPreference = 'Continue'
Set-Location C:\Users\evija\sacgeometry
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/phase9e_wave1_$ts.log"
"=== Phase-9e Wave-1 launch $ts ===" | Tee-Object -FilePath $logFile
"=== G9 combo winners: H87 + H88 (CIFAR-100 30-ep, 3 seeds) + H91 (rotated_cifar100 30-ep, 3 seeds) ===" | Tee-Object -FilePath $logFile -Append

# Gate on Phase-9f completion. The launcher checks for the
# 'Phase-9f done' sentinel in any of the recent phase9f logs.
"=== Waiting for Phase-9f completion sentinel ===" | Tee-Object -FilePath $logFile -Append
while ($true) {
    $latestLog = Get-ChildItem logs/phase9f_*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        $tail = Get-Content $latestLog.FullName -Tail 5 -ErrorAction SilentlyContinue
        if ($tail -match "Phase-9f done") {
            "=== Phase-9f sentinel seen in $($latestLog.Name) at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
            break
        }
    }
    Start-Sleep -Seconds 900
}

# Wave-1 sub-launches. H87 + H88 share configs/cifar100_phase4.yaml (3 seeds, 30 ep).
# H88 is currently SKIPPED until two additive runner patches land:
#   (1) phi_budget + NaturePrior-flag-overlay path for toroidal, and
#   (2) TrainConfig.betti_loss_weight + Trainer callback for Betti aux loss.
# Per the Phase-9e brief failure mode: "document the gap and SKIP".
# If those patches land before Wave-1 launches, uncomment the H88 line.

"=== Wave-1 sub-launch 1/3: combo_n4_pair_slot CIFAR-100 seeds 0 1 2 ===" | Tee-Object -FilePath $logFile -Append
"Start: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
& .\.venv\Scripts\python -u scripts\run_sweep.py --config configs\cifar100_phase4.yaml --seeds 0 1 2 --only combo_n4_pair_slot --root experiments --skip-existing 2>&1 | Tee-Object -FilePath $logFile -Append
"End: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append

# H88 — SKIPPED pending wiring patches. Uncomment when ready:
# "=== Wave-1 sub-launch 2/3: combo_novelty_betti_torus CIFAR-100 seeds 0 1 2 ===" | Tee-Object -FilePath $logFile -Append
# "Start: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
# & .\.venv\Scripts\python -u scripts\run_sweep.py --config configs\cifar100_phase4.yaml --seeds 0 1 2 --only combo_novelty_betti_torus --root experiments --skip-existing 2>&1 | Tee-Object -FilePath $logFile -Append
# "End: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
"=== Wave-1 sub-launch 2/3: combo_novelty_betti_torus — SKIPPED pending wiring (toroidal-flag-on-phi_budget + Betti-loss-trainer-callback) ===" | Tee-Object -FilePath $logFile -Append

# H91 — rotated_cifar100 + vit_tiny_icosa. The row's overrides set
# dataset=rotated_cifar100 in cfg, so run_sweep.main's
# cfg.update(row['overrides']) routes through load_dataset's
# rotated_cifar branch. Output lands under experiments/rotated_cifar100/.
"=== Wave-1 sub-launch 3/3: combo_domain_icosa_rotation rotated_cifar100 seeds 0 1 2 ===" | Tee-Object -FilePath $logFile -Append
"Start: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
& .\.venv\Scripts\python -u scripts\run_sweep.py --config configs\cifar100_phase4.yaml --seeds 0 1 2 --only combo_domain_icosa_rotation --root experiments --skip-existing 2>&1 | Tee-Object -FilePath $logFile -Append
"End: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append

"=== Phase-9e Wave-1 done at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
