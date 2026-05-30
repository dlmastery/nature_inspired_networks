$ErrorActionPreference = 'Continue'
Set-Location C:\Users\evija\sacgeometry
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/phase9a_$ts.log"
"=== Phase-9a hill-climb launch $ts ===" | Tee-Object -FilePath $logFile
"=== 3 winners, coordinate-descent, budget=25, epochs=30, CIFAR-100 ===" | Tee-Object -FilePath $logFile -Append

$winners = @(
    @{tag="sg_only_phi_budget"; idea="09_phi_budget"},
    @{tag="pair_gm_pdw"; idea="91_pair_gm_pdw"},
    @{tag="slot_act_sine"; idea="92_slot_act_sine"}
)

foreach ($w in $winners) {
    "=== HILL-CLIMB: $($w.tag) -> ideas/$($w.idea) ===" | Tee-Object -FilePath $logFile -Append
    "Start: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
    & .\.venv\Scripts\python -u scripts\run_hillclimb.py `
        --tag $w.tag `
        --idea $w.idea `
        --config configs\cifar100_phase4.yaml `
        --algorithm coordinate `
        --budget 25 `
        --epochs 30 `
        --seeds 0 1 2 `
        --root experiments 2>&1 | Tee-Object -FilePath $logFile -Append
    "End: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
}

"=== Phase-9a all 3 hill-climbs done at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
