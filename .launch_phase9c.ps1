$ErrorActionPreference = 'Continue'
Set-Location C:\Users\evija\sacgeometry
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/phase9c_n7_hillclimbed_$ts.log"
"=== Phase-9c launch $ts: n=3 -> n=7 at hill-climbed best configs ===" | Tee-Object -FilePath $logFile
"=== Required to resolve phi_budget borderline CI at hill-climbed best ===" | Tee-Object -FilePath $logFile -Append

# Each winner ran cells at slightly different best configs - use the per-winner best
# phi_budget:  lr=3e-3 wd=5e-4 bs=128 AdamW
# pair_gm_pdw: lr=3e-3 wd=5e-4 bs=128 AdamW  
# slot_act_sine: lr=3e-3 wd=2e-3 bs=128 AdamW
# baseline:    lr=3e-3 wd=5e-4 bs=256 AdamW (different bs - keep separate)

# Use run_hillclimb.py with --algorithm grid + budget 4 + only the 4 new seeds at the existing best config
# Configs differ across winners so launch 4 separate jobs

$jobs = @(
    @{tag="sg_only_phi_budget"; idea="09_phi_budget"; lr=0.003; wd=0.0005; bs=128; opt="AdamW"},
    @{tag="pair_gm_pdw"; idea="91_pair_gm_pdw"; lr=0.003; wd=0.0005; bs=128; opt="AdamW"},
    @{tag="slot_act_sine"; idea="92_slot_act_sine"; lr=0.003; wd=0.002; bs=128; opt="AdamW"},
    @{tag="baseline_resnet20"; idea="00_baseline_resnet20"; lr=0.003; wd=0.0005; bs=256; opt="AdamW"}
)

foreach ($j in $jobs) {
    "=== n=7 EXTENSION: $($j.tag) seeds 3-6 at lr=$($j.lr) wd=$($j.wd) bs=$($j.bs) opt=$($j.opt) ===" | Tee-Object -FilePath $logFile -Append
    "Start: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
    & .\.venv\Scripts\python -u scripts\run_hillclimb.py `
        --tag $j.tag `
        --idea $j.idea `
        --config configs\cifar100_phase4.yaml `
        --algorithm grid `
        --budget 16 `
        --lr $j.lr `
        --wd $j.wd `
        --batch $j.bs `
        --seeds 3 4 5 6 `
        --optimizer $j.opt `
        --epochs 30 `
        --root experiments 2>&1 | Tee-Object -FilePath $logFile -Append
    "End: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
}
"=== Phase-9c all 4 n=7-extensions done at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
