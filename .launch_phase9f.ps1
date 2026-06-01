$ErrorActionPreference = 'Continue'
Set-Location C:\Users\evija\sacgeometry
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/phase9f_iso_tuned_n7_$ts.log"
"=== Phase-9f launch $ts: iso-tuned (bs=128, lr=3e-3, wd=5e-4) seeds 3-6 ===" | Tee-Object -FilePath $logFile
"=== Push baseline + 3 winners from n=3 to n=7 at the iso-tuned cell ===" | Tee-Object -FilePath $logFile -Append

$tags = @(
    @{tag="baseline_resnet20"; idea="00_baseline_resnet20"},
    @{tag="sg_only_phi_budget"; idea="09_phi_budget"},
    @{tag="pair_gm_pdw"; idea="91_pair_gm_pdw"},
    @{tag="slot_act_sine"; idea="92_slot_act_sine"}
)

foreach ($t in $tags) {
    "=== iso-tuned n=7 extension: $($t.tag) seeds 3,4,5,6 at lr=3e-3 wd=5e-4 bs=128 AdamW ===" | Tee-Object -FilePath $logFile -Append
    "Start: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
    & .\.venv\Scripts\python -u scripts\run_hillclimb.py --tag $t.tag --idea $t.idea --config configs\cifar100_phase4.yaml --algorithm grid --budget 4 --lr 0.003 --wd 0.0005 --batch 128 --seeds 3 4 5 6 --optimizer AdamW --epochs 30 --root experiments 2>&1 | Tee-Object -FilePath $logFile -Append
    "End: $(Get-Date -Format 'yyyyMMdd_HHmmss')" | Tee-Object -FilePath $logFile -Append
}
"=== Phase-9f done at $(Get-Date -Format 'yyyyMMdd_HHmmss') ===" | Tee-Object -FilePath $logFile -Append
