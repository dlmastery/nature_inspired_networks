$ErrorActionPreference = 'Continue'
Start-Sleep -Seconds 18000
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 600
    Set-Location C:\Users\evija\sacgeometry
    git fetch origin 2>$null | Out-Null
    $st = git status --porcelain 2>$null
    if ([string]::IsNullOrWhiteSpace($st)) {
        $local = (git rev-parse HEAD 2>$null).Trim()
        $remote = (git rev-parse origin/main 2>$null).Trim()
        if ($local -eq $remote) { Write-Host "[loop2 $i] clean and synced"; continue }
    }
    for ($r = 0; $r -lt 5; $r++) {
        try {
            git add -A 2>$null | Out-Null
            git -c user.name="dlmastery" -c user.email="eranti@gmail.com" commit -m "Auto-checkpoint: n=7 extension (tick $i)" 2>$null | Out-Null
            $pushOut = git push origin main 2>&1
            if ($LASTEXITCODE -eq 0) { Write-Host "[loop2 $i.$r] committed + pushed"; break }
            git pull --rebase origin main 2>$null | Out-Null
        } catch { Write-Host "[loop2 $i.$r] retry: $_" }
        Start-Sleep -Seconds 5
    }
}
Write-Host "[loop2] done after 60 ticks (10 hours, started after 5h sleep)"
