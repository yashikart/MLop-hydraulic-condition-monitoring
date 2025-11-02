# Quick script to check MLflow runs
Write-Host "Checking MLflow experiments and runs..." -ForegroundColor Green

if (Test-Path ".\mlruns") {
    $experiments = Get-ChildItem ".\mlruns" -Directory | Where-Object { $_.Name -match '^\d+$' }
    
    Write-Host "`nFound experiments:" -ForegroundColor Yellow
    foreach ($exp in $experiments) {
        Write-Host "  Experiment ID: $($exp.Name)" -ForegroundColor Cyan
        $runs = Get-ChildItem $exp.FullName -Directory | Where-Object { $_.Name -match '^[a-f0-9]{32}$' }
        Write-Host "    Runs: $($runs.Count)" -ForegroundColor $(if ($runs.Count -gt 0) { "Green" } else { "Red" })
        
        if ($runs.Count -eq 0) {
            Write-Host "    ⚠️  No runs found. Train models in Streamlit app!" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "mlruns directory not found!" -ForegroundColor Red
}

Write-Host "`nTo train models:" -ForegroundColor Green
Write-Host "1. Open http://localhost:8501" -ForegroundColor White
Write-Host "2. Click 'Load Dataset'" -ForegroundColor White
Write-Host "3. Go to 'Model Training' page" -ForegroundColor White
Write-Host "4. Click 'Train All Models'" -ForegroundColor White

