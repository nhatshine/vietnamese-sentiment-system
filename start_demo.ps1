param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$runtimePath = Join-Path $projectPath 'backend\outputs\epoch21_deployment\runtime'
New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
$preferredPython = Join-Path $env:USERPROFILE 'miniconda3\python.exe'
$pythonExe = if (Test-Path -LiteralPath $preferredPython) { $preferredPython } else { (Get-Command python -ErrorAction Stop).Source }
$nodeExe = (Get-Command node -ErrorAction Stop).Source
$viteScript = Join-Path $projectPath 'frontend\node_modules\vite\bin\vite.js'
if (-not (Test-Path -LiteralPath $viteScript)) { throw 'Thieu thu vien giao dien. Chay npm ci trong thu muc frontend truoc.' }
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

function Get-DemoModelInfo {
    try { return Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/model-info' -TimeoutSec 2 } catch { return $null }
}
$backendProcess = $null
$frontendProcess = $null
$deploymentInfo = Get-DemoModelInfo
if ($null -eq $deploymentInfo) {
    $backendProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-X','utf8','-m','uvicorn','backend.src.api:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $projectPath -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimePath 'backend.log') -RedirectStandardError (Join-Path $runtimePath 'backend-error.log')
    $waitTimer = [Diagnostics.Stopwatch]::StartNew()
    do {
        Start-Sleep -Milliseconds 250
        $deploymentInfo = Get-DemoModelInfo
        if ($backendProcess.HasExited) { throw ('Backend dung. Xem log: ' + (Join-Path $runtimePath 'backend-error.log')) }
    } until ($null -ne $deploymentInfo -or $waitTimer.Elapsed.TotalSeconds -gt 60)
}
if ($null -eq $deploymentInfo) { throw 'Backend chua san sang sau 60 giay.' }
if ($deploymentInfo.selected_alpha -ne 0.7 -or $deploymentInfo.models.transformer.name -notmatch 'epoch 21') {
    throw 'Cong 8000 dang chay phien ban khac. Dung backend cu roi chay lai.'
}
$frontendReady = $false
try { $frontendReady = (Invoke-WebRequest -Uri 'http://localhost:5173' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch {}
if (-not $frontendReady) {
    $frontendProcess = Start-Process -FilePath $nodeExe -ArgumentList @(('"' + $viteScript + '"'),'--host','localhost','--port','5173','--strictPort') -WorkingDirectory (Join-Path $projectPath 'frontend') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimePath 'frontend.log') -RedirectStandardError (Join-Path $runtimePath 'frontend-error.log')
    $waitTimer = [Diagnostics.Stopwatch]::StartNew()
    do {
        Start-Sleep -Milliseconds 250
        try { $frontendReady = (Invoke-WebRequest -Uri 'http://localhost:5173' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch {}
        if ($frontendProcess.HasExited) { throw ('Giao dien dung. Xem log: ' + (Join-Path $runtimePath 'frontend-error.log')) }
    } until ($frontendReady -or $waitTimer.Elapsed.TotalSeconds -gt 30)
}
if (-not $frontendReady) { throw 'Giao dien chua san sang.' }
[ordered]@{
    backend_pid = if ($backendProcess) { $backendProcess.Id } else { $null }
    frontend_pid = if ($frontendProcess) { $frontendProcess.Id } else { $null }
    project = $projectPath
    frontend_url = 'http://localhost:5173'
    api_url = 'http://127.0.0.1:8000'
    checkpoint_epoch = 21
    selected_alpha = 0.7
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimePath 'session.json') -Encoding UTF8
Write-Output 'Demo ready: Transformer epoch 21; SVM 70% + Transformer 30%'
Write-Output 'Open http://localhost:5173'
if (-not $NoBrowser) { Start-Process 'http://localhost:5173' }
