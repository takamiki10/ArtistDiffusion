param([ValidateRange(1,10)][int]$Trial = 1)
$ErrorActionPreference = 'Stop'
$restartRoot = 'C:\RokaeSDK\nrt_v050\path_0003_sessions'
$restartMarker = Join-Path $restartRoot ('trial_{0:00}.json' -f $Trial)
if (-not (Test-Path -LiteralPath $restartMarker)) {
    Write-Output 'No reservation exists. Nothing to archive.'
    return
}
$restartReservation = Get-Content -LiteralPath $restartMarker -Raw | ConvertFrom-Json
$restartFolder = [IO.Path]::GetFullPath($restartReservation.folder)
if ($restartReservation.trial -ne $Trial -or [IO.Path]::GetDirectoryName($restartFolder) -ne $restartRoot) {
    throw 'Unexpected reservation or session location.'
}
$restartFinal = Get-Content -LiteralPath (Join-Path $restartFolder 'final_status.json') -Raw | ConvertFrom-Json
if ($restartFinal.status -ne 'FAILED_EXECUTION' -or $null -eq $restartFinal.attempts.moveStart -or $restartFinal.attempts.moveStart -ne 0 -or $restartFinal.moveStart_success -ne $false -or $null -eq $restartFinal.cleanup_errors -or @($restartFinal.cleanup_errors).Count -ne 0) {
    throw 'Refusing restart: requires a finalized failed attempt with zero START attempts and clean cleanup. A started trial needs separate operator review.'
}
$restartCalls = @(Get-Content -LiteralPath (Join-Path $restartFolder 'calls.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
if (@($restartCalls | Where-Object { $_.api -eq 'moveStart' }).Count -ne 0) {
    throw 'Refusing restart: moveStart call evidence exists.'
}
$restartDisconnect = @($restartCalls | Where-Object { $_.api -eq 'disconnectFromRobot' -and $_.phase -eq 'cleanup' })
if ($restartDisconnect.Count -ne 1 -or $restartDisconnect[0].error.ec -ne 0 -or $restartDisconnect[0].error.message -ne 'success') {
    throw 'Successful disconnect evidence required.'
}
$restartArchive = Join-Path $restartRoot ('trial_{0:00}_unstarted_{1}_reservation.json' -f $Trial, [guid]::NewGuid().ToString('N'))
Move-Item -LiteralPath $restartMarker -Destination $restartArchive
Write-Output ('Reservation archived: ' + $restartArchive)
Write-Output 'Session logs preserved. Launch the trial separately and use fresh authorizations.'
