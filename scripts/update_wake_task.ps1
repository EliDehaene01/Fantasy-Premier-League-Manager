# One-time manual step (run as Administrator) to point the existing
# "FPL pipeline wake" Windows Scheduled Task (WakeToRun, daily) at
# scripts/weekly_pipeline.py instead of its old action, which only
# started Docker Desktop and did nothing else. Needs elevation -
# Set-ScheduledTask fails with "Access is denied" otherwise, which is why
# this wasn't applied automatically.
#
# Usage: right-click this file -> Run with PowerShell (as Administrator),
# or from an elevated PowerShell: .\scripts\update_wake_task.ps1

$repoRoot = "C:\Users\elide\Documents\GitHub\Fantasy-Premier-League-Manager"

$psCommand = @"
docker ps *> `$null
if (`$LASTEXITCODE -ne 0) { Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" }
`$deadline = (Get-Date).AddMinutes(3)
while ((Get-Date) -lt `$deadline) {
    docker ps *> `$null
    if (`$LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 5
}
& "$repoRoot\venv\Scripts\python.exe" "$repoRoot\scripts\weekly_pipeline.py" *>> "$repoRoot\weekly_pipeline.log"
"@

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -Command `"$psCommand`""
Set-ScheduledTask -TaskName "FPL pipeline wake" -Action $action

Write-Host "Updated. New action:"
(Get-ScheduledTask -TaskName "FPL pipeline wake").Actions | Format-List
