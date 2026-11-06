# Granite Studios uninstaller.
#
# Removes the app, its private Python environment, and its shortcuts.
# Never touches your photos, videos, or any -edit folder.

$ErrorActionPreference = 'Continue'

$AppName    = 'Granite Studios'
$InstallDir = Join-Path $env:LOCALAPPDATA 'Granite Studios'

Write-Host ''
Write-Host "  Uninstall $AppName" -ForegroundColor Cyan
Write-Host '  ---------------------------------------------'
Write-Host "  This removes the app from: $InstallDir"
Write-Host '  Your photos, videos and -edit folders are NOT touched.'
Write-Host ''
$answer = Read-Host '  Remove it? (y/N)'
if ($answer -notmatch '^[Yy]') { Write-Host '  Cancelled.'; exit 0 }

Write-Host ''
foreach ($dir in @((Join-Path $env:USERPROFILE 'Desktop'),
                   (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'))) {
    foreach ($name in @("$AppName.lnk", 'Granite_Studios.lnk')) {
        $lnk = Join-Path $dir $name
        if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Host "  removed $name" }
    }
}

$marker = Join-Path $env:USERPROFILE '.granitestudios_shortcut_created'
if (Test-Path $marker) { Remove-Item $marker -Force }

Remove-Item 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\GraniteStudios' `
            -Recurse -Force -ErrorAction SilentlyContinue

# The uninstaller lives inside the folder it is deleting, so hand the final
# delete to a detached process that outlives this script.
if (Test-Path $InstallDir) {
    Write-Host '  removing application files...'
    Start-Process powershell.exe -ArgumentList @(
        '-NoProfile','-ExecutionPolicy','Bypass','-Command',
        "Start-Sleep -Seconds 2; Remove-Item -LiteralPath '$InstallDir' -Recurse -Force -ErrorAction SilentlyContinue"
    ) -WindowStyle Hidden
}

Write-Host ''
Write-Host "  $AppName has been removed." -ForegroundColor Green
Write-Host ''
