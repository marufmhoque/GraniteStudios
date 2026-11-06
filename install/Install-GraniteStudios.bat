@echo off
REM =====================================================================
REM  Granite Studios - one-file installer
REM
REM  Download this single file anywhere and double-click it. It fetches the
REM  latest Granite Studios from GitHub and installs it. Nothing else to
REM  download, no repository to clone.
REM
REM  Re-run it any time to update an existing install to the latest version.
REM =====================================================================

setlocal
title Install Granite Studios

echo.
echo   Granite Studios
echo   ---------------------------------------------
echo   This downloads the latest version and installs it.
echo.

REM A .bat file is used only because it is the one thing Windows will run on
REM a double-click; PowerShell does the actual work. The payload is embedded
REM below rather than shipped as a second file, so this stays one download.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $src='%~f0'; $lines=Get-Content -LiteralPath $src; $i=($lines | Select-String -Pattern '^:PS_PAYLOAD_BEGIN' -List).LineNumber; if (-not $i) { Write-Host '  [x] This file looks truncated - download it again.' -ForegroundColor Red; exit 1 }; $ps=($lines[$i..($lines.Count-1)]) -join [Environment]::NewLine; Invoke-Expression $ps"

set EXITCODE=%ERRORLEVEL%
if %EXITCODE% NEQ 0 (
  echo.
  echo   Install did not finish. The message above explains why.
)
echo.
pause
exit /b %EXITCODE%

:PS_PAYLOAD_BEGIN
# ---------------------------------------------------------------------
# PowerShell payload. Everything below this marker is read and executed
# by the line above; cmd.exe never reaches it because of the exit /b.
# ---------------------------------------------------------------------

$ErrorActionPreference = 'Stop'

$Repo    = 'marufmhoque/GraniteStudios'
$Branch  = 'main'
$AppName = 'Granite Studios'

function Say  ($m) { Write-Host "  $m" }
function Ok   ($m) { Write-Host "  [ok] $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "  [!]  $m" -ForegroundColor Yellow }
function Fail ($m) { Write-Host "  [x]  $m" -ForegroundColor Red }

# TLS 1.2 is not the default on older Windows PowerShell and GitHub requires
# it, so set it before any web call.
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.SecurityProtocolType]::Tls12 -bor [Net.ServicePointManager]::SecurityProtocol
} catch { }

$work = Join-Path $env:TEMP ("GraniteStudios-setup-" + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $work -Force | Out-Null

try {
    $zip = Join-Path $work 'source.zip'
    $url = "https://codeload.github.com/$Repo/zip/refs/heads/$Branch"

    Say "Downloading the latest version from GitHub..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 300
    } catch {
        Fail "Could not download Granite Studios."
        Say  "Check your internet connection, then run this file again."
        Say  "(If you are on a restricted network, the GitHub download may be blocked.)"
        Say  ""
        Say  "Details: $($_.Exception.Message)"
        exit 1
    }

    $size = (Get-Item $zip).Length
    if ($size -lt 50000) {
        Fail "The download looks incomplete ($size bytes). Try again."
        exit 1
    }
    Ok "Downloaded ($([math]::Round($size/1MB,1)) MB)."

    Say 'Extracting...'
    try {
        # Expand-Archive exists from PS 5.0; fall back for anything older.
        if (Get-Command Expand-Archive -ErrorAction SilentlyContinue) {
            Expand-Archive -LiteralPath $zip -DestinationPath $work -Force
        } else {
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            [IO.Compression.ZipFile]::ExtractToDirectory($zip, $work)
        }
    } catch {
        Fail "Could not extract the download: $($_.Exception.Message)"
        exit 1
    }

    # GitHub zips put everything under <repo>-<branch>/.
    $root = Get-ChildItem -LiteralPath $work -Directory |
            Where-Object { Test-Path (Join-Path $_.FullName 'install\install.ps1') } |
            Select-Object -First 1
    if (-not $root) {
        Fail 'The download did not contain the installer. The repository layout may have changed.'
        exit 1
    }
    Ok 'Extracted.'
    Write-Host ''

    # Hand off to the real installer, pointed at what we just downloaded.
    $installer = Join-Path $root.FullName 'install\install.ps1'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer -SourceDir $root.FullName
    $code = $LASTEXITCODE
    if ($code -ne 0) { exit $code }
}
finally {
    # Always clean up the temp download, even on failure.
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

exit 0
