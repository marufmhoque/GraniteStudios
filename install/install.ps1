# Granite Studios installer.
#
# Installs a self-contained copy of the app under %LOCALAPPDATA% with its own
# virtual environment, then puts correctly-quoted shortcuts on the Desktop and
# Start Menu.
#
# Why it copies instead of running in place: a checkout often lives in a synced
# folder (Google Drive, OneDrive, Dropbox). Those paths come and go, are slow to
# read, and break pip installs. A local copy makes launching deterministic.
#
# -SourceDir lets the one-file bootstrap (Install-GraniteStudios.bat) point this
# at a freshly downloaded copy instead of a sibling checkout. Defaults to the
# parent of this script, so running it from a clone still just works.

[CmdletBinding()]
param(
    [string] $SourceDir,
    [string] $LogPath
)

$ErrorActionPreference = 'Stop'

$AppName    = 'Granite Studios'
$InstallDir = Join-Path $env:LOCALAPPDATA 'Granite Studios'
$VenvDir    = Join-Path $InstallDir 'venv'
if (-not $SourceDir) { $SourceDir = Split-Path -Parent $PSScriptRoot }
$SourceDir  = (Resolve-Path -LiteralPath $SourceDir).Path

function Say   ($m) { Write-Host "  $m" }
function Ok    ($m) { Write-Host "  [ok] $m"   -ForegroundColor Green }
function Warn  ($m) { Write-Host "  [!]  $m"   -ForegroundColor Yellow }
function Fail  ($m) { Write-Host "  [x]  $m"   -ForegroundColor Red }

# --- Logging ------------------------------------------------------------ #
# Everything the installer prints, including pip's output and any error, is
# written to a log file. A console window that closes, scrolls past the
# error, or gets dismissed must not take the only copy of the diagnosis with
# it -- on failure the log path is shown and the file is opened.
if (-not $LogPath) {
    $LogPath = Join-Path $env:TEMP 'GraniteStudios-install.log'
}
$PipLog = Join-Path $env:TEMP 'GraniteStudios-pip.log'
$script:Transcribing = $false
try {
    # Append, so lines the bootstrap already wrote are not truncated away.
    Start-Transcript -Path $LogPath -Append | Out-Null
    $script:Transcribing = $true
} catch {
    # Logging is a convenience; never let it stop the install.
    Write-Host "  [!]  Could not open a log file at $LogPath" -ForegroundColor Yellow
}

function Stop-Logging {
    if ($script:Transcribing) {
        try { Stop-Transcript | Out-Null } catch { }
        $script:Transcribing = $false
    }
}

function Abort ($message) {
    # Keep the console short. An earlier version dumped 80 lines of pip log
    # here, which pushed the actual error off the top of the window -- the
    # exact problem the logging was added to solve. Detail goes to the file;
    # the screen gets the error, and gets it last.
    ""
    Fail $message
    ""
    Say  "A full log was saved to:"
    Write-Host "      $LogPath" -ForegroundColor Cyan
    if (Test-Path -LiteralPath $PipLog) {
        Say "Dependency detail:"
        Write-Host "      $PipLog" -ForegroundColor Cyan
    }
    Say  "Send that file along if you want help working out what happened."
    ""
    # Repeated so it is the last thing on screen, whatever scrolled past.
    Fail $message

    Stop-Logging
    # Append pip's log to the saved file (not the console) so one file holds
    # the whole story. Done after Stop-Transcript to avoid interleaving.
    if (Test-Path -LiteralPath $PipLog) {
        try {
            Add-Content -LiteralPath $LogPath -Value "`n--- pip log ---"
            Get-Content -LiteralPath $PipLog | Add-Content -LiteralPath $LogPath
        } catch { }
    }
    # Open the log so the error is in front of the user even if this window
    # disappears.
    try { Start-Process notepad.exe -ArgumentList $LogPath -ErrorAction SilentlyContinue } catch { }
    exit 1
}

# Catches any terminating error anywhere below, so an unexpected failure is
# reported with its location instead of a bare stack trace.
trap {
    ""
    Fail "Unexpected error: $($_.Exception.Message)"
    if ($_.InvocationInfo) { Say "At: $($_.InvocationInfo.PositionMessage.Trim())" }
    Abort 'The installer stopped before finishing.'
}

Write-Host ''
Write-Host "  $AppName installer" -ForegroundColor Cyan
Write-Host '  ---------------------------------------------'
Say "Source : $SourceDir"
Say "Target : $InstallDir"
Write-Host ''

# ---------------------------------------------------------------- Python ----
# Needs 3.10+. Three traps this has to survive:
#   1. "python.exe" on PATH is often the Microsoft Store stub, which prints a
#      nag to stderr and installs nothing.
#   2. Windows PowerShell 5.1 turns a native command's stderr into a thrown
#      error when $ErrorActionPreference is 'Stop', so probing must relax it.
#   3. Passing a -c snippet containing double quotes or % through PowerShell
#      mangles it into a Python SyntaxError, hence the quote-free probe below.
$ProbeSnippet = 'import sys;print(sys.version_info.major,sys.version_info.minor,sys.executable)'

function Get-PythonInfo($exe, $prefixArgs) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $exe @($prefixArgs + @('-c', $ProbeSnippet))
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
        $text = ("$out" -split "`n" | Where-Object { $_ -match '^\s*\d+\s+\d+\s+\S' } | Select-Object -First 1)
        if (-not $text) { return $null }
        $parts = "$text".Trim() -split '\s+', 3
        $maj = [int]$parts[0]; $min = [int]$parts[1]; $real = $parts[2].Trim()
        if ($maj -ne 3 -or $min -lt 10) { return $null }
        if (-not (Test-Path -LiteralPath $real)) { return $null }
        return [pscustomobject]@{ Path = $real; Version = "$maj.$min" }
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Find-Python {
    # The py launcher's default is the best single guess.
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $info = Get-PythonInfo $launcher.Source @()
        if ($info) { return $info }
        # Then whatever runtimes it knows about, newest first.
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        try { $listed = & $launcher.Source -0p } catch { $listed = @() }
        $ErrorActionPreference = $prev
        foreach ($line in @($listed)) {
            $m = [regex]::Match("$line", '([A-Za-z]:\[^

]*?python\.exe)')
            if ($m.Success) {
                $info = Get-PythonInfo $m.Groups[1].Value @()
                if ($info) { return $info }
            }
        }
    }

    # Interpreters on PATH (python.exe here may be the Store stub; the probe
    # rejects it because the stub exits non-zero and prints nothing usable).
    foreach ($name in @('python3.exe', 'python.exe')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $info = Get-PythonInfo $cmd.Source @()
            if ($info) { return $info }
        }
    }

    # Registered installs, then the usual manual locations.
    $roots = @()
    foreach ($hive in @('HKCU:\Software\Python', 'HKLM:\Software\Python')) {
        if (Test-Path $hive) {
            Get-ChildItem $hive -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like '*InstallPath*' } |
                ForEach-Object {
                    $v = (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).'(default)'
                    if ($v) { $roots += $v }
                }
        }
    }
    $roots += @("$env:LOCALAPPDATA\Programs\Python", "$env:LOCALAPPDATA\Python", 'C:\Python313', 'C:\Python312', 'C:\Python311')
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $exes = @()
        $direct = Join-Path $root 'python.exe'
        if (Test-Path -LiteralPath $direct) { $exes += $direct }
        $exes += (Get-ChildItem -LiteralPath $root -Filter 'python.exe' -Recurse -Depth 2 -ErrorAction SilentlyContinue |
                  Sort-Object FullName -Descending | ForEach-Object { $_.FullName })
        foreach ($exe in $exes) {
            $info = Get-PythonInfo $exe @()
            if ($info) { return $info }
        }
    }
    return $null
}

Say 'Looking for Python 3.10 or newer...'
$python = Find-Python
if (-not $python) {
    Write-Host ''
    Fail 'No suitable Python found.'
    Write-Host ''
    Write-Host '  Granite Studios needs Python 3.10 or newer.'
    Write-Host '  Install it once, then run this installer again:'
    Write-Host ''
    Write-Host '      https://www.python.org/downloads/' -ForegroundColor Cyan
    Write-Host ''
    Write-Host '  On the first screen of the Python installer, tick'
    Write-Host '  "Add python.exe to PATH" before clicking Install.'
    Abort 'Python 3.10 or newer is required.'
}
Ok "Python $($python.Version) at $($python.Path)"

# ----------------------------------------------------------------- Copy -----
$required = @('main.py','requirements.txt','src')
foreach ($item in $required) {
    if (-not (Test-Path (Join-Path $SourceDir $item))) {
        Abort "Source looks incomplete - '$item' is missing from $SourceDir"
    }
}

# A running copy holds its files open, so replacing them fails with a
# confusing "being used by another process" error. Check first and say so.
if (Test-Path $InstallDir) {
    $running = @()
    foreach ($name in @('pythonw', 'python')) {
        foreach ($proc in (Get-Process $name -ErrorAction SilentlyContinue)) {
            $exePath = $null
            try { $exePath = $proc.Path } catch { }
            if ($exePath -and $exePath.StartsWith($InstallDir, 'OrdinalIgnoreCase')) {
                $running += $proc
            }
        }
    }
    if ($running.Count -gt 0) {
        Write-Host ''
        Say "Granite Studios is currently running (process id $($running[0].Id))."
        Say 'Close the Granite Studios window, then run this installer again.'
        Abort 'Cannot replace files while the app is open.'
    }
}

if (Test-Path $InstallDir) {
    Say 'Existing install found - replacing app files (your photos are untouched).'
    foreach ($leaf in @('src','main.py','requirements.txt')) {
        $p = Join-Path $InstallDir $leaf
        if (Test-Path $p) { Remove-Item $p -Recurse -Force }
    }
} else {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

Say 'Copying application files...'
# Everything the app needs at runtime; deliberately excludes web/, .git,
# build artefacts and caches.
$copyItems = @('main.py','requirements.txt','src','icon.png','icon.ico','README.md','README_INSTALL.md','LICENSE')
foreach ($leaf in $copyItems) {
    $src = Join-Path $SourceDir $leaf
    if (Test-Path $src) {
        Copy-Item $src -Destination $InstallDir -Recurse -Force
    }
}
Get-ChildItem -Path $InstallDir -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Ok 'Application files copied.'

# ------------------------------------------------------------------ Venv ----
if (-not (Test-Path (Join-Path $VenvDir 'Scripts\python.exe'))) {
    Say 'Creating a private Python environment (this keeps your system Python clean)...'
    & $python.Path -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Abort 'Could not create the private Python environment.' }
    Ok 'Environment created.'
} else {
    Ok 'Reusing the existing environment.'
}

$VenvPy  = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPyW = Join-Path $VenvDir 'Scripts\pythonw.exe'

Say 'Installing dependencies - this takes a few minutes the first time.'
Say 'It downloads image/video libraries (OpenCV, rawpy, FFmpeg, the AI model runtime).'
Write-Host ''
# pip writes its own verbose log. Start-Transcript does not reliably capture
# output from a native command, and pip is by far the likeliest step to fail,
# so relying on the transcript alone would lose exactly the detail that
# matters. Abort folds this file into the main log.
& $VenvPy -m pip install --upgrade pip --quiet --disable-pip-version-check --log $PipLog
# 2.0 moved Windows to the DirectML build of ONNX Runtime. Both builds install
# into the same 'onnxruntime' folder, so an upgrade that left the old one in
# place would have two packages fighting over the same files. Removing it is
# a no-op on a fresh install.
& $VenvPy -m pip uninstall -y onnxruntime --quiet --disable-pip-version-check --log $PipLog 2>$null
& $VenvPy -m pip install -r (Join-Path $InstallDir 'requirements.txt') --disable-pip-version-check --log $PipLog
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Say 'Dependency installation failed. The pip output above says why; the'
    Say 'usual causes are a dropped internet connection or a package that'
    Say 'has no prebuilt wheel for this Python version.'
    Abort 'Could not install the required libraries.'
}
Write-Host ''
Ok 'Dependencies installed.'

# ------------------------------------------------------- Pre-fetch model ----
# The background-removal model (~170MB) otherwise downloads on first use and
# makes that first click look frozen. Failure here is not fatal.
Say 'Fetching the background-removal AI model...'
& $VenvPy -c "from rembg import new_session; new_session()" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Ok 'AI model ready.' } else { Warn 'Model will download on first use instead.' }

# Face detector and face mesh for blink and smile detection (~4MB, checksum-pinned). Optional.
Say 'Fetching the face models (for blink and smile detection, about 4MB)...'
$prevEA = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
Push-Location $InstallDir
& $VenvPy -c "import sys; sys.path.insert(0,'.'); from src.ai_tools.face_analysis import ensure_model; from src.ai_tools import face_mesh, model_store; sys.exit(0 if ensure_model() and model_store.ensure_model(face_mesh.MESH_SPEC) else 1)" 2>$null | Out-Null
$faceOk = ($LASTEXITCODE -eq 0)
Pop-Location
$ErrorActionPreference = $prevEA
if ($faceOk) { Ok 'Face model ready.' } else { Warn 'Blink detection will be unavailable until this model can be downloaded.' }

# ------------------------------------------------------------- Shortcuts ----
# Arguments are quoted here. Without the quotes, an install path containing a
# space is split by the shell and the app exits before showing a window.
function New-AppShortcut($LnkPath, $Target, $Arguments, $WorkDir, $Icon) {
    $sh  = New-Object -ComObject WScript.Shell
    $lnk = $sh.CreateShortcut($LnkPath)
    $lnk.TargetPath       = $Target
    $lnk.Arguments        = $Arguments
    $lnk.WorkingDirectory = $WorkDir
    $lnk.Description      = 'Local AI photo & video editing suite'
    if ($Icon) { $lnk.IconLocation = "$Icon,0" }
    $lnk.Save()
}

$mainPy   = Join-Path $InstallDir 'main.py'
$iconIco  = Join-Path $InstallDir 'icon.ico'
$iconPng  = Join-Path $InstallDir 'icon.png'
$icon     = if (Test-Path $iconIco) { $iconIco } elseif (Test-Path $iconPng) { $iconPng } else { $null }
$launcher = if (Test-Path $VenvPyW) { $VenvPyW } else { $VenvPy }

Say 'Creating Desktop and Start Menu shortcuts...'
$desktop   = Join-Path $env:USERPROFILE 'Desktop'
$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
foreach ($dir in @($desktop, $startMenu)) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    # Clear the old broken shortcut name from previous versions, if present.
    $stale = Join-Path $dir 'Granite_Studios.lnk'
    if (Test-Path $stale) { Remove-Item $stale -Force -ErrorAction SilentlyContinue }
    New-AppShortcut (Join-Path $dir "$AppName.lnk") $launcher "`"$mainPy`"" $InstallDir $icon
}
Ok 'Shortcuts created.'

# The app itself also offers to make a shortcut on first run; this marker
# tells it not to, since the installer has already done a better job.
Set-Content -Path (Join-Path $env:USERPROFILE '.granitestudios_shortcut_created') `
            -Value 'created by installer' -Encoding utf8

# ----------------------------------------------------------- Uninstaller ----
# Take the uninstaller from the source tree, not from beside this script --
# the bootstrap runs install.ps1 out of a downloaded copy.
$uninstallSrc = Join-Path $SourceDir 'install\uninstall.ps1'
if (-not (Test-Path -LiteralPath $uninstallSrc)) { $uninstallSrc = Join-Path $PSScriptRoot 'uninstall.ps1' }
Copy-Item $uninstallSrc -Destination $InstallDir -Force -ErrorAction SilentlyContinue
$uninstallBat = @"
@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1"
pause
"@
Set-Content -Path (Join-Path $InstallDir 'Uninstall Granite Studios.bat') -Value $uninstallBat -Encoding ascii

# Register in Settings > Apps so it uninstalls like any normal program.
try {
    $regKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\GraniteStudios'
    New-Item -Path $regKey -Force | Out-Null
    Set-ItemProperty $regKey DisplayName     $AppName
    Set-ItemProperty $regKey DisplayVersion  '1.0'
    Set-ItemProperty $regKey Publisher       'Granite Studios'
    Set-ItemProperty $regKey InstallLocation $InstallDir
    Set-ItemProperty $regKey UninstallString "`"$InstallDir\Uninstall Granite Studios.bat`""
    if ($icon) { Set-ItemProperty $regKey DisplayIcon $icon }
    Ok 'Registered in Settings > Apps.'
} catch {
    Warn 'Could not register in Settings > Apps (harmless).'
}

# ------------------------------------------------------------- Smoke test ---
# This check must never be able to fail the install. Everything above has
# already succeeded by this point, so a problem here is information, not a
# reason to tell someone their working install is broken.
#
# Qt writes harmless warnings to stderr ("QFontDatabase: Cannot find font
# directory ..."), and under $ErrorActionPreference = 'Stop' PowerShell 5.1
# turns a native command's stderr into a *terminating* error. Capturing with
# `2>&1` therefore aborted the installer on a warning. Streams go to a file
# instead, with the preference relaxed for the duration.
Say 'Verifying the install can actually start...'
$StartupLog = Join-Path $env:TEMP 'GraniteStudios-startup-check.log'
$checkCode = @"
import os, sys
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, r'$InstallDir')
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from src.ui.main_window import MainWindow
w = MainWindow()
assert w.windowTitle(), 'no window title'
print('STARTUP_OK')
"@
$prevEA = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $VenvPy -c $checkCode *> $StartupLog
} catch {
    # Deliberately swallowed: see the note above.
    "startup check raised: $($_.Exception.Message)" | Out-File $StartupLog -Append
}
$ErrorActionPreference = $prevEA
$check = if (Test-Path -LiteralPath $StartupLog) { Get-Content -LiteralPath $StartupLog -Raw } else { '' }
if ("$check" -match 'STARTUP_OK') {
    Ok 'Verified - the app builds its window successfully.'
} else {
    Write-Host ''
    Warn 'The app did not print its startup marker. Details:'
    Write-Host "$check"
    Write-Host ''
    Say  'This check is informational - the install itself completed, and the'
    Say  'shortcuts were created. Try launching the app; if it does not open,'
    Say  "send along $LogPath."
}

Write-Host ''
Write-Host "  $AppName is installed." -ForegroundColor Green
Write-Host ''
Say "Launch it from the Desktop shortcut or the Start Menu."
Say "Installed to: $InstallDir"
Say "To remove it: Settings > Apps > $AppName (or run Uninstall Granite Studios.bat there)."
Say "Install log: $LogPath"
Write-Host ''
Stop-Logging
exit 0
