# Granite Studios installer.
#
# Installs a self-contained copy of the app under %LOCALAPPDATA% with its own
# virtual environment, then puts correctly-quoted shortcuts on the Desktop and
# Start Menu.
#
# Why it copies instead of running in place: a checkout often lives in a synced
# folder (Google Drive, OneDrive, Dropbox). Those paths come and go, are slow to
# read, and break pip installs. A local copy makes launching deterministic.

$ErrorActionPreference = 'Stop'

$AppName    = 'Granite Studios'
$InstallDir = Join-Path $env:LOCALAPPDATA 'Granite Studios'
$VenvDir    = Join-Path $InstallDir 'venv'
$SourceDir  = Split-Path -Parent $PSScriptRoot

function Say   ($m) { Write-Host "  $m" }
function Ok    ($m) { Write-Host "  [ok] $m"   -ForegroundColor Green }
function Warn  ($m) { Write-Host "  [!]  $m"   -ForegroundColor Yellow }
function Fail  ($m) { Write-Host "  [x]  $m"   -ForegroundColor Red }

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
    Write-Host ''
    exit 1
}
Ok "Python $($python.Version) at $($python.Path)"

# ----------------------------------------------------------------- Copy -----
$required = @('main.py','requirements.txt','src')
foreach ($item in $required) {
    if (-not (Test-Path (Join-Path $SourceDir $item))) {
        Fail "Source looks incomplete - '$item' is missing from $SourceDir"
        exit 1
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
    if ($LASTEXITCODE -ne 0) { Fail 'Could not create the virtual environment.'; exit 1 }
    Ok 'Environment created.'
} else {
    Ok 'Reusing the existing environment.'
}

$VenvPy  = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPyW = Join-Path $VenvDir 'Scripts\pythonw.exe'

Say 'Installing dependencies - this takes a few minutes the first time.'
Say 'It downloads image/video libraries (OpenCV, rawpy, FFmpeg, the AI model runtime).'
Write-Host ''
& $VenvPy -m pip install --upgrade pip --quiet --disable-pip-version-check
& $VenvPy -m pip install -r (Join-Path $InstallDir 'requirements.txt') --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Fail 'Dependency installation failed - see the pip output above.'
    Say 'The most common cause is no internet connection.'
    exit 1
}
Write-Host ''
Ok 'Dependencies installed.'

# ------------------------------------------------------- Pre-fetch model ----
# The background-removal model (~170MB) otherwise downloads on first use and
# makes that first click look frozen. Failure here is not fatal.
Say 'Fetching the background-removal AI model...'
& $VenvPy -c "from rembg import new_session; new_session()" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { Ok 'AI model ready.' } else { Warn 'Model will download on first use instead.' }

# Face detector for blink detection (~230KB, checksum-pinned). Also optional.
Say 'Fetching the face-detection model (for blink detection)...'
$prevEA = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
Push-Location $InstallDir
& $VenvPy -c "import sys; sys.path.insert(0,'.'); from src.ai_tools.face_analysis import ensure_model; sys.exit(0 if ensure_model() else 1)" 2>$null | Out-Null
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
Copy-Item (Join-Path $PSScriptRoot 'uninstall.ps1') -Destination $InstallDir -Force -ErrorAction SilentlyContinue
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
Say 'Verifying the install can actually start...'
$check = & $VenvPy -c @"
import os, sys
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, r'$InstallDir')
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from src.ui.main_window import MainWindow
w = MainWindow()
assert w.windowTitle(), 'no window title'
print('STARTUP_OK')
"@ 2>&1
if ("$check" -match 'STARTUP_OK') {
    Ok 'Verified - the app builds its window successfully.'
} else {
    Write-Host ''
    Warn 'The app did not pass its startup check. Details:'
    Write-Host "$check"
    Write-Host ''
    Warn 'The shortcuts were still created; try launching and report what you see.'
}

Write-Host ''
Write-Host "  $AppName is installed." -ForegroundColor Green
Write-Host ''
Say "Launch it from the Desktop shortcut or the Start Menu."
Say "Installed to: $InstallDir"
Say "To remove it: Settings > Apps > $AppName (or run Uninstall Granite Studios.bat there)."
Write-Host ''
exit 0
