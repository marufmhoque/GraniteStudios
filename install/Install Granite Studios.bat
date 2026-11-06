@echo off
REM ============================================================
REM  Granite Studios - installer
REM  Double-click this file to install Granite Studios.
REM ============================================================
REM This wrapper exists because .ps1 files are not double-clickable
REM (Windows opens them in an editor). It hands straight off to
REM install.ps1, which does the real work.

setlocal
cd /d "%~dp0"

echo.
echo  Installing Granite Studios...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
  echo.
  echo  Install did not finish. The message above explains why.
  echo.
)

echo.
pause
exit /b %EXITCODE%
