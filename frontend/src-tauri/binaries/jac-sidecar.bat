@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
  set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"
  "%REPO_ROOT%\.venv\Scripts\python.exe" -m jac_client.plugin.src.targets.desktop.sidecar.main %*
  exit /b %ERRORLEVEL%
)

set "REPO_ROOT=%SCRIPT_DIR%.."
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
  set "PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%"
  "%REPO_ROOT%\.venv\Scripts\python.exe" -m jac_client.plugin.src.targets.desktop.sidecar.main %*
  exit /b %ERRORLEVEL%
)

if /I "%KFORGE_ALLOW_SYSTEM_PYTHON%"=="1" (
  python -m jac_client.plugin.src.targets.desktop.sidecar.main %*
  exit /b %ERRORLEVEL%
)

echo Kernel Forge desktop could not find a bundled or repo .venv Python runtime. 1>&2
echo Set KFORGE_ALLOW_SYSTEM_PYTHON=1 only for local debugging if you need to bypass the packaged runtime. 1>&2
exit /b 1
