@echo off
setlocal

set "REPO_DIR=%~dp0.."
set "PARENT_DIR=%REPO_DIR%\.."
set "LOG_DIR=%REPO_DIR%\outputs\logs"
set "OUT_LOG=%LOG_DIR%\distill_oldbest_light.out.log"
set "ERR_LOG=%LOG_DIR%\distill_oldbest_light.err.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PARENT_DIR%"
conda run --no-capture-output -n alphazero-gomoku python "%REPO_DIR%\scripts\distill_old_best.py" %* > "%OUT_LOG%" 2> "%ERR_LOG%"
