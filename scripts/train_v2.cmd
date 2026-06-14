@echo off
setlocal

set "REPO_DIR=%~dp0.."
set "PARENT_DIR=%REPO_DIR%\.."
set "LOG_DIR=%REPO_DIR%\outputs\logs"
set "OUT_LOG=%LOG_DIR%\v2_train.out.log"
set "ERR_LOG=%LOG_DIR%\v2_train.err.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%PARENT_DIR%"
conda run --no-capture-output -n alphazero-gomoku python -m alphazero_gomoku.train --preset v2 %* > "%OUT_LOG%" 2> "%ERR_LOG%"
