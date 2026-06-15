@echo off
setlocal

set "REPO_DIR=%~dp0.."
set "PARENT_DIR=%REPO_DIR%\.."
set "LOG_DIR=%REPO_DIR%\outputs\logs"
set "TMP_DIR=%REPO_DIR%\outputs\tmp"
set "OUT_LOG=%LOG_DIR%\v3_student_local_train.out.log"
set "ERR_LOG=%LOG_DIR%\v3_student_local_train.err.log"
set "PYTHON_EXE=%USERPROFILE%\.conda\envs\alphazero-gomoku\python.exe"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%TMP_DIR%" mkdir "%TMP_DIR%"
set "TEMP=%TMP_DIR%"
set "TMP=%TMP_DIR%"

cd /d "%PARENT_DIR%"
"%PYTHON_EXE%" -m alphazero_gomoku.train --preset v3-student-local %* > "%OUT_LOG%" 2> "%ERR_LOG%"
