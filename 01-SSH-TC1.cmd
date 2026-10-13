@echo off
setlocal
title CCDS-TC1 SSH
where ssh.exe >nul 2>&1
if errorlevel 1 (
    echo Windows OpenSSH Client is required.
    pause
    exit /b 1
)
echo Connecting to CCDS-TC1. Off campus: connect NTU VPN first.
echo This is the Head Node. Submit training with SLURM sbatch.
ssh.exe CCDS-TC1
set "result=%errorlevel%"
if not "%result%"=="0" (
    echo Connection failed. Check NTU VPN and the CCDS-TC1 SSH configuration.
    pause
)
exit /b %result%
