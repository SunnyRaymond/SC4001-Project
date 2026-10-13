@echo off
setlocal
title CCDS-TC1 VS Code
where ssh.exe >nul 2>&1
if errorlevel 1 (
    echo Windows OpenSSH Client is required.
    pause
    exit /b 1
)
where code.cmd >nul 2>&1
if errorlevel 1 (
    echo Install VS Code and add it to PATH, then try again.
    pause
    exit /b 1
)
echo Checking passwordless access. Off campus: connect NTU VPN first.
call code.cmd --list-extensions | powershell.exe -NoProfile -Command "if (@($input) -notcontains 'ms-vscode-remote.remote-ssh') { exit 1 }"
if errorlevel 1 (
    echo Install the VS Code Remote - SSH extension, then try again.
    pause
    exit /b 1
)
ssh.exe -o ConnectTimeout=15 CCDS-TC1 "test -d /tc1home/UG/yguo017/SC4001-Assignment"
if errorlevel 1 (
    echo Cannot access remote folder. Check NTU VPN, SSH config and remote folder.
    pause
    exit /b 1
)
echo Opening remote assignment folder. Submit training with SLURM sbatch.
call code.cmd --new-window --remote ssh-remote+CCDS-TC1 /tc1home/UG/yguo017/SC4001-Assignment
set "result=%errorlevel%"
if not "%result%"=="0" (
    echo VS Code could not start. Check VS Code and its Remote - SSH extension.
    pause
)
exit /b %result%
