@echo off
setlocal EnableExtensions DisableDelayedExpansion

if /i "%~1"=="--install" goto installer

set "ACE_EXE="
set "ACE_UNINSTALL=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall"
for %%R in ("ACE Offline" "A.C.E.") do (
    if not defined ACE_EXE (
        for /f "tokens=2,*" %%A in ('%SystemRoot%\System32\reg.exe query "%ACE_UNINSTALL%\%%~R" /v InstallLocation 2^>nul') do (
            if /i "%%A"=="REG_SZ" (
                if exist "%%~B\openjarvis-desktop.exe" if exist "%%~B\ace-backend\python.exe" set "ACE_EXE=%%~B\openjarvis-desktop.exe"
            )
        )
    )
)

for %%D in ("%LOCALAPPDATA%\A.C.E" "%LOCALAPPDATA%\ACE Offline") do (
    if not defined ACE_EXE if exist "%%~D\openjarvis-desktop.exe" if exist "%%~D\ace-backend\python.exe" set "ACE_EXE=%%~D\openjarvis-desktop.exe"
)

if defined ACE_EXE (
    if /i "%~1"=="--check" (
        echo Instalado: "%ACE_EXE%"
        exit /b 0
    )
    start "" "%ACE_EXE%"
    exit /b
)

:installer
set "ACE_SETUP="
for %%I in ("Instalar A.C.E.exe" "Instalar-A.C.E.exe" "Instalar ACE Offline.exe") do (
    if not defined ACE_SETUP if exist "%~dp0%%~I" set "ACE_SETUP=%~dp0%%~I"
)
if not defined ACE_SETUP (
    for %%I in ("%~dp0A.C.E._*_x64-setup.exe") do (
        if not defined ACE_SETUP if exist "%%~fI" set "ACE_SETUP=%%~fI"
    )
)
if defined ACE_SETUP (
    if /i "%~1"=="--check" (
        echo Instalador: "%ACE_SETUP%"
        exit /b 0
    )
    if /i "%~2"=="--check" (
        echo Instalador: "%ACE_SETUP%"
        exit /b 0
    )
    start "" "%ACE_SETUP%"
    exit /b
)

echo O instalador do A.C.E. ainda nao foi encontrado.
echo Coloque "Instalar-A.C.E.exe" ao lado deste arquivo e abra-o novamente.
if /i "%~1"=="--check" exit /b 1
if /i "%~2"=="--check" exit /b 1
pause
exit /b 1
