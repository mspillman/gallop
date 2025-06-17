@echo off
setlocal

REM =================================================================
REM  Gallop Setup Script
REM  This will create a Python virtual environment and install all
REM  necessary dependencies. Run this script once before first use.
REM =================================================================

echo Checking for Python...

set "MIN_MAJOR=3"
set "MIN_MINOR=11"

REM Check if python command exists at all
python --version >nul 2>&1

if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not found in your PATH.
    echo Please install Python %MIN_MAJOR%.%MIN_MINOR% or newer and ensure it's added to the system PATH.
    pause
    exit /b 1
)

echo Checking for Pip...
pip --version >nul 2>&1

if %errorlevel% neq 0 (
    echo ERROR: Pip is not installed or not found in your PATH.
    echo Please install Python %MIN_MAJOR%.%MIN_MINOR% or newer and ensure it's added to the system PATH.
    pause
    exit /b 1
)

echo Checking for uv...
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing it now...
    pip install uv
)

set VENV_DIR=.venv
echo Creating virtual environment in "%VENV_DIR%"...
if exist "%VENV_DIR%" (
    echo Virtual environment already exists. Skipping creation.
) else (
    uv venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

echo Activating virtual environment and installing packages...
call "%VENV_DIR%\Scripts\activate.bat"

echo Installing gallop and its dependencies. This might take a few minutes...
uv pip install git+https://github.com/mspillman/gallop

if %errorlevel% neq 0 (
    echo ERROR: Installation failed. Please check the errors above.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo  Setup Complete!
echo  You can now run gallop by double-clicking
echo  on the 'run-gallop.bat' file.
echo ======================================================
echo.
pause
