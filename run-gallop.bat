@echo off
setlocal

REM =================================================================
REM  Gallop Application Launcher
REM  This script activates the virtual environment and starts the
REM  Gallop Streamlit user interface.
REM =================================================================

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Define the path to the virtual environment
set VENV_PATH=%SCRIPT_DIR%.venv

echo Looking for the virtual environment...

REM Check if the virtual environment exists. If not, give a helpful error.
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Please run the 'setup.bat' script first to install the application.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

echo.
echo Starting the Gallop UI...
echo (The web application will open in your browser automatically)
echo (To stop the application, close this command window or press Ctrl+C)
echo.

REM Run the console script entry point
gallop-ui

endlocal