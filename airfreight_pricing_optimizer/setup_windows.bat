@echo off
setlocal EnableDelayedExpansion
echo.
echo ============================================================
echo   Airfreight Pricing Optimizer -- Windows Setup
echo ============================================================
echo.

:: Check for Microsoft Store Python
for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set PYEXE=%%i
echo Python found at: %PYEXE%

echo %PYEXE% | findstr /i "WindowsApps" >nul
if %errorlevel% == 0 (
    echo.
    echo [ERROR] Microsoft Store Python detected!
    echo         This causes "Access is denied" errors with SQL Server.
    echo.
    echo  Steps to fix:
    echo  1. Open Windows Settings
    echo  2. Go to: Apps ^> Advanced App Settings ^> App Execution Aliases
    echo  3. Toggle OFF python.exe and python3.exe
    echo  4. Download Python from: https://www.python.org/downloads/
    echo     (check "Add Python to PATH" during install)
    echo  5. Re-run this script from a NEW terminal window
    echo.
    pause
    exit /b 1
)

echo [OK] Python is not the Microsoft Store version.
echo.

:: Create virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

:: Install dependencies
echo Installing Python dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt

:: Check ODBC Driver
python -c "import pyodbc; drivers=pyodbc.drivers(); exit(0 if any('SQL Server' in d for d in drivers) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] ODBC Driver 17 for SQL Server not detected.
    echo           Download from: https://aka.ms/downloadmsodbcsql
    echo           Install it then re-run this script.
    echo.
) else (
    echo [OK] ODBC Driver 17 for SQL Server found.
)

echo.
echo ============================================================
echo   Setup complete! Run the app with:
echo     .venv\Scripts\activate
echo     streamlit run app.py
echo ============================================================
echo.
pause
