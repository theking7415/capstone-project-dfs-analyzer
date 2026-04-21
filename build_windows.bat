@echo off
:: ============================================================
:: Build Random DFS Graph Analyzer  —  Windows .exe
:: ============================================================
:: Prerequisites:
::   pip install pyinstaller
::   pip install -r requirements_desktop.txt
::
:: Run this script from the project root:
::   build_windows.bat
:: ============================================================

echo.
echo ============================================================
echo  Random DFS Graph Analyzer  --  Windows build
echo ============================================================
echo.

:: Check PyInstaller
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller is not installed.
    echo Run:  pip install pyinstaller
    pause
    exit /b 1
)

:: Check PyQt6
python -c "import PyQt6" 2>nul
if errorlevel 1 (
    echo [ERROR] PyQt6 is not installed.
    echo Run:  pip install -r requirements_desktop.txt
    pause
    exit /b 1
)

echo [OK] Dependencies found
echo.
echo Building...  (this takes 1-3 minutes)
echo.

:: Clean previous build
if exist build\RandomDFSAnalyzer rmdir /s /q build\RandomDFSAnalyzer
if exist dist\RandomDFSAnalyzer  rmdir /s /q dist\RandomDFSAnalyzer

:: Run PyInstaller
python -m PyInstaller random_dfs_analyzer.spec --noconfirm

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Check output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Build complete!
echo  Executable folder:  dist\RandomDFSAnalyzer\
echo  Launch:             dist\RandomDFSAnalyzer\RandomDFSAnalyzer.exe
echo ============================================================
echo.
echo To distribute: zip the entire dist\RandomDFSAnalyzer\ folder.
echo The recipient just extracts and double-clicks the .exe  --
echo no Python or conda installation required.
echo.
pause
