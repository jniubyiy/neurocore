@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo NeuroCore Layer Shader Compiler
echo ============================================

where glslc >nul 2>nul
if errorlevel 1 (
    echo [ERROR] glslc not found in PATH.
    echo Please install Vulkan SDK or add glslc to PATH.
    pause
    exit /b 1
)

set "PASS=0"
set "FAIL=0"
set "FAILED_FILES="

for /r %%f in (*.comp) do (
    set "COMP=%%f"
    set "SPV=%%~dpnf.spv"

    echo Compiling: !COMP!
    glslc -fshader-stage=compute -o "!SPV!" "!COMP!"

    if errorlevel 1 (
        echo [ERROR] Failed: !COMP!
        set /a FAIL+=1
        set "FAILED_FILES=!FAILED_FILES! !COMP!"
    ) else (
        echo [OK] !SPV!
        set /a PASS+=1
    )
)

echo ============================================
echo   Results: !PASS! succeeded, !FAIL! failed
if defined FAILED_FILES (
    echo Failed shaders:
    for %%f in (!FAILED_FILES!) do echo   %%f
) else (
    echo All layer shaders compiled successfully!
)
echo ============================================

pause
exit /b 0