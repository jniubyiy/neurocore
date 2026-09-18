@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo Compiling all shaders...
echo ============================================

:: Common shaders
if exist "common\*.comp" (
    echo --- Common ---
    for %%f in (common\*.comp) do (
        echo %%f
        glslc -fshader-stage=compute -o "%%~dpnf.spv" "%%f"
        if errorlevel 1 (
            echo [ERROR] Compilation failed for %%f
            set FAILED=1
        )
    )
) else (
    echo No common shaders found.
)

:: Layers shaders
if exist "layers\*.comp" (
    echo --- Layers ---
    for %%f in (layers\*.comp) do (
        echo %%f
        glslc -fshader-stage=compute -o "%%~dpnf.spv" "%%f"
        if errorlevel 1 (
            echo [ERROR] Compilation failed for %%f
            set FAILED=1
        )
    )
) else (
    echo No layers shaders found.
)

:: Loss shaders
if exist "loss\*.comp" (
    echo --- Loss ---
    for %%f in (loss\*.comp) do (
        echo %%f
        glslc -fshader-stage=compute -o "%%~dpnf.spv" "%%f"
        if errorlevel 1 (
            echo [ERROR] Compilation failed for %%f
            set FAILED=1
        )
    )
) else (
    echo No loss shaders found.
)

:: Optimizer shaders
if exist "optim\*.comp" (
    echo --- Optimizers ---
    for %%f in (optim\*.comp) do (
        echo %%f
        glslc -fshader-stage=compute -o "%%~dpnf.spv" "%%f"
        if errorlevel 1 (
            echo [ERROR] Compilation failed for %%f
            set FAILED=1
        )
    )
) else (
    echo No optimizer shaders found.
)

if defined FAILED (
    echo ============================================
    echo [FAIL] Some shaders failed to compile!
    echo Check errors above.
    echo ============================================
) else (
    echo ============================================
    echo All shaders compiled successfully!
    echo ============================================
)

pause
exit /b 0