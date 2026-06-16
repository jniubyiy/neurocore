@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Запуск всех примеров neurocore
echo ============================================
echo.

set "EXAMPLES_DIR=examples"
set "PASSED=0"
set "FAILED=0"
set "FAILED_LIST="

if not exist "%EXAMPLES_DIR%\*.rs" (
    echo Папка examples не найдена или в ней нет файлов .rs
    exit /b 1
)

for %%f in ("%EXAMPLES_DIR%\*.rs") do (
    set "EXAMPLE=%%~nf"
    echo.
    echo --- Запуск примера: !EXAMPLE! ---
    cargo run --example "!EXAMPLE!" 2>&1
    if errorlevel 1 (
        echo [FAIL] !EXAMPLE!
        set /a FAILED+=1
        set "FAILED_LIST=!FAILED_LIST! !EXAMPLE!"
    ) else (
        echo [PASS] !EXAMPLE!
        set /a PASSED+=1
    )
    echo.
)

echo ============================================
echo   Итоги:
echo   Успешно: !PASSED!
echo   Ошибок:  !FAILED!
if not "!FAILED_LIST!"=="" (
    echo   Провалены:!FAILED_LIST!
)
echo ============================================
endlocal
pause