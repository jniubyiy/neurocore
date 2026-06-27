@echo off
setlocal enabledelayedexpansion

$env:RUSTFLAGS="-C link-args=/STACK:33554432
cargo build

endlocal
pause