@echo off
setlocal enabledelayedexpansion

$env:RUSTFLAGS="-C link-args=/STACK:33554432
cargo clean && cargo build

endlocal
pause