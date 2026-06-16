@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python merge_rust_code_smart.py
pause