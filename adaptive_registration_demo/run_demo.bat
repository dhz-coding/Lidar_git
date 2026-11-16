@echo off
setlocal

if "%~2"=="" (
  echo 用法: run_demo.bat 装沙前路径 装沙后路径 [输出目录]
  exit /b 1
)

set "OUTPUT_DIR=%~3"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=output"

python adaptive_registration_demo.py --before "%~1" --after "%~2" --output_dir "%OUTPUT_DIR%" %4 %5 %6
if errorlevel 1 exit /b %errorlevel%

echo.
echo Demo finished.
endlocal
