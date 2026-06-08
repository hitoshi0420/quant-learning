@echo off
chcp 65001 >nul
title 量化股票分析系统

echo.
echo ╔══════════════════════════════════════════════╗
echo ║       量化股票分析系统 - 启动中...          ║
echo ╚══════════════════════════════════════════════╝
echo.

:: ============================================
:: 1. 清理占用端口的旧进程
:: ============================================
echo [1/4] 正在清理旧进程...

for %%p in (8050 8051) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr "0.0.0.0:%%p.*LISTENING" 2^>nul') do (
        echo   终止端口 %%p 上的进程 PID=%%a
        taskkill /F /PID %%a >nul 2>&1
    )
)

timeout /t 2 /nobreak >nul
echo   清理完成
echo.

:: ============================================
:: 2. 启动仪表盘 (端口 8050)
:: ============================================
echo [2/4] 正在启动仪表盘 (端口 8050)...
start "量化仪表盘-8050" /D "%~dp0" python dashboard.py

:: ============================================
:: 3. 等待仪表盘就绪
:: ============================================
echo [3/4] 等待仪表盘加载数据...

set "n=0"
:wait_dash
timeout /t 3 /nobreak >nul
set /a n+=3
curl -s -o nul http://localhost:8050 2>nul
if %errorlevel%==0 goto dash_ready
if %n% lss 60 goto wait_dash
echo   警告: 仪表盘 30 秒内未就绪，请手动检查
goto start_report

:dash_ready
echo   仪表盘已就绪 (耗时 %n% 秒)
echo.

:: ============================================
:: 4. 启动单股分析 (端口 8051)
:: ============================================
:start_report
echo [4/4] 正在启动单股分析 (端口 8051)...
start "单股分析-8051" /D "%~dp0" python report.py

timeout /t 3 /nobreak >nul
echo.

echo ╔══════════════════════════════════════════════╗
echo ║            系统启动完成！                   ║
echo ╠══════════════════════════════════════════════╣
echo ║  仪表盘:     http://localhost:8050          ║
echo ║  单股分析:   http://localhost:8051          ║
echo ║  命令行浏览: python explore.py [代码]       ║
echo ╚══════════════════════════════════════════════╝
echo.

pause
