@echo off
chcp 65001 >nul
title APX-240 诊断 Agent - 在线演示（本地服务 + 隧道）

echo ============================================
echo   APX-240 故障诊断 Agent 启动脚本
echo ============================================
echo.

REM 检查 ngrok
if not exist "%~dp0ngrok\ngrok.exe" (
    echo [错误] ngrok.exe 不存在，请确认 ngrok 目录完整
    pause
    exit /b 1
)

REM 演示前体检：确认 LLM / 飞书 是 live 还是自动降级
echo [1/3] 演示前体检...
python "%~dp0run.py" doctor
echo.

REM 启动 Web 服务（后台）。--llm 是全局开关，必须写在子命令之前
echo [2/3] 启动 Web 服务 (端口 8765)...
start /min python "%~dp0run.py" --llm web --no-browser --port 8765
timeout /t 3 /nobreak >nul

REM 启动 ngrok 隧道
echo [3/3] 启动 ngrok 隧道...
echo.
echo   公网访问地址：
echo   https://retrieval-stowing-cascade.ngrok-free.dev
echo.
echo   本地访问地址：
echo   http://127.0.0.1:8765/
echo.
echo   按 Ctrl+C 停止所有服务
echo ============================================
echo.

"%~dp0ngrok\ngrok.exe" http 8765 --url=retrieval-stowing-cascade.ngrok-free.dev
