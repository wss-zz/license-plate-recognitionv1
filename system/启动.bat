@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=%~dp0
echo ============================================
echo   车牌识别一体闸机系统
echo   默认地址: http://127.0.0.1:8088
echo   关闭本窗口即停止服务
echo ============================================
python server.py --port 8088
pause
