@echo off
chcp 65001 >nul
echo ========================================
echo   依赖安装脚本
echo ========================================
echo.

echo [1/4] 清理 pip 缓存...
pip cache purge 2>nul

echo.
echo [2/4] 升级 pip...
python -m pip install --upgrade pip

echo.
echo [3/4] 安装核心依赖...
pip install numpy pandas scipy

echo.
echo [4/4] 安装 RecBole 和 PyTorch...
echo 正在安装 RecBole (这可能需要几分钟)...

REM 尝试使用清华镜像加速
pip install recbole -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 如果 recbole 安装失败，尝试安装 torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 已安装的包:
pip list | findstr -i "recbole torch pandas numpy scipy"
echo.
echo 现在可以运行:
echo   python run_sequence_models.py
echo.
pause
