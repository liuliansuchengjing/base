# 修复 PyTorch DLL 问题的脚本

Write-Host "正在检查并修复 PyTorch 安装..."

# 1. 卸载当前 PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. 安装 CPU 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 验证安装
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"

Write-Host "PyTorch 安装完成！"
