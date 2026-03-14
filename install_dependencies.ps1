# RecBole 依赖安装脚本 (PowerShell)
# 使用方法: 右键 -> 使用 PowerShell 运行

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RecBole 依赖安装脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
Write-Host "[检查] Python 环境..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "  ✓ 已安装: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ 未找到 Python，请先安装 Python" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

# 检查 pip
Write-Host "[检查] pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version
    Write-Host "  ✓ 已安装: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ 未找到 pip" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

Write-Host ""
Write-Host "[1/5] 清理 pip 缓存..." -ForegroundColor Yellow
pip cache purge 2>$null
Write-Host "  ✓ 完成" -ForegroundColor Green

Write-Host ""
Write-Host "[2/5] 升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "  ✓ 完成" -ForegroundColor Green

Write-Host ""
Write-Host "[3/5] 安装核心依赖 (numpy, pandas, scipy)..." -ForegroundColor Yellow
pip install numpy pandas scipy --quiet
Write-Host "  ✓ 完成" -ForegroundColor Green

Write-Host ""
Write-Host "[4/5] 安装 PyTorch (CPU版本)..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
Write-Host "  ✓ 完成" -ForegroundColor Green

Write-Host ""
Write-Host "[5/5] 安装 RecBole..." -ForegroundColor Yellow
pip install recbole -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
Write-Host "  ✓ 完成" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "已安装的包:" -ForegroundColor Yellow
pip list | Select-String -Pattern "recbole|torch|pandas|numpy|scipy"
Write-Host ""
Write-Host "现在可以运行以下命令:" -ForegroundColor Yellow
Write-Host "  cd d:\code\learningpath\notebook\base" -ForegroundColor White
Write-Host "  python run_sequence_models.py" -ForegroundColor White
Write-Host ""
Read-Host "按回车键退出"
