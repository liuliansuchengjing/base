@echo off
echo 正在运行 Assist09 数据集的推荐算法实验...
cd /d d:\code\learningpath\notebook\base
python -u run_traditional.py
echo.
echo 实验完成！结果保存在 results_assist09.pkl
pause
