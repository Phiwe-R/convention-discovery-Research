@echo off
cd /d "%~dp0"
python -m scripts.train_goal_only_batch --experiments experiment1 experiment2 --seeds 20 21 22 --budgets 100000 500000 --guide-strength 5 --device cpu
if errorlevel 1 (
    echo A3 training failed. Review console logs.
    exit /b 1
)
echo A3 training completed. TensorBoard: tensorboard --logdir runs\goal_only
