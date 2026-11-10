
@echo off

cd /d "%~dp0"

python -m scripts.train_goal_guided_batch --experiments experiment1 experiment2 --seeds 20 21 22 --budgets 100000 500000 --guide-strength 5 --device cpu

if errorlevel 1 (
    echo Training stopped because of an error.
    exit /b 1
)

echo Training batch completed successfully.
echo PPO: tensorboard --logdir runs\goal_guided_clean
echo Beliefs: tensorboard --logdir runs\belief_pretraining
