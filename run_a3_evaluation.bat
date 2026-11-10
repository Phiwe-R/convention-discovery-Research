@echo off
cd /d "%~dp0"
python -m scripts.evaluate_goal_only_batch --experiments experiment1 experiment2 --seeds 20 21 22 --budgets 100000 500000 --modes real uniform --episodes 1000 --seed-start 10000
if errorlevel 1 (
    echo A3 evaluation failed. Check output above.
    exit /b 1
)
echo A3 evaluation completed. Check results\goal_only\evaluation\master_evaluation_results.csv
