#!/bin/bash
# Run all 125M experiments sequentially inside tmux.
# Connection-safe: detaches after launch.
#
# Usage:  bash scripts/run_125m.sh
# Monitor: tmux attach -t train125m
# Logs:   tail -f /workspace/logs/125m_*.jsonl

SESSION="train125m"
REPO="/workspace/boosted-attention"
LOG="/workspace/logs/run_125m_$(date +%Y%m%d_%H%M%S).log"

# Kill existing session if any
tmux kill-session -t $SESSION 2>/dev/null || true

tmux new-session -d -s $SESSION "bash -c '
cd $REPO
export DATA_DIR=/workspace/data
export CKPT_DIR=/workspace/checkpoints
export LOG_DIR=/workspace/logs
export RESULTS_DIR=/workspace/results

echo \"Starting 125M experiments at \$(date)\" | tee $LOG

# Standard × 2 seeds
echo \"\" | tee -a $LOG
echo \"=== standard seed42 ===\" | tee -a $LOG
python -u experiments/train_openwebtext.py --scale 125m --attn standard --seed 42 2>&1 | tee -a $LOG

echo \"\" | tee -a $LOG
echo \"=== standard seed123 ===\" | tee -a $LOG
python -u experiments/train_openwebtext.py --scale 125m --attn standard --seed 123 2>&1 | tee -a $LOG

# Boosted × 2 seeds
echo \"\" | tee -a $LOG
echo \"=== boosted seed42 ===\" | tee -a $LOG
python -u experiments/train_openwebtext.py --scale 125m --attn boosted --seed 42 2>&1 | tee -a $LOG

echo \"\" | tee -a $LOG
echo \"=== boosted seed123 ===\" | tee -a $LOG
python -u experiments/train_openwebtext.py --scale 125m --attn boosted --seed 123 2>&1 | tee -a $LOG

# Twicing × 2 seeds
echo \"\" | tee -a $LOG
echo \"=== twicing seed42 ===\" | tee -a $LOG
python -u experiments/train_openwebtext.py --scale 125m --attn twicing --seed 42 2>&1 | tee -a $LOG

echo \"\" | tee -a $LOG
echo \"=== twicing seed123 ===\" | tee -a $LOG
python -u experiments/train_openwebtext.py --scale 125m --attn twicing --seed 123 2>&1 | tee -a $LOG

# Param-fair × 2 seeds
echo \"\" | tee -a $LOG
echo \"=== param_fair seed42 ===\" | tee -a $LOG
python -u experiments/train_openwebtext.py --scale 125m --attn param_fair --seed 42 2>&1 | tee -a $LOG

echo \"\" | tee -a $LOG
echo \"=== param_fair seed123 ===\" | tee -a $LOG
python -u experiments/train_openwebtext.py --scale 125m --attn param_fair --seed 123 2>&1 | tee -a $LOG

# Post-training evaluation
echo \"\" | tee -a $LOG
echo \"=== Evaluating all 125M checkpoints ===\" | tee -a $LOG
python -u experiments/eval_benchmarks.py --scale 125m --eval-all 2>&1 | tee -a $LOG

echo \"\" | tee -a $LOG
echo \"All 125M experiments complete at \$(date)\" | tee -a $LOG
' 2>&1"

echo ""
echo "125M experiments launched in tmux session '$SESSION'"
echo "  Monitor:  tmux attach -t $SESSION"
echo "  Logs:     tail -f $LOG"
echo "  Status:   tmux ls"
echo ""
