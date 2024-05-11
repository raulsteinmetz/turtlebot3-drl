#!/bin/bash

agent="${1:-ddpg}"
stage="${2:-1}"
# load_models="${3:-False}"

echo "Agent: $agent - Stage: $stage - Load_models=$load_models"

sleep 5

# echo "...Training..."
# python3 train.py --agent $agent --stage $stage --load $load_models &&
# echo "...Saving models to best_models..."
# python3 save_to_best.py --agent  $agent --stage $stage &&
# echo "...Testing..."
# python3 test.py --agent $agent --stage $stage &&
# echo "...Plotting learning curve..."
# python3 learning_curve.py --agent $agent --stage $stage

echo "...Training..."
python3 train.py --agent $agent --stage $stage &&
echo "...Saving models to best_models..."
python3 save_to_best.py --agent  $agent --stage $stage &&
echo "...Testing..."
python3 test.py --agent $agent --stage $stage &&
echo "...Plotting learning curve..."
python3 learning_curve.py --agent $agent --stage $stage