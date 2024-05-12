#!/bin/bash

agent="${1:-ddpg}"
stage="${2:-1}"
lidar="${3:-10}"
# load_models="${4:-False}"

echo "Agent: $agent - Stage: $stage - Lidar=$lidar"
sleep 5


echo "...Training..."
python train.py --agent $agent --stage $stage --lidar $lidar
sleep 5

echo "...Saving models to best_models..."
python save_to_best.py --agent  $agent --stage $stage --lidar $lidar
sleep 5

echo "...Testing..."
python test.py --agent $agent --stage $stage --lidar $lidar
sleep 5

echo "...Plotting learning curve..."
python learning_curve.py --agent $agent --stage $stage --lidar $lidar