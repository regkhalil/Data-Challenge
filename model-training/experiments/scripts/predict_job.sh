#!/bin/bash
#SBATCH --job-name=occlusion_predict
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kbelmajd-25@telecom-paris.fr

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

# B0 baseline (checkpoint from epoch 18)
python src/predict.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/baseline_b0_best.pth \
    --tta 1

echo "B0 submission done"
ls -la submissions/
