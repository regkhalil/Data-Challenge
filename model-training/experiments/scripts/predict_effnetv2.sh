#!/bin/bash
#SBATCH --job-name=predict_effnetv2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/predict.py \
    --config configs/config_effnetv2_m.yaml \
    --checkpoint checkpoints/effnetv2_m_v1_best.pth \
    --tta 5
