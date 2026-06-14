#!/bin/bash
#SBATCH --job-name=predict_swin
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kbelmajd-25@telecom-paris.fr

source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/predict.py \
    --config configs/config_swin_base.yaml \
    --checkpoint checkpoints/swin_base_v1_best.pth \
    --tta 5
