#!/bin/bash
#SBATCH --job-name=predict_cnx_synth
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/predict.py \
    --config configs/config_cnx_synth.yaml \
    --checkpoint checkpoints/cnx_synth_ft_best.pth \
    --tta 5
