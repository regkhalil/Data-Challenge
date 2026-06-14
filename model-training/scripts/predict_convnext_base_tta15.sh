#!/bin/bash
#SBATCH --job-name=predict_convnext_tta15
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/predict.py \
    --config configs/config_convnext_base_tta15.yaml \
    --checkpoint checkpoints/convnext_base_v1_best.pth \
    --tta 15
