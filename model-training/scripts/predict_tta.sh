#!/bin/bash
#SBATCH --job-name=predict_tta
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/predict.py \
    --config configs/config_b4_base.yaml \
    --checkpoint checkpoints/run_b4_842870_best.pth \
    --tta 5
