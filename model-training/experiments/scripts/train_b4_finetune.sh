#!/bin/bash
#SBATCH --job-name=b4_finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/train.py \
    --config configs/config_b4_finetune.yaml \
    --resume checkpoints/run_b4_842870_best.pth
