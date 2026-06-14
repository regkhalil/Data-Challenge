#!/bin/bash
#SBATCH --job-name=analyze_val
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/analyze_val_error.py \
    --config configs/config_convnext_base_v2.yaml \
    --checkpoint checkpoints/convnext_base_v2_best.pth \
    --out logs/val_error_decomp.csv
