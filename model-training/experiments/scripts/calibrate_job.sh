#!/bin/bash
#SBATCH --job-name=calibrate_gender
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kbelmajd-25@telecom-paris.fr

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/calibrate_gender.py \
    --config configs/config_convnext_base.yaml \
    --checkpoint checkpoints/convnext_base_v1_best.pth \
    --submission submissions/submission_mega_ensemble.csv \
    --out submissions/submission_mega_calibrated.csv
