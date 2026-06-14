#!/bin/bash
#SBATCH --job-name=mega_ensemble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kbelmajd-25@telecom-paris.fr

source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/ensemble_csv.py \
    --csvs \
        submissions/submission_run_b4.csv \
        submissions/submission_b4_v2_s123.csv \
        submissions/submission_convnext_base_v1.csv \
        submissions/submission_vit_base_v1.csv \
    --weights 0.15 0.20 0.35 0.30 \
    --out submissions/submission_mega_ensemble_4_models.csv
