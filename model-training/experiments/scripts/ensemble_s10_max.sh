#!/bin/bash
#SBATCH --job-name=ensemble_s10_max
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL

# S10 (réserve) — max(S9, S7) sample-par-sample
# Laisse le spécialiste dégradation "gagner" sur les cas extrêmes
# sans dégrader le bulk géré par l'ancien ensemble

source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/ensemble_max.py \
    --specialist submissions/submission_s9_new3_ensemble.csv \
    --generalist submissions/submission_s6_cnx_v1v2_b4.csv \
    --out submissions/submission_s10_max_new3_s6.csv
