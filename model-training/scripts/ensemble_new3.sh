#!/bin/bash
#SBATCH --job-name=ensemble_new3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kbelmajd-25@telecom-paris.fr

# S9 — Ensemble des 3 modèles dégradation-aware
# cnx_synth_ft (fine-tune ConvNeXt + synth_degrad) + swin_base_v1 + effnetv2_m_v1
# Tous ont synth_degrad_p=0.12 → ils voient scan-lines/poster/flou = haute occlusion
# Poids : cnx_synth plus fort (warm-start depuis meilleur modèle existant)

source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/ensemble_csv.py \
    --csvs \
        submissions/submission_cnx_synth_ft.csv \
        submissions/submission_swin_base_v1.csv \
        submissions/submission_effnetv2_m_v1.csv \
    --weights 0.45 0.275 0.275 \
    --out submissions/submission_s9_new3_ensemble.csv
