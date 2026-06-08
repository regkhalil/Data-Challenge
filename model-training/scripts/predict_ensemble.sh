#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kbelmajd-25@telecom-paris.fr

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

# Vérifie que les deux checkpoints existent
if [ ! -f checkpoints/run_b4_842870_best.pth ]; then
    echo "ERREUR : checkpoint B4 manquant"
    exit 1
fi
if [ ! -f checkpoints/convnext_small_v1_best.pth ]; then
    echo "ERREUR : checkpoint ConvNeXt manquant"
    exit 1
fi

echo "=== Ensemble B4 + ConvNeXt-Small | TTA x5 ==="

python src/ensemble.py \
    --models configs/config_b4_base.yaml:checkpoints/run_b4_842870_best.pth \
             configs/config_convnext.yaml:checkpoints/convnext_small_v1_best.pth \
    --tta 5 \
    --weights 0.5 0.5 \
    --out submissions/submission_ensemble_b4_convnext.csv
