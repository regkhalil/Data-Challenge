#!/bin/bash
#SBATCH --job-name=occlusion_predict_b4
#SBATCH --output=logs/predict_b4_%j.out
#SBATCH --error=logs/predict_b4_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

# Trouver le meilleur checkpoint B4
CKPT=$(ls checkpoints/run_b4_*_best.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "Aucun checkpoint B4 trouvé!"
    exit 1
fi
echo "Utilisation du checkpoint : $CKPT"

python src/predict.py \
    --config configs/config_b4.yaml \
    --checkpoint "$CKPT" \
    --tta 5

echo "B4 submission done"
ls -la submissions/
