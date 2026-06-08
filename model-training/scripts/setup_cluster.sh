#!/bin/bash
# À exécuter une seule fois sur le cluster pour créer l'environnement
set -e

mkdir -p ~/data_challenge/logs ~/data_challenge/checkpoints ~/data_challenge/submissions

python3 -m venv ~/envs/occlusion
source ~/envs/occlusion/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm pandas numpy scikit-learn pyyaml tqdm

echo "Setup terminé. Lance : source ~/envs/occlusion/bin/activate"
