#!/bin/bash
#SBATCH --job-name=occlusion_b4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mail-type=NONE

module load python/3.11
module load cuda/12.4
source ~/envs/occlusion/bin/activate
cd ~/data_challenge

python src/train.py --config configs/config_b4.yaml --exp-name "run_b4_${SLURM_JOB_ID}"
