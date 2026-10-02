#!/bin/bash
#SBATCH --job-name=plantcnn_optuna_success
#SBATCH --output=outputs/logs/train_%j_%x.out
#SBATCH --error=outputs/logs/train_%j_%x.err
#SBATCH --time=05:00:00

source /opt/miniconda3/bin/activate /opt/conda_envs/course_env

python src/train.py --config config.yaml
