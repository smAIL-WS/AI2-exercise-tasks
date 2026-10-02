#!/bin/bash
#SBATCH --job-name=optuna_plantcnn_%u
#SBATCH --output=outputs/logs/optuna_%j_%x.out
#SBATCH --error=outputs/logs/optuna_%j_%x.err
#SBATCH --time=05:00:00

source /opt/miniconda3/bin/activate /opt/conda_envs/course_env

python optuna_search.py --config config.yaml
