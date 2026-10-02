#!/bin/bash
#SBATCH --job-name=training_%u
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err
#SBATCH --time=02:00:00

source /opt/miniconda3/bin/activate /opt/conda_envs/course_env

python src/train.py --config config.yaml
