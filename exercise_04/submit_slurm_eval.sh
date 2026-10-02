#!/bin/bash
#SBATCH --job-name=eval_plantcnn
#SBATCH --output=outputs/logs/eval_%j_%x.out
#SBATCH --error=outputs/logs/eval_%j_%x.err
#SBATCH --time=00:30:00

source /opt/miniconda3/bin/activate /opt/conda_envs/course_env

python src/evaluate.py --config config.yaml \
    --checkpoint outputs/checkpoints/plantwild_plantcnn_best.pt
