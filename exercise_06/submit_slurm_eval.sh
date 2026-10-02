
#!/bin/bash
#SBATCH --job-name=eval_%u
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --time=01:00:00

source /opt/miniconda3/bin/activate /opt/conda_envs/course_env

python src/evaluate.py --config config.yaml \
    --checkpoint outputs/checkpoints/phenobench_unet_best.pt
