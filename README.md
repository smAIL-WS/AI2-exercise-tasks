# AI2 Course — Hands-on Exercises

This repository contains all exercises for the AI2 course. Each exercise builds on the previous one, progressing from Python and PyTorch fundamentals to training deep learning models for image classification, object detection, and semantic segmentation.

## Exercises

| Exercise | Topic | What You Build |
|----------|-------|----------------|
| [Exercise 00](exercise_01/exercise_00_setup.md) | Environment Setup | VS Code, SSH connection, conda environment, Slurm basics |
| [Exercise 01](exercise_01/exercise_01_github_intro.md) | Git & GitHub | Fork, clone, commit, push, branches, .gitignore |
| [Exercise 02](exercise_02/README.md) | PyTorch Fundamentals | Tensors, autograd, nn.Module, loss, optimizer, dataset, dataloader |
| [Exercise 03](exercise_03/README.md) | Image Classification | CNN from scratch, training loop, TensorBoard, checkpointing |
| [Exercise 04](exercise_04/README.md) | Hyperparameter Optimization | Optuna search, pretrained models, model factory |
| [Exercise 05](exercise_05/README.md) | Object Detection | Faster R-CNN, COCO evaluation, RT-DETR (transformer) |
| [Exercise 06](exercise_06/README.md) | Semantic Segmentation | U-Net, mIoU/Dice metrics, SegFormer (transformer) |
| [Exercise 07](exercise_07/README.md) | Fine-Tuning | Freezing layers, few-shot learning, foundation models |
| [Exercise 08](exercise_08/README.md) | Fine-Tuning - Extended | PEFT, distillation |

## How Each Exercise Works

Every exercise README is split into two parts:

- **Part I — Concepts:** Background theory explained by the instructor in class. Read this to understand what you're building and why.
- **Part II — Exercises:** Step-by-step tasks you solve by completing TODO markers in the Python scripts. Each task has a verification command to check your work.


## Repository Structure

```
AI2-exercise/
├── README.md                 ← You are here
├── exercise_00/              ← Environment setup
├── exercise_01/              ← Git & GitHub
├── exercise_02/              ← PyTorch fundamentals
├── exercise_03/              ← Image classification
├── exercise_04/              ← Hyperparameter optimization
├── exercise_05/              ← Object detection
├── exercise_06/              ← Semantic segmentation
└── exercise_07/              ← Fine-tuning
└── exercise_08/              ← Fine-tuning - Extended
```

Each exercise folder contains:

```
exercise_XX/
├── README.md                 ← Instructions (concepts + tasks)
├── config.yaml               ← Training configuration
├── submit_slurm.sh           ← GPU job submission script
├── src/                      ← Scripts with TODO markers
├── data/                     ← Symlink to shared dataset
└── outputs/                  ← Checkpoints, logs, visualizations
```

## Dataset

All exercises that use data (03–07) share datasets stored at a central location on the server. You do not copy data — each exercise README tells you how to create a symlink. The datasets are read-only.

## GPU Usage

All GPU-intensive work must be submitted through Slurm:

```bash
sbatch submit_slurm.sh        # submit a training job
squeue -u $USER               # check your job status
scancel <job_id>               # cancel a job
```

Do not run training scripts directly in the terminal. See [Exercise 00](exercise_00/README.md) Section 5 for details.

## Course Progression

```
Ex00–01: Setup & Git
    │
    ▼
Ex02: PyTorch Basics (tensors, autograd, nn.Module)
    │
    ▼
Ex03: Classification (CNN, training loop, TensorBoard)
    │
    ▼
Ex04: HPO + Pretrained Models (Optuna, model factory)
    │
    ▼
Ex05: Detection (Faster R-CNN, COCO metrics, RT-DETR)
    │
    ▼
Ex06: Segmentation (U-Net, mIoU/Dice, SegFormer)
    │
    ▼
Ex07: Fine-Tuning (freezing, few-shot, foundation models)
    │
    ▼
Ex08: Fine-Tuning - Extended (PEFT, distillation)
```

Each exercise reuses and extends the patterns from the previous one. The training loop structure, config-driven design, and model factory pattern remain consistent throughout — only the task, model architecture, and evaluation metrics change.
