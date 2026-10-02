# Exercise 04: Hyperparameter Optimization with Optuna

## Overview

In Exercise 03, you trained a CNN for plant disease classification with hyperparameters (learning rate, dropout, batch size) chosen manually. In this exercise, you will automate that search using [Optuna](https://optuna.org/), a hyperparameter optimization framework that uses Bayesian principles to find better configurations with fewer trials than brute-force grid search.

The code from Exercise 03 (`src/`) is provided fully implemented. Your only task is to complete `optuna_search.py` — the script that defines the search space, runs the optimization, and visualizes the results.

**Estimated time:** 60–90 minutes

**Prerequisites:** Exercise 03 (image classification pipeline)

**Reference:** This exercise follows the structure of Optuna's official PyTorch example:
[optuna/optuna-examples/pytorch/pytorch_simple.py](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py)

---

## Project Structure

```
exercise_04/
├── README.md                 ← You are here
├── config.yaml               ← Base configuration
├── optuna_search.py          ← TODO: Define search space and run optimization
├── submit_slurm.sh              ← Slurm submission script for training
├── submit_slurm_eval.sh         ← Slurm submission script for evaluation on testset
├── src/
│   ├── dataset.py            ← Fully coded (from Exercise 03)
│   ├── transforms.py         ← Fully coded
│   ├── model.py              ← Fully coded
│   ├── train.py              ← Fully coded (exposes train_and_evaluate function)
│   └── utils.py              ← Fully coded
├── solutions/
│   └── optuna_search.py
├── data/                     ← Symlink to shared dataset
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── optuna/               ← Saved visualization plots
```

---

## Dataset Setup

Same as Exercise 03. If you have not already created the symlink:

```bash
ln -s /ai2_ex/data/classification/plantwild_subset data/plantwild
```

---

# Part I: Concepts
 
---
 
## 1. What is Hyperparameter Optimization?
 
In machine learning, there are two types of parameters:
 
- **Model parameters**: learned during training (weights, biases). You do not set these manually — backpropagation handles them.
- **Hyperparameters**: set *before* training begins and control how the model learns. Examples: learning rate, dropout probability, batch size, optimizer choice, number of layers.
Choosing good hyperparameters matters. A learning rate that is too high causes divergence; too low and training stalls. The right dropout prevents overfitting; too much and the model underfits. Manually trying combinations is slow and does not scale — you need a systematic approach.
 
---

## 2. Approaches to Hyperparameter Search
 
### Grid Search
 
Try every combination on a predefined grid (e.g. lr ∈ {0.1, 0.01, 0.001} × dropout ∈ {0.3, 0.5, 0.7}). Simple but expensive — 3 × 3 = 9 trials, and this grows exponentially with the number of hyperparameters. Most combinations are wasted on unimportant regions of the search space, and the rigid spacing can miss good values between grid points.
 
### Random Search
 
Sample hyperparameters randomly from defined ranges. Surprisingly effective — often finds good configurations faster than grid search because it covers more of the space per trial. But it does not learn from previous trials. Trial 50 is sampled with the same strategy as trial 1, even if by trial 10 it was already clear that learning rates below 1e-4 consistently underperform.

### Bayesian Optimization

The key insight: **use the results of previous trials to decide what to try next.**

Bayesian optimization builds a probabilistic model (called a *surrogate*) of how hyperparameters map to performance. After each trial, the surrogate is updated, and the next set of hyperparameters is chosen to maximize the *expected improvement* — balancing exploration (trying uncertain regions) with exploitation (refining near the current best).

**Note**: Grid search evaluates every combination on a fixed grid — the number of trials grows exponentially with each new hyperparameter, and most land in unproductive regions. Random search covers the space more efficiently but treats every trial as independent, learning nothing from previous results. Bayesian optimization addresses both problems: it uses results from completed trials to decide where to search next, concentrating evaluations on promising regions rather than wasting compute on configurations that are unlikely to improve. In practice, 5 well-directed Bayesian trials can outperform 30 random ones.

## 3. What is Optuna?
 
Optuna is a Python framework for hyperparameter optimization. Its core concepts:
 
- **Study**: an optimization session — a collection of trials.
- **Trial**: a single evaluation — one set of hyperparameters, one training run, one result.
- **Objective function**: the function Optuna optimizes. It receives a `trial` object, suggests hyperparameters, trains the model, and returns the metric to optimize (e.g. validation accuracy).
- **`trial.suggest_*` methods**: define the search space inside the objective function.

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.7)
    # ... train model with these hyperparameters ...
    return val_accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
```

Optuna implements this using the **Tree-structured Parzen Estimator (TPE)** algorithm:
 
1. Run a trial with suggested hyperparameters.
2. Observe the result (e.g. validation accuracy).
3. Update the surrogate model — separate the hyperparameter distributions into "good" (above median performance) and "bad" (below median).
4. Sample the next trial's hyperparameters from the "good" distribution, biased toward regions that previously produced better results.
5. Repeat.
This means Optuna spends more time evaluating promising regions and less time on configurations that are unlikely to improve.

### Suggest Methods

| Method | Usage | Example |
|--------|-------|---------|
| `suggest_float(name, low, high)` | Continuous float | `trial.suggest_float("lr", 1e-5, 1e-2, log=True)` |
| `suggest_int(name, low, high)` | Integer | `trial.suggest_int("n_layers", 1, 5)` |
| `suggest_categorical(name, choices)` | Categorical choice | `trial.suggest_categorical("optimizer", ["adam", "sgd"])` |

The `log=True` flag samples on a logarithmic scale — essential for learning rates where the difference between 0.001 and 0.01 matters more than between 0.091 and 0.1.

---

## 4. Which Hyperparameters to Tune?

### Training hyperparameters (tuned in this exercise)

These control *how* the model learns, independent of architecture:

- **Learning rate** — most impactful hyperparameter; always worth tuning.
- **Dropout** — regularization strength; depends on dataset size and complexity.
- **Batch size** — affects gradient noise and convergence speed.

### Optimizer choice

In this exercise, we fix the optimizer to **AdamW**. AdamW (Adam with decoupled weight decay) is a strong default for most vision tasks. While it is possible to tune the optimizer type (Adam vs. SGD vs. AdamW), in practice AdamW works well enough that the gains from tuning this are usually small compared to tuning learning rate and regularization.

### Architectural hyperparameters

It is possible to tune architectural parameters (number of layers, channels per layer, kernel sizes) with Optuna's `suggest_int` and `suggest_categorical`. This makes sense when building a custom architecture from scratch.

However, in most practical settings, you use established architectures (ResNet, VGG, EfficientNet) that have already been optimized through extensive research. Tuning their architecture would mean designing a new model — a separate research problem. For pretrained models, the architecture is fixed and only training hyperparameters are tuned.

In this exercise, the PlantCNN architecture remains fixed. Only training hyperparameters are searched.

---

## 5. Pruning (Not Used in This Exercise)

Optuna supports **pruning** — automatically stopping unpromising trials early. If a trial's validation loss at epoch 5 is already worse than the median of completed trials, there is no reason to continue training it for 30 epochs. Pruning saves significant GPU time.

Optuna provides several pruning strategies (MedianPruner, PercentilePruner, HyperbandPruner). To use them, you report intermediate values with `trial.report(val_accuracy, epoch)` and check `trial.should_prune()` after each epoch.

We do not use pruning in this exercise to keep the implementation simple, but for real optimization runs with many trials and long training times, pruning is strongly recommended.

---

## 6. Using Pretrained Models
 
Instead of training a CNN from scratch, you can leverage architectures that have been pretrained on ImageNet (1.2 million images, 1000 classes). The pretrained weights already capture general visual features (edges, textures, shapes) — fine-tuning adapts them to your specific dataset. This is called **transfer learning**.
 
To switch models, change `model.name` in `config.yaml`:
 
```yaml
model:
  name: "resnet18"       # change this
  pretrained: true       # ignored for plantcnn; loads ImageNet weights for all others
```
 
### Available Models
 
| Model | Params | Year | Notes |
|-------|--------|------|-------|
| `plantcnn` | ~100K | — | Custom CNN from Exercise 03. Baseline — no pretrained weights. |
| `resnet18` | 11.7M | 2015 | Introduced skip connections. Best speed/accuracy trade-off. **Recommended.** |
| `resnet50` | 25.6M | 2015 | Deeper ResNet. More capacity but slower, may overfit on small datasets. |
| `vgg16` | 138M | 2014 | Simple stacked 3×3 convolutions. Very large — demonstrates more parameters ≠ better results. |
| `alexnet` | 61M | 2012 | First deep CNN to win ImageNet. Historically important, outperformed by all modern architectures. |
 
### Suggested Experiment
 
1. Train with `plantcnn` (your baseline from Exercise 03).
2. Switch to `resnet18` with `pretrained: true` and train again.
3. Compare validation accuracy in TensorBoard — the pretrained model should converge faster and achieve higher accuracy, even with the same number of epochs.
---
 
## 7. Optuna Visualization
 
After the optimization run, Optuna provides built-in plotting functions to analyze the results:
 
- **Optimization history** — shows how the best objective value improved over successive trials. A steep early improvement followed by a plateau means Optuna found a good region quickly.
- **Hyperparameter importances** — ranks which hyperparameters had the most impact on the result. If learning rate dominates and dropout barely matters, you know where to focus next.
- **Parallel coordinate plot** — shows the relationship between all parameter values and the objective across trials. Reveals whether high-performing trials cluster in a specific parameter range.


# Part II: Exercises
 
---
 
### Exercise A — Understand the training interface (section 3)
 
Open `src/train.py` and read the `train_and_evaluate()` function. It takes a config dictionary, trains the model, and returns the best validation accuracy. This is the function your Optuna objective will call — you do not modify it.
 
---
 
### Exercise B — Complete the Optuna search script (sections 3–4)
 
**→ Open `optuna_search.py` and complete the TODOs.**
 
Covers: defining the `objective(trial)` function — suggesting learning rate (log scale), dropout (uniform), and batch size (categorical), updating the config dict, calling `train_and_evaluate()`, returning the result. Creating a study with `direction="maximize"`. Running optimization with `n_trials=5`. Printing the best trial's hyperparameters and accuracy. Saving visualization plots.
 
---
 
### Exercise C — Run the optimization

Submit the training job via Slurm:

```bash
sbatch submit_slurm_optuna.sh
```

Monitor the job:

```bash
# Check job status
squeue -u $USER

# View output while running
tail -f outputs/logs/slurm_*.out

# View errors
cat outputs/logs/slurm_*.err
```
 
---
 
### Exercise D — Analyze results (section 7)
 
After optimization completes, check:
- Terminal output for the best trial's hyperparameters and accuracy.
- Saved plots in `outputs/optuna/`:
  - `optimization_history.png` — objective value over trials.
  - `param_importances.png` — which hyperparameters mattered most.
  - `param_relationships.png` — parallel coordinate view.
---
 
### Exercise E — Try pretrained models (section 6)
 
1. Train with `plantcnn` using the best hyperparameters from the Optuna search by adapting the values inside the config file.
```bash
sbatch submit_slurm_optuna.sh
```
2. Switch `model.name` to `resnet18` in `config.yaml` and train again. Make sure to rename the exeriment name to avoid overwriting the previous results. Optional: try `resnet50`, `vgg16`, or `alexnet` as well. Submit the new job.
3. Compare results in TensorBoard.
---

### Exercise F — Evaluate on test set

After training completes, evaluate the best checkpoint on the held-out test set.

Edit `slurm_submit_eval.sh` if needed — update the checkpoint path to match your best saved model. Then submit:

```bash
sbatch slurm_submit_eval.sh
```

Check the results:

```bash
cat outputs/logs/eval_*.out
```

This prints overall accuracy, per-class precision/recall/F1, and a confusion matrix. By default it evaluates on the **test** split. To evaluate on val instead, edit the last line in `slurm_submit_eval.sh`:

```bash
python src/evaluate.py --config config.yaml \
    --checkpoint outputs/checkpoints/plantwild_best.pt \
    --split val
```
---
 
### Exercise G — Push to GitHub
 
```bash
git add optuna_search.py
git commit -m "Complete exercise 04: Optuna hyperparameter optimization"
git push
```
 
---
 
## Summary
 
| Concept | What you learned |
|---------|-----------------|
| Grid vs. random vs. Bayesian search | Why Bayesian optimization is more efficient |
| TPE algorithm | How Optuna uses past trials to guide future ones |
| Optuna study/trial/objective | The core API for defining and running a search |
| `suggest_*` methods | How to define search spaces for different parameter types |
| Training vs. architectural hyperparameters | When to tune what |
| Pruning (concept) | Why early stopping of bad trials saves compute |
| Transfer learning | Why pretrained models outperform training from scratch |
| Model factory pattern | How to swap architectures via config without changing training code |

