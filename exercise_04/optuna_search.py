"""
Exercise 04 - Hyperparameter Optimization with Optuna
======================================================
This script uses Optuna to search for the best hyperparameters for PlantCNN.

Hyperparameters tuned: learning_rate, dropout, batch_size
Fixed: optimizer (AdamW), architecture (PlantCNN)

Reference:
    https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py

Usage:
    python optuna_search.py --config config.yaml
"""

import os
import sys
import copy
import argparse

import yaml
import optuna
import matplotlib
matplotlib.use("Agg")  # headless backend for SLURM nodes
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from train import train_and_evaluate


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def objective(trial, base_config):
    """
    Optuna objective function — called once per trial.

    Args:
        trial (optuna.Trial): Optuna trial object with suggest_* methods.
        base_config (dict): Base configuration dictionary.

    Returns:
        float: Best validation accuracy for this trial.
    """
    # Deep copy so each trial starts with a clean config
    config = copy.deepcopy(base_config)

    # TODO: Suggest hyperparameter values and update the config.
    #
    #   Step 1: Suggest a learning rate.
    #           Use trial.suggest_float() with name="learning_rate".
    #           Search between 1e-5 and 1e-2.
    #           Set log=True — learning rate is best searched on a
    #           logarithmic scale because the difference between
    #           0.001 and 0.01 matters more than between 0.091 and 0.1.
    #
    #   Step 2: Suggest a dropout value.
    #           Use trial.suggest_float() with name="dropout".
    #           Search between 0.2 and 0.7 (uniform scale is fine here).
    #
    #   Step 3: Suggest a batch size.
    #           Use trial.suggest_categorical() with name="batch_size".
    #           Choose from [16, 32, 64].
    #           Categorical because batch size must be a specific value,
    #           not a continuous range.
    #
    #   Step 4: Update the config dict with the suggested values:
    #           config["training"]["learning_rate"] = lr
    #           config["model"]["dropout"] = dropout
    #           config["data"]["batch_size"] = batch_size

    # TODO: Call train_and_evaluate(config) and return the result.
    #       This trains the model with the suggested hyperparameters
    #       and returns the best validation accuracy.
    val_accuracy = None

    return val_accuracy


def main(config_path):
    """Run Optuna hyperparameter search."""
    base_config = load_config(config_path)
    print(f"Loaded base config from: {config_path}")
    print(f"Starting Optuna hyperparameter search...")

    # TODO: Create an Optuna study.
    #
    #   Step 1: Use optuna.create_study().
    #   Step 2: Set direction="maximize" — we want the highest
    #           validation accuracy, not the lowest.
    #   Step 3: Optionally set study_name to the experiment name from config.
    study = None

    # TODO: Run the optimization.
    #
    #   Step 1: Call study.optimize().
    #   Step 2: The first argument is the objective function. Since our
    #           objective takes two arguments (trial, base_config) but
    #           Optuna passes only the trial, use a lambda:
    #           lambda trial: objective(trial, base_config)
    #   Step 3: Set n_trials=5 in the config file (config["training"]["n_trials"]) to limit the number of trials.


    # --- Print results ---
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nNumber of finished trials: {len(study.trials)}")

    # TODO: Print the best trial's results.
    #
    #   Step 1: Get the best trial using study.best_trial.
    #   Step 2: Print its .value (the best validation accuracy).
    #   Step 3: Print its .params (dict of hyperparameters that achieved it).


    # --- Print all trials summary ---
    print("\nAll trials:")
    for t in study.trials:
        print(f"  Trial {t.number}: val_acc={t.value:.4f} | params={t.params}")

    # --- Visualization ---
    os.makedirs("outputs/optuna", exist_ok=True)

    ax = plot_optimization_history(study)
    ax.figure.savefig(f"outputs/optuna/{base_config['experiment_name']}_optimization_history.png", bbox_inches="tight")
    plt.close(ax.figure)
    
    ax = plot_param_importances(study)
    ax.figure.savefig(f"outputs/optuna/{base_config['experiment_name']}_param_importances.png", bbox_inches="tight")
    plt.close(ax.figure)
    
    ax = plot_parallel_coordinate(study)
    ax.figure.savefig(f"outputs/optuna/{base_config['experiment_name']}_param_relationships.png", bbox_inches="tight")
    plt.close(ax.figure)
    
    print("\nVisualization plots saved to outputs/optuna/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to base config YAML file")
    args = parser.parse_args()

    main(args.config)
