# Exercise 03: Image Classification with PyTorch

## Overview

In this exercise, you will build an end-to-end image classification pipeline from scratch using PyTorch. You will work with a curated subset of the [PlantWild](https://huggingface.co/datasets/Voxel51/PlantWild) dataset — an in-the-wild plant disease recognition dataset — to train a CNN that classifies plant diseases from leaf images.

The project structure is inspired by [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template), simplified for clarity.

**Estimated time:** 1.5-2 hours

**Prerequisites:** Exercise 02 (PyTorch fundamentals)

**File structure:**
```
exercise_03/
├── README.md                 ← You are here
├── config.yaml               ← Training configuration
├── submit_slurm.sh           ← Slurm submission script for training
├── submit_slurm_eval.sh      ← Slurm submission script for evaluation on testset
├── src/
│   ├── dataset.py            ← TODO: Custom Dataset class
│   ├── transforms.py         ← TODO: Train/val transform pipelines
│   ├── model.py              ← TODO: CNN architecture
│   ├── train.py              ← TODO: Training + validation loop
│   └── utils.py              ← TODO: Utility functions
├── data/                     ← Symlink to shared dataset
├── outputs/
│   ├── checkpoints/          ← Saved model weights
│   └── logs/                 ← TensorBoard log files
└── optional/                 ← Instructor-led demos (complete scripts)
```

---

## Dataset Setup

The PlantWild subset is pre-downloaded on the server at a shared, read-only location. **Do not copy the dataset.** Create a symlink instead:

```bash
ln -s /ai2_ex/data/classification/plantwild_subset data/plantwild
```

Your instructor will provide the exact path.

### Expected Folder Structure

The dataset follows the standard **ImageFolder** layout — one subfolder per class:

```
data/plantwild/
├── train/
│   ├── apple_scab/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   ├── tomato_early_blight/
│   │   ├── 001.jpg
│   │   └── ...
│   └── ... (10–15 class folders)
├── val/
│   ├── apple_scab/
│   │   └── ...
│   └── ...
└── test/
    ├── apple_scab/
    │   └── ...
    └── ...
```

Each subfolder name is the class label. The train/val/test split is already done — you do not need to split the data yourself.

---

# Part I: Concepts

---

## 1. Transform Pipelines

Before images can be fed into a CNN, they must be resized to a uniform dimension, converted to tensors, and normalized. Two separate pipelines are needed:

**Training transforms** include random augmentations — flips, crops, color jitter — that create slightly different versions of each image every epoch. This forces the model to learn features that are robust to these variations rather than memorizing specific pixel patterns. The model never sees the exact same image twice.

**Validation transforms** are deterministic — no randomness. The same image always produces the same tensor. This ensures validation metrics are stable and comparable across epochs.

Normalization uses ImageNet mean and standard deviation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) because these values are the standard for models pretrained on ImageNet. Even when training from scratch, using these values keeps the input distribution in a range that works well with standard weight initialization.

---

## 2. Custom Dataset

PyTorch's `Dataset` class provides a standard interface to load and preprocess data. For the ImageFolder layout, the Dataset scans the directory structure, maps sorted subfolder names to integer class indices (e.g. `{"apple_scab": 0, "tomato_blight": 1, ...}`), and returns `(image_tensor, label)` pairs.

The key methods:
- `__len__`: returns the total number of samples (used by DataLoader to know when an epoch ends)
- `__getitem__`: loads one image by index, applies transforms, returns `(image, label)`

Images are loaded lazily — only when `__getitem__` is called, not all at once. This keeps memory usage proportional to the batch size, not the dataset size.

---

## 3. CNN Architecture

A Convolutional Neural Network for classification follows a standard pattern: repeating blocks of `Conv2d → BatchNorm2d → ReLU → MaxPool2d` that progressively reduce spatial dimensions while increasing channels, followed by a classification head (`Linear` layer) that maps the learned features to class scores.

Each block extracts increasingly abstract features: early blocks detect edges and textures, middle blocks detect parts (leaf shapes, spots), and deeper blocks detect patterns specific to each disease. `MaxPool2d` halves the spatial dimensions at each stage, so the network's receptive field grows while the computation stays manageable.

`BatchNorm2d` normalizes activations within each batch, which stabilizes training and allows higher learning rates. `Dropout` in the classification head randomly zeroes neurons during training, preventing the model from relying too heavily on any single feature.

---

## 4. The Training Loop

The training loop repeats the same five steps for every batch, for every epoch:

1. **Forward pass** — images go through the model, producing logits (raw, unnormalized class scores)
2. **Loss computation** — `CrossEntropyLoss` compares logits against true labels
3. **Backward pass** — gradients are computed for every parameter
4. **Optimizer step** — weights are updated based on gradients
5. **Zero gradients** — old gradients are cleared before the next batch

After each epoch, the model is evaluated on the validation set (forward pass only, no backward/update) to track generalization.

### Training vs Evaluation Mode

`model.train()` enables dropout and batch normalization updates. `model.eval()` disables dropout (all neurons active) and freezes batch norm statistics. Forgetting to switch modes is a common bug — validation accuracy will appear worse than it should if dropout is still active.

### Config-Driven Training

All hyperparameters (learning rate, batch size, epochs, image size, etc.) are read from `config.yaml`. Nothing is hardcoded in the scripts. To compare experiments, copy the config, change one setting, and use a different `experiment_name` so TensorBoard logs don't overwrite.

---

## 5. Evaluation Metrics for Image Classification

### Accuracy

The simplest metric — what fraction of predictions are correct?

```
Accuracy = correct predictions / total predictions
```

Works well when classes are balanced (roughly equal number of samples per class). Misleading when classes are imbalanced — if 90% of images are "healthy", a model that always predicts "healthy" gets 90% accuracy while being useless for detecting diseases.

### Confusion Matrix

A table showing how predictions map to actual classes. Rows are true classes, columns are predicted classes. The diagonal shows correct predictions; off-diagonal shows errors.

```
                Predicted
              Scab  Blight  Healthy
True  Scab    [ 45    3       2   ]
      Blight  [  1   38       6   ]
      Healthy [  0    2      48   ]
```

Reveals which classes the model confuses — e.g. if the model frequently predicts "healthy" for "blight" images, the (Blight, Healthy) cell will be high. Far more informative than a single accuracy number.

### Precision, Recall, and F1-Score (per class)

For each class:

**Precision** — of all images the model predicted as this class, how many actually are?

```
Precision = True Positives / (True Positives + False Positives)
```

High precision means few false alarms. A model with high precision for "scab" rarely labels a healthy leaf as scab.

**Recall** — of all images that actually belong to this class, how many did the model find?

```
Recall = True Positives / (True Positives + False Negatives)
```

High recall means few misses. A model with high recall for "scab" catches most scab cases, even if it occasionally misclassifies something else as scab.

**F1-Score** — the harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Balances both — a model needs both high precision and high recall to achieve a high F1. Useful when you care about both false alarms and misses equally.

### Top-k Accuracy

Instead of checking only the top prediction, check if the correct class is among the top k predictions. `Top-5 accuracy` is common in ImageNet benchmarks — it asks "is the correct class in the model's 5 most confident guesses?"

Useful when classes are visually similar (e.g. different types of leaf blight that look alike). A model might rank the correct class second, which counts as wrong for top-1 accuracy but correct for top-5.

---

## 6. Practical Considerations

### Overfitting and Underfitting

**Overfitting**: training loss keeps decreasing but validation loss starts increasing. The model is memorizing training data rather than learning generalizable features. Signs: high training accuracy, low validation accuracy. Remedies: more data, stronger augmentation, dropout, weight decay, early stopping.

**Underfitting**: both training and validation loss are high. The model is too simple or hasn't trained long enough. Signs: low accuracy on both sets. Remedies: larger model, more epochs, higher learning rate.

Watch the gap between training and validation curves in TensorBoard — a growing gap signals overfitting.

### Learning Rate

The most impactful hyperparameter. Too high: loss oscillates or diverges. Too low: training is slow and may get stuck. Typical starting values: 0.001 for Adam, 0.01 for SGD with momentum. Exercise 04 will cover systematic hyperparameter search with Optuna.

### Batch Size

Affects both training speed and generalization. Larger batches give more stable gradient estimates and train faster (better GPU utilization), but can converge to sharper minima that generalize worse. Smaller batches add noise that can help escape local minima. Common range: 16–64 for image classification with limited GPU memory.

### Checkpointing

Saving the model's state after each epoch (or only when validation accuracy improves) ensures you don't lose progress if training crashes. The best checkpoint — the one with the highest validation accuracy — is what you use for final evaluation, not the last epoch's weights.

---

## 7. TensorBoard

TensorBoard is a visualization tool that tracks training progress in real time. The training script logs scalar metrics (loss, accuracy per epoch) and sample images.

**Starting TensorBoard on the server:**

```bash
tensorboard --logdir outputs/logs --port 6006
```

If port 6006 is taken by another student, use a different port (e.g. 6007, 6008).

You should see:
- **Scalars tab:** train/val loss and accuracy curves per epoch
- **Images tab:** sample training images (logged in the first epoch)

---

## 8. Optional Advanced Topics

The scripts in the `optional/` folder are complete, working implementations. Your instructor will demonstrate and explain these in class. You are encouraged to read the code and run them yourself, but they are not required.

| Script | Topic |
|--------|-------|
| `optional/cross_validation.py` | K-fold cross-validation using `sklearn.model_selection.KFold` |
| `optional/early_stopping.py` | Patience-based early stopping on validation loss |
| `optional/weighted_loss.py` | Weighted `CrossEntropyLoss` for imbalanced class distribution |
| `optional/mixup.py` | MixUp data augmentation during training |
| `optional/feature_visualization.py` | Visualizing learned filters and feature maps using forward hooks |

---

# Part II: Exercises

Work through each script in order. Open the file, look for `# TODO` markers, write your solution, and run the script to verify.

---

### Exercise A — Transform pipelines (section 1)

**→ Open `src/transforms.py` and complete the TODOs.**

Covers: building separate train (with augmentation) and val (deterministic) transform pipelines using `torchvision.transforms.Compose`, with `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `ToTensor`, and `Normalize`.

Verify: `python src/transforms.py`

---

### Exercise B — Custom Dataset (section 2)

**→ Open `src/dataset.py` and complete the TODOs.**

Covers: scanning the ImageFolder directory, building a sorted class-to-index mapping, loading images with PIL, applying transforms in `__getitem__`.

Verify: `python src/dataset.py`

This prints sample counts, class names, batch shapes, and per-class distribution. Fix any errors before moving on.

---

### Exercise C — CNN architecture (section 3)

**→ Open `src/model.py` and complete the TODOs.**

Covers: building PlantCNN with Conv-BN-ReLU-Pool blocks, `AdaptiveAvgPool2d`, a classifier head with dropout, and `forward()`. Also: counting trainable parameters.

Verify: `python src/model.py`

---

### Exercise D — Utility functions (section 4)

**→ Open `src/utils.py` and complete the TODOs.**

Covers: `set_seed()` for reproducibility, `compute_accuracy()` from logits and targets, `save_checkpoint()` and `load_checkpoint()` with `state_dict`.

Verify: `python src/utils.py`

---

### Exercise E — Training script (section 4)

**→ Open `src/train.py` and complete the TODOs.**

Requires: all previous scripts completed (imports from each).

Covers: loading config, creating datasets/dataloaders, instantiating model/loss/optimizer, the training loop (forward → loss → backward → step), the validation loop (with `torch.no_grad()`), TensorBoard logging with `SummaryWriter`, and best-checkpoint saving.

---

### Exercise F — Run training

Submit the training job via Slurm:

```bash
sbatch submit_slurm.sh
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

### Exercise G — Compare with vs without augmentation

1. Run training once with the default `config.yaml` (augmentation enabled).
2. Modify `config.yaml`: set `augmentation: false`.
3. Run training again (use a different `experiment_name` so TensorBoard logs don't overwrite).
4. Open TensorBoard and compare the two runs side by side — observe the difference in validation accuracy and whether the augmented model generalizes better.

---

### Exercise H — Evaluate on test set

After training completes, evaluate the best checkpoint on the held-out test set.

Edit `slurm_submit_eval.sh` if needed — update the checkpoint path to match your best saved model. Then submit:

```bash
sbatch submit_slurm_eval.sh
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

### Exercise I — Push to GitHub

1. Make sure `data/`, `outputs/checkpoints/`, and `outputs/logs/` are in your `.gitignore`.
2. Stage, commit, and push:

```bash
git add src/
git commit -m "Complete exercise 03: image classification pipeline"
git push
```

---

## Summary

| Component | File | What you built |
|-----------|------|----------------|
| Transforms | `src/transforms.py` | Train (with augmentation) and val pipelines |
| Dataset | `src/dataset.py` | Custom Dataset class for ImageFolder layout |
| Model | `src/model.py` | CNN from scratch with Conv-BN-ReLU-Pool blocks |
| Utilities | `src/utils.py` | Seed setting, checkpoint save/load, accuracy |
| Training | `src/train.py` [slurm job] | Full train/val loop with TensorBoard and checkpointing |
| Evaluation | `src/evaluate.py` [slurm job] | Evaluating the best model on held-out testset |

This is the standard structure of a PyTorch classification project. Detection and segmentation exercises that follow will extend this pattern — same file layout, same training loop skeleton, different model architectures and loss functions.

