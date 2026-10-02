# Exercise 07: Few-Shot Finetuning for Segmentation

## Overview

Exercise 06 trained a U-Net **from scratch** on PhenoBench's full training
split — hundreds of images, 30 epochs, every weight starting random. This
exercise asks the opposite question: if you only had a *handful* of labeled images, how can we still get a model that segments crop from weed?

### Transfer Learning Strategies
 
In deep learning, starting from a model that has already learned on a large dataset and adapting it to a new task is called **transfer learning**. There is a spectrum of approaches depending on how much labeled data you have for your target task:

**Zero-shot** — Use a pretrained model directly on the new task with no training at all. The model has never seen your data or your classes. It works only when the pretrained model's knowledge generalizes to your domain — for example, a model trained on general scene segmentation might label "vegetation" but would not distinguish crop from weed. Zero-shot is a useful baseline: it tells you how much the model already knows before any adaptation.
 
**Few-shot fine-tuning** — Fine-tune a pretrained model using only a handful of labeled examples (typically 5–50 images). Because the dataset is tiny, most of the model must stay frozen — only a small head or the last few layers are trained. This prevents the model from forgetting what it already knows (catastrophic forgetting) or memorizing the few examples it sees (overfitting). Few-shot is the practical reality for most domain-specific tasks where labeling is expensive.
 
**Full fine-tuning** — Load pretrained weights and train all layers on the new dataset. Every parameter is updated. This is what Exercises 05 and 06 already did: Faster R-CNN was loaded with COCO weights and trained on PhenoBench with all layers trainable; DeepLabV3 and SegFormer were loaded with pretrained weights and all parameters were updated during training. Full fine-tuning works well when you have enough labeled data (hundreds to thousands of images) to update all parameters without overfitting.
 
**Training from scratch** — No pretrained weights at all. Every parameter starts random. This is what U-Net did in Exercise 06 and PlantCNN did in Exercise 03. Requires the most data and the most training time because the model must learn everything — from what an edge looks like to what separates crop from weed.

### Bridging from Exercises 05–06
 
In Exercises 05 and 06, every time you set `pretrained: true` in `config.yaml`, you were already doing fine-tuning. Faster R-CNN loaded COCO weights and adapted to PhenoBench's 2 detection classes. DeepLabV3 and SegFormer loaded weights trained on ADE20K (150 classes) and adapted to PhenoBench's 3 segmentation classes. All layers were trainable — this is *full* fine-tuning.
 
Full fine-tuning worked because PhenoBench's training set had hundreds of images — enough data to update every parameter without destroying the pretrained knowledge. But what happens when you have only 20 labeled images? Full fine-tuning would overfit almost immediately: the model has millions of parameters and only 20 examples to learn from. The pretrained features that took thousands of images to learn would be overwritten by noise from those 20 samples.
 
This exercise introduces the constraint that makes fine-tuning practical in low-data regimes: **freeze most of the model and train only the task-specific head**. The pretrained encoder keeps its general visual knowledge intact, and only the lightweight decoder learns to map those features to your specific classes.
 
### Exercise Structure
 
The session tells one story in three parts:
 
1. **Zero-Shot (Demo).** We look at a some demos of pretrained foundation models and see how it performs on PhenoBench images "out of the box".
2. **Few-Shot Finetuning.** We use the same SegFormer model with pre-trained weights from previous exercise , freeze its entire encoder, and train *only* the segmentation head — on a tiny, few-shot subset of PhenoBench.
3. **Comparison.** Evaluate this few-shot model against your best Exercise 06 U-Net checkpoint, on the same held-out test images, and see what a pretrained foundation model buys you when labeled data is scarce.
**Estimated time:** 2–3 hours
 
**Prerequisites:** Exercise 06 (PhenoBench segmentation pipeline, the
mIoU/Dice evaluation logic reused here, and — needed for the final
comparison — your best saved Ex06 checkpoint, e.g.
`phenobench_unet_best.pt`)

**File structure:**
```
exercise_07/
├── README.md                    ← You are here
├── config.yaml                  ← Few-shot configuration
├── submit_slurm.sh              ← Slurm submission script for training
├── submit_slurm_eval.sh         ← Slurm submission script for evaluation on testset
├── src/
│   ├── dataset.py                ← TODO: seeded few-shot subset sampling
│   ├── transforms.py             ← Fully coded (identical to Ex06)
│   ├── model.py                  ← TODO: load SegFormer, replace head, freeze encoder
│   ├── train.py                  ← TODO: training loop tuned for a tiny dataset
│   ├── evaluate.py                ← TODO: mIoU/Dice (same math as Ex06)
│   ├── compare_to_ex06.py        ← Fully coded: final comparison on the test set
│   └── utils.py                  ← Fully coded (seed, checkpointing)
├── solutions/
├── optional/
│   └── partial_unfreeze.py      ← Take-home: unfreezing encoder stages (fully coded)
├── data/                          ← Symlink to shared dataset (same one as Ex06)
└── outputs/
    ├── checkpoints/
    └── logs/
```

---

## Dataset Setup

Same shared PhenoBench segmentation data as Exercise 06 — same symlink,
same layout:

```bash
ln -s /ai2_ex/data/segmentation/phenobench_256 data/phenobench_256
```

```
data/phenobench/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

`test/` is a held-out split neither Exercise 06 nor this exercise ever
trains or tunes on — it exists solely for the final comparison in
Exercise H, so that both models are judged on images neither has seen.

The difference from Exercise 06 is not the data — it's how much of it
you're allowed to use for training. Instead of the full `train/` split,
the dataloader samples a small, fixed, **seeded** subset of `data.num_shots` images. The full `val/` split is still used for monitoring during training.

> The exact few-shot budget is still being finalized with the course team.
> `config.yaml` currently sets `data.num_shots: 20` as a working default —
> update this single value once it's confirmed; nothing else needs to
> change, since every script reads it from config.

---

# Part I: Concepts

---

## 1. The Zero-Shot Gap

**Demo**
Roboflow and Hugging Face provide interactive demo environments to test the zero-shot object detection capabilities of leading models.
- [Roboflow - Best Zero-Shot Object Detection Models](https://playground.roboflow.com/models/feature/zero-shot-detection)
- [Huggingface - Zero-Shot Object Detection Models](https://huggingface.co/models?pipeline_tag=zero-shot-object-detection)


---

## 2. From Zero-Shot to Few-Shot: When Each Strategy Applies
 
### When to use what
 
| Strategy | Labeled data available | What you do | When it's suitable |
|----------|----------------------|-------------|--------------------|
| Zero-shot | None | Use pretrained model directly | Quick baseline, exploring a new domain, checking if a model is worth fine-tuning |
| Few-shot fine-tuning | 5–50 images | Freeze most layers, train only head | Labeling is expensive, domain is niche, rapid prototyping |
| Full fine-tuning | 100s–1000s of images | Train all layers from pretrained weights | Standard approach when enough labeled data exists (Ex05, Ex06) |
| Training from scratch | 1000s+ of images | Random initialization, train everything | No pretrained model exists for your modality, or data is abundant (Ex03, Ex06 U-Net) |
 
### Why the strategy must match the data budget
 
In Exercise 06, full fine-tuning worked because hundreds of training images could support updating all ~3.7M parameters (SegFormer-B0) or ~31M parameters (U-Net) without overfitting. The ratio of data to parameters was sufficient.
 
With 20 images, **full** fine-tuning (all layers trainable) would overfit almost immediately — the model has millions of parameters and only 20 examples to learn from. This is why Exercise 07 uses a different fine-tuning strategy: freeze the encoder and train only the lightweight decode head. The pretrained encoder keeps its visual knowledge intact, and the small
head (~0.3M parameters) is all that adapts — a ratio that 20 images can plausibly support.
 
| | Ex06 (full fine-tuning) | Ex07 (few-shot fine-tuning) |
|---|---|---|
| Starting weights | Pretrained (SegFormer) or random (U-Net) | Pretrained SegFormer |
| Training images | Full `train/` split (hundreds) | `data.num_shots` (default: 20) |
| What's trainable | Every parameter | Only the segmentation head |
| Trainable params | ~3.7M (SegFormer-B0) or ~31M (U-Net) | ~0.3M (decode head only) |
| Main risk | Underfitting if model is too small | Overfitting even with head-only training |
| Augmentation role | Improves generalization | Essential — only source of variety |
 
---

## 3. Anatomy of SegFormer: Encoder vs. Decode Head
> Check if this part is already mentioned in Exercise 06!

SegFormer has two parts: a hierarchical transformer **encoder**
(`model.segformer`, internally four stages of increasing abstraction and
decreasing resolution — the general-purpose visual features) and a
lightweight MLP **decode head** (`model.decode_head`) that fuses those
features into a per-pixel class map. The pretrained checkpoint's decode
head outputs ADE20K's 150 classes — it must be replaced with one that
outputs 3 (background/crop/weed) before anything else happens.

```python
from transformers import AutoModelForSemanticSegmentation

model = AutoModelForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=3,
    ignore_mismatched_sizes=True,   # old head (150 classes) doesn't fit the new one (3)
)
```

Know exactly which named parameters are the encoder and which are the head before writing any freezing logic — `model.named_parameters()` gives you every parameter's name, and every SegFormer parameter's name starts with either `segformer.` (encoder) or `decode_head.` (head).

**One more architectural detail matters before you get to evaluation:**
SegFormer's decode head deliberately outputs logits at **1/4 of the input resolution** (e.g. a 512×512 input produces `(B, num_classes, 128, 128)` logits) — this keeps the head lightweight. When you give the model `labels` during training, it upsamples internally before computing the loss, so training needs no special handling. When you call the model *without* labels (as evaluation does, to get predictions), you get the raw, low-resolution logits back and must upsample them yourself before comparing to the mask. This is different from every model in Exercise 06 (U-Net, DeepLabV3, FCN), which all output full-resolution logits directly — `evaluate.py` below handles this explicitly.

---

## 4. Freeze Everything, Train Only the Head

The core strategy for this exercise, and the direct answer to Section 2:
with only `num_shots` images, the decode head (small) can plausibly learn something; the encoder (large, general-purpose) cannot be safely retrained on so little data without destroying what it already knows or overfitting immediately.

```python
for param in model.segformer.parameters():
    param.requires_grad = False       # encoder: frozen

for param in model.decode_head.parameters():
    param.requires_grad = True        # head: the only thing that trains
```

This connects data budget to parameter budget directly: with `num_shots`
images, only a parameter count on that same rough scale can be fit
reliably. Freezing the encoder is that constraint applied in code, not an
arbitrary default.

---

## 5. Optional / Take-Home: Unfreezing More Layers

**Not required — for fast students in the session, or at home.** If you
have more labeled data or more patience than pure few-shot assumes, you can
unfreeze deeper into the encoder for better adaptation, trading overfitting
risk for capacity. SegFormer's encoder is organized into 4 hierarchical
stages, so the natural next step past "head only" is to also unfreeze the
*last* stage — closest to the output, most task-specific, least likely to
destroy the general-purpose low-level features in the earlier stages.

**The exact internal attribute names for SegFormer's four stages differ
between `transformers` versions** (this changed as recently as the 5.x
line) — so the reliable way to unfreeze "the last stage" is to inspect your
installed version first, not to copy a hardcoded path:

```python
# Step 1 -- always inspect first. Run this once and read the names:
for name, _ in model.segformer.named_parameters():
    print(name)
# Depending on your installed `transformers` version you'll see either
# "encoder.block.3...." / "encoder.layer_norm.3...." (older versions) or
# "stages.3.blocks...." / "stages.3.layer_norm...." (newer versions).
# Use whichever prefix your printout actually shows in step 2.
```

```python
# Step 2 -- unfreeze by name substring, using the prefix you just confirmed.
last_stage_prefix = "stages.3"   # <- replace with what step 1 printed
for name, param in model.segformer.named_parameters():
    if last_stage_prefix in name:
        param.requires_grad = True
```

A version-agnostic alternative that sidesteps naming entirely — unfreeze
the last `k` parameter tensors in encoder order, whatever they're called:

```python
# Version-agnostic: unfreeze roughly the last stage by position, not by name.
encoder_params = list(model.segformer.named_parameters())
num_tensors_last_stage = 20   # tune by inspecting len(encoder_params) and the printout above
for name, param in encoder_params[-num_tensors_last_stage:]:
    param.requires_grad = True
```

`optional/partial_unfreeze.py` is a complete, runnable version that detects
which naming scheme is present at runtime rather than assuming one, then
retrains on the same few-shot subset. Compare its mIoU and its train/val
loss gap against your head-only run from Exercise E.

---

## 6. Training on a Tiny Dataset: What Changes

With only `num_shots` images, a few things shift relative to Ex06's
full-data training loop:

- **More epochs, not more data.** One epoch over 20 images is a handful of
  gradient steps — expect to need many more epochs than Ex06 to see the
  same number of total updates.
- **Watch the train/val gap closely.** Even with the encoder frozen, a
  handful of images can be memorized. Overfitting is the default outcome to
  expect, not an edge case.
- **Augmentation matters more, not less.** The same `num_shots` images get
  reused every epoch — flips and color jitter (from Ex06's `transforms.py`,
  reused unchanged) are one of the only sources of variety the head ever
  sees.

---

## 7. The Final Comparison: Few-Shot Foundation Model vs. Full-Data U-Net

Ex06's U-Net saw the entire training set and trained until convergence.
This exercise's SegFormer saw `num_shots` images and only trained its head.
The fair way to compare them is the same held-out images neither model was
trained or tuned on: `data/phenobench/test/`.

`src/compare_to_ex06.py` runs both checkpoints against the test set and
prints mIoU, per-class IoU, trainable-parameter count, and wall-clock eval
time side by side — see Exercise H. That table is this exercise's actual
payoff: an evidence-based answer to "is a few-shot-finetuned foundation
model competitive with a fully-trained from-scratch model, and at what
fraction of the data and compute cost?" That tradeoff, not architecture
trivia, is what a final project actually has to reason about when deciding
whether to finetune.

---

# Part II: Exercises

Work through each script in order. Open the file, look for `# TODO`
markers, write your solution, and run the script to verify.

---

### Exercise A — Load, freeze, and summarize the model (Sections 3–4)

**→ Open `src/model.py` and complete the TODOs.**

Covers: loading the pretrained SegFormer checkpoint with a replaced
3-class head (`load_pretrained_segformer`), freezing the encoder while
keeping the head trainable (`freeze_encoder`), and reporting
trainable-vs-total parameter counts split by encoder/head
(`parameter_summary`).

Verify:
```bash
python src/model.py            # everything trainable (default)
python src/model.py --freeze   # only the head should be trainable
```
Both runs also print the model's raw output shape, so you can see the 1/4
resolution logits from Section 3 for yourself.

---

### Exercise B — Few-shot dataset sampling (Section 2, Dataset Setup)

**→ Open `src/dataset.py` and complete the TODOs.**

Covers: loading PhenoBench images/masks the same way as Ex06, then
restricting the training split to a seeded `data.num_shots` sample. Mask
loading/resizing rules are unchanged from Ex06 (nearest-neighbor only,
never bilinear) — `transforms.py` is reused as-is.

Verify: `python src/dataset.py` — prints exactly `num_shots` training
images, the full val set size, and confirms the same seed reproduces the
same subset every run.

---

### Exercise C — Training loop for a tiny dataset (Section 6)

**→ Open `src/train.py` and complete the TODOs.**

Requires: Exercises A and B completed (imports from both).

Covers: building the few-shot train/val dataloaders, applying
`freeze_encoder`, building an optimizer over only the parameters with
`requires_grad=True`, the training step (SegFormer returns a loss directly
from `model(pixel_values=images, labels=masks)` — unlike Ex06 there is no
separate `criterion` call), TensorBoard logging, and best-checkpoint
saving on validation mIoU.

---

### Exercise D — Utility functions

`src/utils.py` is fully coded — no TODOs. Read through it to understand:

- `set_seed`: reproducibility.
- `save_checkpoint` / `load_checkpoint`: model state persistence.
- `colorize_mask`: converts a (H, W) class mask to an RGB image for visualization.
- `overlay_mask`: blends a colorized mask onto its (normalized) image for inspection.
- `denormalize`: reverses ImageNet normalization for display.

---

### Exercise E — Segmentation evaluation (section 5)

**→ Open `src/evaluate.py` and complete the TODOs.**

Covers: building a confusion matrix from predictions and targets, computing pixel accuracy / per-class IoU / mIoU / per-class Dice / mean Dice from the confusion matrix, accumulating over all validation batches. Handles torchvision dict output (`output["out"]`).

This script is imported by `train.py` — complete it before running training.

It can also be run standalone after training to evaluate on the test set (see Exercise H).

---

### Exercise F — Run few-shot training

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

### Exercise H — Evaluate on test set

After training completes, evaluate the best checkpoint on the held-out test set.

Edit `slurm_submit_eval.sh` if needed — update the checkpoint path to match your best saved model. Then submit:

```bash
sbatch slurm_submit_eval.sh
```

Check the results:

```bash
cat outputs/logs/eval_*.out
```

This prints the segmentation evaluation metrics. By default it evaluates on the **test** split. To evaluate on val instead, edit the last line in `slurm_submit_eval.sh`:

```bash
python src/evaluate.py --config config.yaml \
    --checkpoint outputs/checkpoints/phenobench_segformer_fewshot_best.pt \
    --split val
```

---

### Exercise I — TensorBoard

```bash
tensorboard --logdir outputs/logs --port 6006 --bind_all
```

Watch the train/val loss gap in particular (Section 6) — with `num_shots`
this small, a widening gap should appear early if it's going to appear at
all.

---

### Exercise J — Optional / take-home: partial unfreezing (Section 5)

**→ Read (and optionally run) `optional/partial_unfreeze.py`.**

Not required. For students who finish early in the session, or as a
take-home extension:

```bash
python optional/partial_unfreeze.py --config config.yaml
```

Unfreezes the last encoder stage in addition to the head, retrains on the
same few-shot subset, and prints a best mIoU to compare against Exercise E.

---

### Exercise K — Final comparison vs. Exercise 06 (Section 7)

**→ Run `src/compare_to_ex06.py`** — fully coded, nothing to implement.

```bash
python src/compare_to_ex06.py \
    --config config.yaml \
    --segformer_checkpoint outputs/checkpoints/phenobench_segformer_fewshot_best.pt \
    --ex06_src ../exercise_06/src \
    --ex06_config ../exercise_06/config.yaml \
    --ex06_checkpoint ../exercise_06/outputs/checkpoints/phenobench_unet_best.pt
```

Prints a side-by-side table on `data/phenobench/test/`: mIoU, pixel
accuracy, per-class IoU, trainable-parameter count, training images used,
and evaluation wall-time, for your Ex06 U-Net and this exercise's few-shot
SegFormer.

---

### Exercise L — Push to GitHub

```bash
git add src/
git commit -m "Complete exercise 07: few-shot finetuning for segmentation"
git push
```

---

## Summary

| Component | File | What's new vs. Ex06 |
|-----------|------|--------------------------|
| Motivation | Live  demo (Section 1) | A visible zero-shot failure motivates finetuning |
| Model loading | `src/model.py` | Load SegFormer via HuggingFace `transformers`, replace the 150-class head with a 3-class one |
| Freezing | `src/model.py` | Explicit encoder/head split; only the head trains by default |
| Data | `src/dataset.py` | Seeded `num_shots`-image subset instead of the full training split |
| Training | `src/train.py` | Same loop shape as Ex06, tuned for a tiny dataset (more epochs, close overfitting watch, optimizer built over trainable params only) |
| Evaluation | `src/evaluate.py` | Same mIoU/Dice math as Ex06, plus upsampling SegFormer's 1/4-resolution logits before scoring |
| Comparison | `src/compare_to_ex06.py` | Few-shot foundation model vs. full-data from-scratch model, evaluated on the same held-out test set |
