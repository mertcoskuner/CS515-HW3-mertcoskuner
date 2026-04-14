# CS515 Homework 3 — Build On HW2 Transfer Learning & Knowledge Distillation on CIFAR-10

## Overview
## Repository Structure

```
CS515-HW2-mertcoskuner/
├── main.py                  # Entry point — runs Part A or Part B
├── train.py                 # Training loops (standard, KD, modified KD)
├── test.py                  # Evaluation on CIFAR-10 test set
├── helper.py                # Model builders and FLOPs counter
├── parameters.py            # Argument parser
├── visualize.py             # Plot generation (training curves, confusion matrices)
├── requirements.txt
│
├── models/
│   ├── CNN.py               # SimpleCNN
│   ├── ResNet.py            # ResNet-18 (from scratch, CIFAR-10 adapted)
│   ├── VGG.py               # VGG-11/13/16/19
│   └── mobilenet.py         # MobileNetV2
│
├── params/
│   ├── data_params.py       # Dataset configuration
│   ├── model_params.py      # Per-model parameter dataclasses
│   └── model_training_params.py  # Training hyperparameters
│
├── scripts/
│   ├── run_parta.sh         # SLURM job for Part A
│   ├── run_partb.sh         # SLURM job for Part B
│   ├── cnn_visualize.sh     # SLURM array job for visualizations
│   ├── cnn_visualize_tasks.txt
│   └── logs/
│
├── data/                    # CIFAR-10 dataset (auto-downloaded)
├── results/
│   ├── parta/               # Saved checkpoints and JSON results for Part A
│   └── partb/               # Saved checkpoints and JSON results for Part B
└── plots/
    ├── parta/               # Generated figures for Part A
    └── partb/               # Generated figures for Part B
```

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `torch>=2.0.0`, `torchvision>=0.15.0`, `numpy>=1.24.0`, `ptflops>=0.7`, `matplotlib`, `seaborn`, `scikit-learn`

## Usage

### Part A — Transfer Learning

Trains pretrained ResNet-18 on CIFAR-10 using both adaptation options:

```bash
python main.py \
    --hw_part PART_A \
    --model   PartA \
    --dataset Cifar10 \
    --epochs  20 \
    --lr      1e-3 \
    --batch_size 128 \
    --data_dir   ./data \
    --save_path  results/parta/best_model.pth
```

Two options are run automatically:
- **resize** — Upsample images to 224×224, freeze all layers except FC.
- **modify** — Replace 7×7 stem with 3×3 conv, remove maxpool, fine-tune all layers.

### Part B — Knowledge Distillation

Trains all five model configurations sequentially:

```bash
python main.py \
    --hw_part         PART_B \
    --dataset         Cifar10 \
    --epochs          50 \
    --lr              1e-3 \
    --batch_size      128 \
    --label_smoothing 0.1 \
    --kd_temperature  4.0 \
    --kd_alpha        0.5 \
    --data_dir        ./data \
    --save_path       results/partb/best_model.pth
```

Models trained in order:
1. SimpleCNN (baseline, no label smoothing)
2. ResNet-18 (no label smoothing)
3. ResNet-18 (label smoothing ε=0.1)
4. SimpleCNN with Hinton KD (T=4, α=0.5), teacher = best ResNet
5. MobileNetV2 with modified KD (teacher-probability soft labels)

### Visualization

Generate all plots (training curves, accuracy bars, FLOPs charts, confusion matrices):

```bash
python visualize.py \
    --hw_part     ALL \
    --results_dir results \
    --plots_dir   plots \
    --data_dir    ./data
```

To skip confusion matrices (no GPU needed):

```bash
python visualize.py --hw_part ALL --no_confusion
```

### Test Only

```bash
python main.py --hw_part PART_A --mode test --save_path results/parta/best_model.pth
python main.py --hw_part PART_B --mode test --save_path results/partb/best_model.pth
```

## SLURM

```bash
sbatch scripts/run_parta.sh        # Part A training
sbatch scripts/run_partb.sh        # Part B training
sbatch scripts/cnn_visualize.sh    # Visualization (array job)
```

## Results Summary

### Part A

| Option | Trainable Layers | Test Accuracy |
|---|---|---|
| Resize + Freeze | FC only | 79.79% |
| Modify + Fine-tune | All layers | **93.93%** |

### Part B

| Model | Method | Test Accuracy | MACs |
|---|---|---|---|
| SimpleCNN | Baseline CE | 74.32% | 6.3M |
| ResNet-18 | CE (no LS) | 92.38% | 557M |
| ResNet-18 | CE + Label Smoothing | **92.71%** | 557M |
| SimpleCNN | Hinton KD | 72.25% | 6.3M |
| MobileNetV2 | Modified KD | 89.68% | 96M |
