# CS515 Homework 3 — Build On HW2 Transfer Learning & Knowledge Distillation on CIFAR-10

## Overview


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
### Part  C —HW3

```bash
python -u main.py \
    --hw_part         PART_C \
    --mode            both \
    --dataset         Cifar10 \
    --model           PartA \
    --epochs          15 \
    --lr              1e-3 \
    --weight_decay    1e-4 \
    --batch_size      128 \
    --num_workers     8 \
    --log_interval    100 \
    --patience        10 \
    --label_smoothing 0.1 \
    --kd_temperature  4.0 \
    --kd_alpha        0.5 \
    --augmix_severity 3 \
    --augmix_width    3 \
    --augmix_depth    -1 \
    --lambda_jsd      12.0 \
    --seed            42 \
    --data_dir        ./data \
    --cifar10c_dir    ./data/CIFAR-10-C \
    --save_path       results/partc/best_model.pth \
    --parta_save_path results/parta/best_model.pth \
    --partb_save_path results/partb/best_model.pth \
    2>&1 | tee -a "$LOG_FILE"
Models trained in order:
1. SimpleCNN (baseline, no label smoothing)
2. ResNet-18 (no label smoothing)
3. ResNet-18 (label smoothing ε=0.1)
4. SimpleCNN with Hinton KD (T=4, α=0.5), teacher = best ResNet
5. MobileNetV2 with modified KD (teacher-probability soft labels)
```


