#!/bin/bash
#SBATCH --job-name=hw3_partc
#SBATCH --output=/cta/users/mert.coskuner/CS515-HW3-mertcoskuner/scripts/logs/%j_partc.out
#SBATCH --error=/cta/users/mert.coskuner/CS515-HW3-mertcoskuner/scripts/logs/%j_partc.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1

source /cta/capps/conda/ana/20250420/etc/profile.d/conda.sh
conda activate openfgl

cd /cta/users/mert.coskuner/CS515-HW3-mertcoskuner

export PYTHONPATH=/cta/users/mert.coskuner/CS515-HW3-mertcoskuner:$PYTHONPATH

mkdir -p scripts/logs
mkdir -p results/partc
mkdir -p plots/partc

# Verify GPU is available and accessible to PyTorch before starting
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: PyTorch cannot access GPU on $(hostname), exiting."
    exit 1
fi

LOG_FILE="scripts/logs/${SLURM_JOB_ID}_partc_run.log"

echo "======================================" | tee -a "$LOG_FILE"
echo "Job ID  : ${SLURM_JOB_ID}"             | tee -a "$LOG_FILE"
echo "Host    : $(hostname)"                  | tee -a "$LOG_FILE"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "$LOG_FILE"
echo "Date    : $(date)"                      | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

python -u main.py \
    --hw_part         PART_C \
    --mode            both \
    --dataset         Cifar10 \
    --model           PartA \
    --epochs          50 \
    --lr              1e-3 \
    --weight_decay    1e-4 \
    --batch_size      128 \
    --num_workers     4 \
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

echo "Done: $(date)" | tee -a "$LOG_FILE"
