#!/bin/bash
#SBATCH --job-name=hw2_parta
#SBATCH --output=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner/scripts/logs/%j_parta.out
#SBATCH --error=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner/scripts/logs/%j_parta.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn07

source /cta/capps/conda/ana/20250420/etc/profile.d/conda.sh
conda activate openfgl

cd /cta/users/mert.coskuner/CS515-HW2-mertcoskuner

export PYTHONPATH=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner:$PYTHONPATH

mkdir -p scripts/logs
mkdir -p results/parta

# Verify GPU is available and accessible to PyTorch before starting
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: PyTorch cannot access GPU on $(hostname), exiting."
    exit 1
fi

echo "======================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Host    : $(hostname)"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date    : $(date)"
echo "======================================"

python -u main.py \
    --hw_part       PART_A \
    --model         PartA \
    --dataset       Cifar10 \
    --epochs        20 \
    --lr            1e-3 \
    --weight_decay  1e-4 \
    --batch_size    128 \
    --num_workers   4 \
    --log_interval  100 \
    --patience      10 \
    --seed          42 \
    --data_dir      ./data \
    --save_path     results/parta/best_model.pth

echo "Done: $(date)"
