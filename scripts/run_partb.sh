#!/bin/bash
#SBATCH --job-name=hw2_partb
#SBATCH --output=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner/scripts/logs/%j_partb.out
#SBATCH --error=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner/scripts/logs/%j_partb.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1

source /cta/capps/conda/ana/20250420/etc/profile.d/conda.sh
conda activate openfgl

cd /cta/users/mert.coskuner/CS515-HW2-mertcoskuner

export PYTHONPATH=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner:$PYTHONPATH

mkdir -p scripts/logs
mkdir -p results/partb

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
    --hw_part         PART_B \
    --dataset         Cifar10 \
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
    --seed            42 \
    --data_dir        ./data \
    --save_path       results/partb/best_model.pth

echo "Done: $(date)"
