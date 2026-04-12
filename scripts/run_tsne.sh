#!/bin/bash
#SBATCH --job-name=hw3_tsne
#SBATCH --output=/cta/users/mert.coskuner/CS515-HW3-mertcoskuner/scripts/logs/%j_tsne.out
#SBATCH --error=/cta/users/mert.coskuner/CS515-HW3-mertcoskuner/scripts/logs/%j_tsne.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1

source /cta/capps/conda/ana/20250420/etc/profile.d/conda.sh
conda activate openfgl

cd /cta/users/mert.coskuner/CS515-HW3-mertcoskuner

export PYTHONPATH=/cta/users/mert.coskuner/CS515-HW3-mertcoskuner:$PYTHONPATH

mkdir -p plots/partc

python -u tsne_only.py \
    --data_dir     ./data \
    --batch_size   128 \
    --num_workers  8 \
    --n_samples    2000 \
    --seed         42

echo "Done: $(date)"
