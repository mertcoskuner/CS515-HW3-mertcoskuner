#!/bin/bash
#SBATCH --job-name=cnn_hw2_viz
#SBATCH --output=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner/scripts/logs/%A_%a.out
#SBATCH --error=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner/scripts/logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1
#SBATCH --array=1-2%2

source /cta/capps/conda/ana/20250420/etc/profile.d/conda.sh
conda activate openfgl

cd /cta/users/mert.coskuner/CS515-HW2-mertcoskuner

export PYTHONPATH=/cta/users/mert.coskuner/CS515-HW2-mertcoskuner:$PYTHONPATH

mkdir -p /cta/users/mert.coskuner/CS515-HW2-mertcoskuner/scripts/logs
mkdir -p /cta/users/mert.coskuner/CS515-HW2-mertcoskuner/plots/parta
mkdir -p /cta/users/mert.coskuner/CS515-HW2-mertcoskuner/plots/partb

# Filter out comment (#) and empty lines, then pick the task line by array index.
ARGS=$(grep -v "^#" scripts/cnn_visualize_tasks.txt | grep -v "^[[:space:]]*$" | sed -n "${SLURM_ARRAY_TASK_ID}p")

echo "======================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Array Task : ${SLURM_ARRAY_TASK_ID}"
echo "Args       : ${ARGS}"
echo "Host       : $(hostname)"
echo "Date       : $(date)"
echo "======================================"

python -u visualize.py $ARGS

echo "Done: $(date)"
