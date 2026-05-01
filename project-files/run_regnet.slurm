#!/bin/bash
#SBATCH --job-name=run_regnet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=<gpu_type>.out

source ~/.bashrc
conda activate <milabench conda environment name>
source env.sh
cd [path to milabench directory]

for i in 1 2;
do
    python -m milabench run --base "$MILABENCH_BASE" --config "$MILABENCH_CONFIG"
done

# Command: sbatch run_regnet.slurm