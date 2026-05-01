#!/bin/bash

source ~/.bashrc
conda activate <milabench conda environment name>
source env.sh
cd [path to milabench directory]

for i in 1 2;
do
    python -m milabench run --base "$MILABENCH_BASE" --config "$MILABENCH_CONFIG"
done

# Command: nohup bash run_regnet.sh > <gpu_type>.log 2>&1 &