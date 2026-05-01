# COMP400 Project: Porting a Milabench Workload to MLPerf Storage

This repository contains the code and configuration files used for my COMP 400 project on modeling a Milabench training workload in MLPerf Storage.

The project uses the `regnet_y_128gf` workload from Milabench, measures its training behavior on real GPU systems, then creates equivalent MLPerf Storage workload configurations to simulate the same workload. The final comparison checks whether MLPerf Storage can reproduce the real Milabench training behavior using the measured workload parameters.

## Repository Layout

COMP400-project\
├── milabench/\
├── storage/\
└── project-files/

### milabench/

This folder contains the Milabench files that were changed or added for this project.

The folder structure is intentionally kept the same as the original Milabench repository. This makes it easier to see where each file should go if copying the project files back into a full Milabench checkout.

Only the relevant changed or added files are included here. A new user still needs to clone and install the original Milabench repository separately.

Changed sections in Milabench source files are marked with comments such as:

`##### --- CHANGES ---`

### storage/

This folder contains the MLPerf Storage files that were changed or added for this project.

Like the Milabench folder, this follows the original MLPerf Storage directory structure. The files here should be copied into the matching locations of a full MLPerf Storage checkout.

Only the relevant changed or added files are included. A new user still needs to clone and install the original MLPerf Storage repository separately.

Changed sections are marked with comments such as:

`##### --- CHANGES ---`

### project-files/

This folder contains the project-specific scripts used to run experiments, parse logs, summarize results, and generate plots.

Important files include:

project-files/\
├── env.sh\
├── run_regnet.sh\
├── run_regnet.slurm\
├── model_stats/\
├── plot_files/\
└── results_data_scripts/\

- env.sh stores environment variables used by the Milabench scripts.
- run_regnet.sh runs the Milabench RegNetY-128GF experiment directly.
- run_regnet.slurm runs the Milabench RegNetY-128GF experiment through SLURM.
- results_data_scripts/ contains scripts for parsing and summarizing Milabench and MLPerf Storage outputs.
- plot_files/ contains scripts for generating plots from parsed result files.
- model_stats/ contains scripts used to inspect model and dataset properties.

## Prerequisites

This project assumes access to:

- Python 3.10
- Conda
- Git

## Usage

The exact machine paths are intentionally left as placeholders in the scripts. Update them for your own environment before running.

### 1. Clone This Repository

```
git clone https://github.com/Ge0888/COMP400-project.git
cd COMP400-project
```

### 2. Set Up Milabench

Clone the original Milabench repository separately:

```
git clone https://github.com/mila-iqia/milabench.git
cd milabench
```

Create and activate a Conda environment:

```
conda create -n milabench-env python=3.10
conda activate milabench-env
```

Install Milabench in editable mode:

`pip install -e .`

Copy the project’s changed Milabench files into the matching locations of the full Milabench repository.

For example:

`COMP400-project/milabench/config/regnet128.yaml → <milabench-repo>/config/regnet128.yaml`

and:

`COMP400-project/milabench/benchmarks/torchvision/main.py → <milabench-repo>/benchmarks/torchvision/main.py`

The copied files preserve the same directory structure as the original Milabench repository.

### 3. Configure the Milabench Environment

Open:

`project-files/env.sh`

Update the placeholder values:

```
export CONDA_ENV_NAME=<milabench conda environment name>
export MILABENCH_DIR=<path to milabench directory>
export MILABENCH_BASE=<path to milabench base directory>
export MILABENCH_CONFIG=config/regnet128.yaml
export MILABENCH_SELECT=regnet_y_128gf
export MILABENCH_GPU_ARCH=cuda
export CUDA_VISIBLE_DEVICES=0
```

Then source the file:

`source project-files/env.sh`

### 4. Prepare the Milabench Dataset

From inside the Milabench repository, run:

`milabench prepare --base "$MILABENCH_BASE" --config "$MILABENCH_CONFIG"`

This prepares the dataset needed for the RegNetY-128GF workload.

### 5. Run the Milabench Experiment

There are two ways to run the experiment.

#### Option A: Direct Shell Run

From the project-files/ folder:

`bash run_regnet.sh`

To keep the run alive after disconnecting:

`nohup bash run_regnet.sh > <gpu_type>.log 2>&1 &`

#### Option B: SLURM Run

If using a SLURM-managed machine:

`sbatch run_regnet.slurm`

The Milabench run produces output files under the configured Milabench base directory. These outputs are later parsed into clean metric files.

### 6. Parse Milabench Output Data

After the Milabench runs finish, parse each run’s raw output file.

From `project-files/results_data_scripts/`, run:

`python parse_milabench_data.py [path to milabench_base]/runs/<run_id>/regnet_y_128gf.D0.data`

This creates a parsed metrics JSON file for that run.

The parser extracts metrics such as:

```
epoch time
epoch time after dropping the first epoch
batch time
batch time after dropping the first epoch
items per second
GPU load
GPU memory
CPU load
read bytes
write bytes
```
### 7. Summarize Milabench Runs

After parsing the individual Milabench runs, summarize repeated runs:

`python summarize_runs.py <path to milabench runs folder>`

This creates summary JSON files for each hardware group.

These summary files are used for the final comparison with MLPerf Storage.

### 8. Generate Milabench Plots

The plotting scripts are in:

`project-files/plot_files/`

You must change the names of the run files to use these scripts, it has to match the following naming convention:

`rtxa2000_<run_number>` or `v100_<run number>`

Then you can generate the plots:

```
python plot_batch_timelines.py [path to milabench_base]/runs
python plot_batch_drop_first_timelines.py [path to milabench_base]/runs
python plot_items_per_second_timelines.py [path to milabench_base]/runs
```

These scripts generate timeline plots for the Milabench runs.

### 9. Get Dataset and Model Statistics

The scripts in `project-files/model_stats/` were used to collect information needed for the MLPerf Storage configuration.

`python train_stats.py`

This reports dataset statistics such as the number of files and average file size.

`python regnet_model_size.py`

This reports the size of the RegNetY-128GF model.

These values help configure the MLPerf Storage workload YAML files.

### 9.5 Create your own .yaml Workload files (Optional)

Using the `avg_batch_time_drop` value from the Milabench runs, and the Dataset and Model statistics collected, you have the required information to create a `.yaml` configuration file.

Following the structure of the other configuration files, fill in the data in the labeled locations.

### 10. Set Up MLPerf Storage

Clone the original MLPerf Storage repository separately:

```
git clone https://github.com/mlcommons/storage
cd storage
```

Create and activate a Conda environment:

```
conda create -n mlpstorage python=3.10
conda activate mlpstorage
```

Install MLPerf Storage:

`pip install -e .`

Copy the changed files from this repository’s `storage/` folder into the matching locations of the full MLPerf Storage checkout.

The `storage/` folder in this repository preserves the MLPerf Storage directory structure, so files should be copied to the same relative paths.

### 11. Generate the MLPerf Storage Dataset

After copying the RegNetY-128GF MLPerf Storage workload files, generate the synthetic dataset:

```
mlpstorage datagen \
  --model regnet_y_128gf \
  --data-dir [path to mlperf data directory]
```

Use the dataset location again when running the MLPerf Storage training simulation.

### 12. Run MLPerf Storage

Run the MLPerf Storage simulation using the RegNetY-128GF workload.

Example command:

```
mlpstorage training run \
  --hosts 127.0.0.1 \
  --num-client-hosts 1 \
  --client-host-memory-in-gb 16G \
  --num-accelerators 1 \
  --accelerator-type <accelerator type> \
  --model regnet_y_128gf \
  --data-dir [path to mlperf data directory]/[path to generated dataset]
```

Replace:

`<accelerator type>`

with the values for the system being simulated.

The project used separate MLPerf Storage configurations for the hardware profiles being compared.

### 13. Parse and Summarize MLPerf Storage Results

After the MLPerf Storage run finishes, summarize the output folder:

`python summarize_mlperf_run.py [path to mlperf storage run results folder]/<run_name>`

This script reads the MLPerf Storage output files and creates a compact summary file in the same folder.

The summary includes:

```
number of epochs
mean epoch duration
epoch duration standard deviation
mean epoch duration after dropping the first epoch
drop-first epoch duration standard deviation
accelerator utilization
throughput
I/O throughput
```

This summary is used to compare the MLPerf Storage simulation against the real Milabench runs.

### 14. Compare Milabench and MLPerf Storage

The final comparison uses:

- Milabench summary JSON files
- MLPerf Storage summary JSON files

The main comparison metrics are:

- epoch time
- drop-first epoch time
- items per second / throughput
- percentage error between Milabench and MLPerf Storage

The goal is to check whether MLPerf Storage can reproduce the real Milabench workload behavior after being configured using the measured Milabench results.

## Notes

- Large generated datasets and raw benchmark output folders are not meant to be committed to this repository.

- This repository focuses on the code, scripts, configuration files, and analysis tools needed to reproduce the experiment.

- Machine-specific paths are left as placeholders and should be updated before running.

- The Milabench and MLPerf Storage folders do not contain full copies of the original upstream repositories. They only contain the files that were changed or added for this project.