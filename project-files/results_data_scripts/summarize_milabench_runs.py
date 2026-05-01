import json
import os
import sys

GROUPS = ["rtxa2000", "v100"]
RUN_IDS = [1, 2]
FILENAME = "regnet_y_128gf.D0.metrics.json"

METRIC_PREFIXES = [
    "epoch_time",
    "epoch_drop_first_time",
    "batch_time",
    "batch_drop_first_time",
    "gpu_load",
    "gpu_mem",
    "cpu_load",
    "read_bytes",
    "write_bytes",
    "items_per_second",
]

SUFFIXES = ["avg", "std", "variance"]

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def mean(values):
    return sum(values) / len(values)

def summarize_group(runs_dir: str, group: str) -> dict:
    values_by_key = {}

    for i in RUN_IDS:
        run_folder = f"{group}_{i}"
        path = os.path.join(runs_dir, run_folder, FILENAME)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing metrics file: {path}")

        d = load_json(path)

        for prefix in METRIC_PREFIXES:
            for suf in SUFFIXES:
                key = f"{prefix}_{suf}"
                if key not in d:
                    raise KeyError(f"Missing key '{key}' in {path}")
                values_by_key.setdefault(key, []).append(float(d[key]))

    out = {}

    for key, vals in values_by_key.items():
        out[key] = round(mean(vals), 2)

    return out

def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    runs_dir = sys.argv[1]

    if not os.path.isdir(runs_dir):
        print(f"Error: runs_dir is not a directory: {runs_dir}")
        sys.exit(1)

    for group in GROUPS:
        summary = summarize_group(runs_dir, group)
        out_name = f"{group}_summary.metrics.json"
        with open(out_name, "w") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()

# Command: python summarize_milabench_runs.py [path to milabench_base]/runs