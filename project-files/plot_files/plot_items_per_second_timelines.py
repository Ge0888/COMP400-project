import json
import os
import sys

import matplotlib.pyplot as plt

RUN_IDS = [1, 2]
FILENAME = "regnet_y_128gf.D0.metrics.json"
TIMELINE_KEY = "items_per_second_timeline"

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def get_run_timeline(runs_dir: str, group: str, run_id: int):
    run_folder = f"{group}_{run_id}"
    path = os.path.join(runs_dir, run_folder, FILENAME)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")

    d = load_json(path)

    if TIMELINE_KEY not in d:
        raise KeyError(f"Missing key '{TIMELINE_KEY}' in {path}")

    timeline = d[TIMELINE_KEY]
    if not isinstance(timeline, list):
        raise TypeError(f"Key '{TIMELINE_KEY}' is not a list in {path}")

    return [float(x) for x in timeline]

def plot_group(runs_dir: str, group: str, out_png: str):
    plt.figure()
    for run_id in RUN_IDS:
        y = get_run_timeline(runs_dir, group, run_id)
        x = list(range(1, len(y) + 1))
        plt.plot(x, y, label=f"{group}_{run_id}")

    plt.title(f"{group}: {TIMELINE_KEY}")
    plt.xlabel("Index")
    plt.ylabel("Items per second")
    plt.legend()
    ymin, ymax = plt.ylim()
    padding = (ymax - ymin)
    plt.ylim(0, ymax + padding * 0.1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    runs_dir = sys.argv[1]

    if not os.path.isdir(runs_dir):
        print(f"Error: runs_dir is not a directory: {runs_dir}")
        sys.exit(1)

    plot_group(runs_dir, "rtxa2000", "rtxa2000_items_per_second.png")
    plot_group(runs_dir, "v100", "v100_items_per_second.png")

if __name__ == "__main__":
    main()

# Command: python plot_items_per_second_timelines.py [path to milabench_base]/runs