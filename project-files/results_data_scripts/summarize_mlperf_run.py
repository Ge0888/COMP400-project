import argparse
import json
import statistics
from pathlib import Path

OUTPUT_FILENAME = "summary.metrics.json"

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def find_one_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching pattern '{pattern}' found in: {run_dir}")

    if len(matches) > 1:
        names = "\n".join(str(p.name) for p in matches)
        raise ValueError(
            f"Multiple files matching pattern '{pattern}' found in: {run_dir}\n"
            f"Please clean the folder or make the pattern more specific.\n"
            f"Matches:\n{names}"
        )
    return matches[0]

def mean(values):
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return statistics.mean(values)

def stdev_population(values):
    if not values:
        raise ValueError("Cannot compute standard deviation of empty list")
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values)

def parse_epoch_durations(per_epoch_stats: dict) -> list[float]:
    durations = []
    for epoch_key in sorted(per_epoch_stats.keys(), key=lambda x: int(x)):
        epoch = per_epoch_stats[epoch_key]
        if "duration" not in epoch:
            raise KeyError(f"Missing duration for epoch {epoch_key}")
        durations.append(float(epoch["duration"]))
    if not durations:
        raise ValueError("No epoch durations found")
    return durations


def extract_metric(summary_data: dict, key: str) -> float:
    metric = summary_data.get("metric")
    if not isinstance(metric, dict):
        raise KeyError("Missing or invalid 'metric' section in MLPerf summary file")
    if key not in metric:
        raise KeyError(f"Missing metric key: {key}")
    return float(metric[key])


def summarize_mlperf_run(run_dir: Path) -> dict:
    per_epoch_path = find_one_file(run_dir, "*per_epoch_stats*.json")
    summary_path = find_one_file(run_dir, "summary*.json")
    per_epoch_stats = load_json(per_epoch_path)
    summary_data = load_json(summary_path)
    epoch_durations = parse_epoch_durations(per_epoch_stats)
    epoch_durations_drop_first = epoch_durations[1:]

    if not epoch_durations_drop_first:
        raise ValueError("Need at least 2 epochs to compute drop-first-epoch metrics")

    output = {
        "epochs": len(epoch_durations),
        "epoch_duration_mean": mean(epoch_durations),
        "epoch_duration_stdev": stdev_population(epoch_durations),
        "epoch_duration_mean_drop_first": mean(epoch_durations_drop_first),
        "epoch_duration_stdev_drop_first": stdev_population(epoch_durations_drop_first),
        "train_au_mean_percentage": extract_metric(
            summary_data,
            "train_au_mean_percentage"
        ),
        "train_au_stdev_percentage": extract_metric(
            summary_data,
            "train_au_stdev_percentage"
        ),
        "train_throughput_mean_samples_per_second": extract_metric(
            summary_data,
            "train_throughput_mean_samples_per_second"
        ),
        "train_throughput_stdev_samples_per_second": extract_metric(
            summary_data,
            "train_throughput_stdev_samples_per_second"
        ),
        "train_io_mean_MB_per_second": extract_metric(
            summary_data,
            "train_io_mean_MB_per_second"
        ),
        "train_io_stdev_MB_per_second": extract_metric(
            summary_data,
            "train_io_stdev_MB_per_second"
        ),
    }
    return output

def main():
    parser = argparse.ArgumentParser(
        description="Parse and summarize one MLPerf Storage training run."
    )

    parser.add_argument(
        "run_dir",
        help="Path to the MLPerf Storage run output folder containing per-epoch stats and summary JSON files."
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run_dir is not a directory: {run_dir}")
    summary = summarize_mlperf_run(run_dir)
    out_path = run_dir / OUTPUT_FILENAME
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Wrote summary to: {out_path}")

if __name__ == "__main__":
    main()

# Command:
# python summarize_mlperf_run.py [path to mlperf storage run results folder]/<run_name>