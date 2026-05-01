import json
import os
import sys
import re

import numpy as np

def basic_stats(arr):
    a = np.asarray(arr, dtype=float)
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "variance": float(np.var(a)),
    }

def percentiles(arr):
    a = np.asarray(arr, dtype=float)
    ps = range(10, 100, 10)
    vals = np.percentile(a, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}

def parse_file(path: str):
    epoch_time_re = re.compile(r"^epoch_time:\s*epoch=(\d+)\s+ms=([0-9]*\.?[0-9]+)")
    batch_time_re = re.compile(r"^batch_time:\s*epoch=(\d+)\s+batch=(\d+)\s+ms=([0-9]*\.?[0-9]+)")

    epoch_time = []
    batches_per_epoch = 0
    batch_time = []

    gpu_load = []
    gpu_mem = []
    gpu_time = []

    cpu_load = []
    read_bytes = []
    write_bytes = []
    process_time = []

    items_per_second = []
    rate_time = []

    start_time = None
    end_time = None

    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            event = obj.get("event")
            data = obj.get("data", {})

            # start / end
            if event == "start":
                if isinstance(data, dict) and "time" in data:
                    start_time = data.get("time")
                continue

            if event == "end":
                if isinstance(data, dict) and "time" in data:
                    end_time = data.get("time")
                continue

            if event == "line":
                if isinstance(data, str):
                    m = batch_time_re.match(data.strip())
                    n = epoch_time_re.match(data.strip())
                    if m:
                        epoch = int(m.group(1))
                        batch = int(m.group(2))
                        time = float(m.group(3))
                        if batch > batches_per_epoch:
                            batches_per_epoch = batch
                        batch_time.append(time)
                    if n:
                        epoch = int(n.group(1))
                        time = float(n.group(2))
                        epoch_time.append(time)
                continue

            if event != "data":
                continue

            t = data.get("time")

            # GPU
            if "gpudata" in data:
                # Only GPU "0"
                gpu = data["gpudata"]["0"]
                if "load" in gpu and gpu["load"] is not None:
                    gpu_load.append(float(gpu["load"]))
                    gpu_time.append(float(t))
                if "memory" in gpu and gpu["memory"] is not None:
                    # memory = [used, total]
                    gpu_mem.append(float(gpu["memory"][0]))

            # CPU / IO
            if "process" in data:
                if "load" in data["process"] and data["process"]["load"] is not None:
                    cpu_load.append(float(data["process"]["load"]))
                    process_time.append(float(t))
                if "read_bytes" in data["process"] and data["process"]["read_bytes"] is not None:
                    read_bytes.append(int(data["process"]["read_bytes"]))
                if "write_bytes" in data["process"] and data["process"]["write_bytes"] is not None:
                    write_bytes.append(int(data["process"]["write_bytes"]))

            # items/sec
            if "rate" in data and data["rate"] is not None:
                items_per_second.append(float(data["rate"]))
                rate_time.append(float(t))

    # COMPUTE METRICS
    results = {}

    # Duration
    duration = float(end_time - start_time)
    results["duration"] = int(round(duration, 0))

    # Epoch time
    if epoch_time:
        s = basic_stats(epoch_time)
        results["epoch_measurement_count"] = s["count"]
        results["epoch_time_min"] = int(round(s["min"], 0))
        results["epoch_time_max"] = int(round(s["max"], 0))
        results["epoch_time_avg"] = int(round(s["mean"], 0))
        results["epoch_time_std"] = int(round(s["std"], 0))
        results["epoch_time_variance"] = int(round(s["variance"], 0))
        results["epoch_time_timeline"] = [int(round(x, 0)) for x in epoch_time]

        epoch_time_drop_first = epoch_time[1:]
        s = basic_stats(epoch_time_drop_first)
        results["epoch_drop_first_measurement_count"] = s["count"]
        results["epoch_drop_first_time_min"] = int(round(s["min"], 0))
        results["epoch_drop_first_time_max"] = int(round(s["max"], 0))
        results["epoch_drop_first_time_avg"] = int(round(s["mean"], 0))
        results["epoch_drop_first_time_std"] = int(round(s["std"], 0))
        results["epoch_drop_first_time_variance"] = int(round(s["variance"], 0))
        results["epoch_drop_first_time_timeline"] = [int(round(x, 0)) for x in epoch_time_drop_first]

    # Batch time
    if batches_per_epoch > 0:
        results["batches_per_epoch"] = batches_per_epoch

    if batch_time:
        s = basic_stats(batch_time)
        results["batch_measurement_count"] = s["count"]
        results["batch_time_min"] = int(round(s["min"], 0))
        results["batch_time_max"] = int(round(s["max"], 0))
        results["batch_time_avg"] = int(round(s["mean"], 0))
        results["batch_time_std"] = int(round(s["std"], 0))
        results["batch_time_variance"] = int(round(s["variance"], 0))

        p = percentiles(batch_time)
        results["batch_time_p10"] = int(round(p["p10"], 0))
        results["batch_time_p20"] = int(round(p["p20"], 0))
        results["batch_time_p30"] = int(round(p["p30"], 0))
        results["batch_time_p40"] = int(round(p["p40"], 0))
        results["batch_time_p50"] = int(round(p["p50"], 0))
        results["batch_time_p60"] = int(round(p["p60"], 0))
        results["batch_time_p70"] = int(round(p["p70"], 0))
        results["batch_time_p80"] = int(round(p["p80"], 0))
        results["batch_time_p90"] = int(round(p["p90"], 0))
        results["batch_time_timeline"] = [int(round(x, 0)) for x in batch_time]

        batch_time_drop_first = batch_time[batches_per_epoch:]
        s = basic_stats(batch_time_drop_first)
        results["batch_drop_first_measurement_count"] = s["count"]
        results["batch_drop_first_time_min"] = int(round(s["min"], 0))
        results["batch_drop_first_time_max"] = int(round(s["max"], 0))
        results["batch_drop_first_time_avg"] = int(round(s["mean"], 0))
        results["batch_drop_first_time_std"] = int(round(s["std"], 0))
        results["batch_drop_first_time_variance"] = int(round(s["variance"], 0))

        p = percentiles(batch_time_drop_first)
        results["batch_drop_first_time_p10"] = int(round(p["p10"], 0))
        results["batch_drop_first_time_p20"] = int(round(p["p20"], 0))
        results["batch_drop_first_time_p30"] = int(round(p["p30"], 0))
        results["batch_drop_first_time_p40"] = int(round(p["p40"], 0))
        results["batch_drop_first_time_p50"] = int(round(p["p50"], 0))
        results["batch_drop_first_time_p60"] = int(round(p["p60"], 0))
        results["batch_drop_first_time_p70"] = int(round(p["p70"], 0))
        results["batch_drop_first_time_p80"] = int(round(p["p80"], 0))
        results["batch_drop_first_time_p90"] = int(round(p["p90"], 0))
        results["batch_drop_first_time_timeline"] = [int(round(x, 0)) for x in batch_time_drop_first]

    # GPU load
    if gpu_load:
        s = basic_stats(gpu_load)
        results["gpu_measurement_count"] = s["count"]
        results["gpu_load_min"] = round(s["min"], 2)
        results["gpu_load_max"] = round(s["max"], 2)
        results["gpu_load_avg"] = round(s["mean"], 2)
        results["gpu_load_std"] = round(s["std"], 2)
        results["gpu_load_variance"] = round(s["variance"], 2)
        results["gpu_load_timeline"] = [round(x, 2) for x in gpu_load]

    # GPU memory
    if gpu_mem:
        s = basic_stats(gpu_mem)
        results["gpu_mem_min"] = int(round(s["min"], 0))
        results["gpu_mem_max"] = int(round(s["max"], 0))
        results["gpu_mem_avg"] = int(round(s["mean"], 0))
        results["gpu_mem_std"] = int(round(s["std"], 0))
        results["gpu_mem_variance"] = int(round(s["variance"], 0))
        results["gpu_mem_timeline"] = [int(round(x, 0)) for x in gpu_mem]

    # CPU load
    if cpu_load:
        s = basic_stats(cpu_load)
        results["process_measurement_count"] = s["count"]
        results["cpu_load_min"] = round(s["min"], 2)
        results["cpu_load_max"] = round(s["max"], 2)
        results["cpu_load_avg"] = round(s["mean"], 2)
        results["cpu_load_std"] = round(s["std"], 2)
        results["cpu_load_variance"] = round(s["variance"], 2)
        results["cpu_load_timeline"] = [round(x, 2) for x in cpu_load]

    # IO
    if read_bytes:
        results["read_bytes_total"] = sum(read_bytes)
        s = basic_stats(read_bytes)
        results["read_bytes_min"] = s["min"]
        results["read_bytes_max"] = s["max"]
        results["read_bytes_avg"] = s["mean"]
        results["read_bytes_std"] = s["std"]
        results["read_bytes_variance"] = s["variance"]
        results["read_bytes_timeline"] = read_bytes

    if write_bytes:
        results["write_bytes_total"] = sum(write_bytes)
        s = basic_stats(write_bytes)
        results["write_bytes_min"] = s["min"]
        results["write_bytes_max"] = s["max"]
        results["write_bytes_avg"] = s["mean"]
        results["write_bytes_std"] = s["std"]
        results["write_bytes_variance"] = s["variance"]
        results["write_bytes_timeline"] = write_bytes

    # Items/sec
    if items_per_second:
        s = basic_stats(items_per_second)
        results["rate_measurement_count"] = s["count"]
        results["items_per_second_min"] = round(s["min"], 2)
        results["items_per_second_max"] = round(s["max"], 2)
        results["items_per_second_avg"] = round(s["mean"], 2)
        results["items_per_second_std"] = round(s["std"], 2)
        results["items_per_second_variance"] = round(s["variance"], 2)
        results["items_per_second_timeline"] = [round(x, 2) for x in items_per_second]

    return results

def write_json(metrics, data_path: str):
    base, _ = os.path.splitext(data_path)
    out_path = base + ".metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return out_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    metrics = parse_file(sys.argv[1])
    out = write_json(metrics, sys.argv[1])

# Command: python parse_milabench_data.py [path to milabench_base]/runs/<run_id>/regnet_y_128gf.D0.data