import os
import statistics

BASE_DIR = "[path to milabench_base]/data/FakeImageNet/train"
sizes = []

for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        if f.lower().endswith(".jpeg"):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            sizes.append(size)

num_files_train = len(sizes)
record_length_bytes = sum(sizes) / num_files_train
record_length_bytes_stdev = statistics.pstdev(sizes)

print("num_files_train:", num_files_train)
print("record_length_bytes:", record_length_bytes)
print("record_length_bytes_stdev:", record_length_bytes_stdev)

# Command: python train_stats.py