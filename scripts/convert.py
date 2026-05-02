import os
import time
from pathlib import Path
import wfdb
import pandas as pd
import numpy as np
import torch

PT_PATH = "dataset_cache.pt"
RAW_DIR = "data/wfdb"

print("Converting WFDB files to single .pt cache (one-time operation)...")
start = time.time()

# Load SNOMED code mapping
csv_path = os.path.join(RAW_DIR, "ConditionNames_SNOMED-CT.csv")
df = pd.read_csv(csv_path)
unique_codes = df["Snomed_CT"].astype(str).unique().tolist()
code_to_idx = {code: idx for idx, code in enumerate(unique_codes)}
num_classes = len(unique_codes)
print(f"  {num_classes} SNOMED-CT classes")

# Find all records
mat_files = sorted(Path(RAW_DIR).rglob("*.mat"))
print(f"  {len(mat_files)} .mat files found")

signals_list = []
labels_list = []
skipped = 0

for i, mat_path in enumerate(mat_files):
    if i % 5000 == 0:
        print(f"  Processing {i}/{len(mat_files)}...")
    try:
        record_path = str(mat_path)[:-4]  # strip .mat
        record = wfdb.rdrecord(record_path)
        sig = np.nan_to_num(record.p_signal).T  # (12, seq_len)

        # Build multi-hot label
        label = np.zeros(num_classes, dtype=np.float32)
        for comment in record.comments:
            if comment.startswith("Dx:"):
                for code in comment.replace("Dx:", "").strip().split(","):
                    code = code.strip()
                    if code in code_to_idx:
                        label[code_to_idx[code]] = 1.0
                break

        signals_list.append(torch.FloatTensor(sig))
        labels_list.append(torch.FloatTensor(label))
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        skipped += 1

elapsed = time.time() - start
print(f"  Done! {len(signals_list)} records loaded, {skipped} skipped, in {elapsed:.1f}s")

# Save as single .pt file
print(f"  Saving to {PT_PATH}...")
torch.save({
    "signals": signals_list,
    "labels": labels_list,
    "unique_codes": unique_codes,
    "code_to_idx": code_to_idx,
    "num_classes": num_classes,
}, PT_PATH)
size_gb = os.path.getsize(PT_PATH) / 1e9
print(f"  Saved! File size: {size_gb:.2f} GB")