"""
Step 11: Federated Learning (FL) Partitions.

This script reads `outputs/final_npy/metadata_train.csv` and generates
IID and Non-IID data partitions across a specified number of clients.
It operates on the `patient_id` level to prevent data leakage during FL.

Output: 
- `outputs/final_npy/fl_partitions_iid.json`
- `outputs/final_npy/fl_partitions_noniid.json`
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from config import OUTPUT_DIR, RANDOM_SEED


N_CLIENTS = 5
FINAL_DIR = OUTPUT_DIR / "final_npy"


def create_iid_partition(patients: list, n_clients: int) -> dict:
    np.random.seed(RANDOM_SEED)
    shuffled = np.random.permutation(patients)
    chunks = np.array_split(shuffled, n_clients)
    return {f"client_{i+1}": chunk.tolist() for i, chunk in enumerate(chunks)}


def create_non_iid_partition(patient_df: pd.DataFrame, n_clients: int) -> dict:
    """Simple Non-IID by sorting patients based on their label."""
    # Group by patient_id to get patient-level label (e.g. max label)
    p_labels = patient_df.groupby("patient_id")["label"].max().sort_values()
    sorted_patients = p_labels.index.tolist()
    
    # Split the sorted patients into chunks. 
    # Clients at the beginning will have mostly label 0, 
    # and clients at the end will have mostly label 1.
    chunks = np.array_split(sorted_patients, n_clients)
    return {f"client_{i+1}": chunk.tolist() for i, chunk in enumerate(chunks)}


def main():
    metadata_path = FINAL_DIR / "metadata_train.csv"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Run previous steps first.")
        return

    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} samples from train split.")

    unique_patients = df["patient_id"].unique().tolist()
    print(f"Total unique patients for FL: {len(unique_patients)}")

    # Create IID
    iid_parts = create_iid_partition(unique_patients, N_CLIENTS)
    iid_out = FINAL_DIR / "fl_partitions_iid.json"
    with open(iid_out, "w") as f:
        json.dump(iid_parts, f, indent=2)
    print(f"[ok] Saved IID partitions to {iid_out}")

    # Create Non-IID
    noniid_parts = create_non_iid_partition(df, N_CLIENTS)
    noniid_out = FINAL_DIR / "fl_partitions_noniid.json"
    with open(noniid_out, "w") as f:
        json.dump(noniid_parts, f, indent=2)
    print(f"[ok] Saved Non-IID partitions to {noniid_out}")


if __name__ == "__main__":
    main()
