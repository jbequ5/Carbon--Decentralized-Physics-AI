"""Benchmark data loader for Hydrogen.

Supports PDEBench and NeuralOperator standardized datasets.
"""

import os
from typing import Dict, Any, Tuple, Optional

import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class PDEBenchDataset(Dataset):
    def __init__(self, file_path: str, split: str = "train", resolution: Optional[Tuple[int, int]] = None):
        self.file_path = file_path
        self.split = split
        self.resolution = resolution

        with h5py.File(file_path, "r") as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as f:
            key = self.keys[idx]
            data = f[key][...]

        x = torch.tensor(data["input"], dtype=torch.float32)
        y = torch.tensor(data["output"], dtype=torch.float32)

        if self.resolution:
            # Simple resize if needed (placeholder)
            pass

        return x, y


def get_pdebench_loader(
    challenge_id: str,
    data_dir: str = "./data/pdebench",
    batch_size: int = 8,
    split: str = "train",
) -> DataLoader:
    """
    Returns a DataLoader for a PDEBench challenge.
    """
    filename_map = {
        "poisson_2d_v1": "poisson_2d.h5",
        "darcy_2d_v1": "darcy_2d.h5",
        "burgers_v1": "burgers.h5",
        "heat_v1": "heat.h5",
        "ns_2d_laminar_v1": "navier_stokes_2d.h5",
    }

    filename = filename_map.get(challenge_id, "default.h5")
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        # Fallback to synthetic data
        print(f"[Data] PDEBench file not found for {challenge_id}, using synthetic data.")
        return get_synthetic_loader(batch_size=batch_size)

    dataset = PDEBenchDataset(file_path, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def get_synthetic_loader(batch_size: int = 8, resolution: Tuple[int, int] = (64, 64)):
    """Fallback synthetic data loader."""
    class SyntheticDataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            x = torch.randn(3, *resolution)
            y = torch.randn(1, *resolution)
            return x, y

    return DataLoader(SyntheticDataset(), batch_size=batch_size, shuffle=True)


def get_benchmark_loader(challenge_id: str, backbone: str = "default", **kwargs):
    """
    Unified entrypoint. Tries PDEBench first, falls back to synthetic.
    Can be extended to use NeuralOperator dataset utilities.
    """
    # Future: Check if neuraloperator has a loader for this challenge
    return get_pdebench_loader(challenge_id, **kwargs)
