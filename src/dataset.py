from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    Sliding-window dataset over a list of trajectories.

    Each trajectory is an array of shape (T, D). For each window we return:
      - x_hist: [hist_len, D]
      - y_fut: [horizon, D]
    """

    def __init__(
        self,
        trajectories: List[np.ndarray],
        hist_len: int = 32,
        horizon: int = 16,
        stride: int = 4,
    ) -> None:
        self.hist_len = hist_len
        self.horizon = horizon
        self.stride = stride

        # Precompute all (trajectory_index, start_index) pairs.
        self.trajectories = trajectories
        self.indices: List[Tuple[int, int]] = []

        for traj_idx, traj in enumerate(trajectories):
            T, _ = traj.shape
            window_size = hist_len + horizon
            max_start = T - window_size
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, stride):
                self.indices.append((traj_idx, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]

        window = traj[start : start + self.hist_len + self.horizon]
        x_hist_np = window[: self.hist_len]
        y_fut_np = window[self.hist_len :]

        x_hist = torch.from_numpy(x_hist_np.astype(np.float32))
        y_fut = torch.from_numpy(y_fut_np.astype(np.float32))

        return x_hist, y_fut

