import argparse
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import WindowDataset
from src.losses import hybrid_loss
from src.models import LSTMForecaster, TransformerForecaster
from src.systems import generate_lorenz_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhysFormer: minimal Lorenz forecasting scaffold"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "transformer"],
        default="lstm",
        help="Model type to use (default: lstm).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hist_len", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--lambda_phys", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, horizon: int, input_dim: int = 3) -> nn.Module:
    if model_name == "lstm":
        return LSTMForecaster(input_dim=input_dim, horizon=horizon, output_dim=input_dim)
    elif model_name == "transformer":
        return TransformerForecaster(
            input_dim=input_dim,
            horizon=horizon,
            output_dim=input_dim,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def make_dataloaders(
    hist_len: int,
    horizon: int,
    batch_size: int,
    dt: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, float]:
    # Generate synthetic Lorenz trajectories.
    trajectories, dt_used = generate_lorenz_dataset(
        n_traj=40,
        T=1200,
        dt=dt,
        seed=seed,
    )

    # Split by trajectory into train/val.
    train_traj = trajectories[:32]
    val_traj = trajectories[32:]

    train_ds = WindowDataset(train_traj, hist_len=hist_len, horizon=horizon, stride=4)
    val_ds = WindowDataset(val_traj, hist_len=hist_len, horizon=horizon, stride=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dt_used


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dt: float,
    lambda_phys: float,
    train: bool = True,
) -> Tuple[float, float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    data_loss_sum = 0.0
    phys_loss_sum = 0.0
    n_batches = 0

    for x_hist, y_fut in loader:
        x_hist = x_hist.to(device)
        y_fut = y_fut.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            # Model expects [B, L, D]
            pred = model(x_hist)
            total, l_data, l_phys = hybrid_loss(
                pred, y_fut, dt=dt, lambda_phys=lambda_phys
            )

            if train:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss_sum += float(total.detach().cpu())
        data_loss_sum += float(l_data.detach().cpu())
        phys_loss_sum += float(l_phys.detach().cpu())
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss_sum / n_batches,
        data_loss_sum / n_batches,
        phys_loss_sum / n_batches,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, dt_used = make_dataloaders(
        hist_len=args.hist_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        dt=args.dt,
        seed=args.seed,
    )

    model = build_model(args.model, horizon=args.horizon, input_dim=3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_total, train_data, train_phys = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            dt=dt_used,
            lambda_phys=args.lambda_phys,
            train=True,
        )
        val_total, val_data, val_phys = run_one_epoch(
            model,
            val_loader,
            optimizer,
            device,
            dt=dt_used,
            lambda_phys=args.lambda_phys,
            train=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train - total: {train_total:.6f}, data: {train_data:.6f}, phys: {train_phys:.6f} | "
            f"Val - total: {val_total:.6f}, data: {val_data:.6f}, phys: {val_phys:.6f}"
        )


if __name__ == "__main__":
    main()

