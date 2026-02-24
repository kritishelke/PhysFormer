from typing import Tuple

import torch
import torch.nn.functional as F

from .systems import lorenz_rhs_torch


def data_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard MSE data loss.

    Parameters
    ----------
    pred, target : torch.Tensor
        Tensors of the same shape.
    """
    return F.mse_loss(pred, target)


def lorenz_physics_residual_loss(pred: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Finite-difference physics residual loss for Lorenz system.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted states of shape [B, H, 3].
    dt : float
        Time step used for finite differences.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """
    if pred.size(1) < 2:
        # Not enough steps to form finite differences.
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    x_t = pred[:, :-1, :]  # [B, H-1, 3]
    x_tp1 = pred[:, 1:, :]  # [B, H-1, 3]

    fd = (x_tp1 - x_t) / dt
    rhs = lorenz_rhs_torch(x_t)
    residual = fd - rhs

    # Mean squared residual over batch, time, and state dimensions.
    return torch.mean(residual ** 2)


def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dt: float,
    lambda_phys: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hybrid loss combining data MSE and physics residual loss.

    Parameters
    ----------
    pred : torch.Tensor
        Model prediction of shape [B, H, 3].
    target : torch.Tensor
        Target trajectory of shape [B, H, 3].
    dt : float
        Time step used in the Lorenz simulator.
    lambda_phys : float
        Weight for the physics residual term.

    Returns
    -------
    (total, l_data, l_phys) : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    l_data = data_loss(pred, target)
    l_phys = lorenz_physics_residual_loss(pred, dt)
    total = l_data + lambda_phys * l_phys
    return total, l_data, l_phys

