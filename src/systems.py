import numpy as np
import torch
from typing import List, Optional, Tuple


def lorenz_rhs_np(
    x: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    Lorenz system right-hand side in NumPy.

    Parameters
    ----------
    x : np.ndarray
        State array of shape (..., 3).
    sigma, rho, beta : float
        Standard Lorenz parameters.

    Returns
    -------
    np.ndarray
        Time derivative dx/dt with same shape as x.
    """
    x_pos = x[..., 0]
    y_pos = x[..., 1]
    z_pos = x[..., 2]

    dx = sigma * (y_pos - x_pos)
    dy = x_pos * (rho - z_pos) - y_pos
    dz = x_pos * y_pos - beta * z_pos

    return np.stack([dx, dy, dz], axis=-1)


def lorenz_rhs_torch(
    x: torch.Tensor,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> torch.Tensor:
    """
    Lorenz system right-hand side in PyTorch.

    Parameters
    ----------
    x : torch.Tensor
        State tensor of shape (..., 3).

    Returns
    -------
    torch.Tensor
        Time derivative dx/dt with same shape as x.
    """
    x_pos = x[..., 0]
    y_pos = x[..., 1]
    z_pos = x[..., 2]

    dx = sigma * (y_pos - x_pos)
    dy = x_pos * (rho - z_pos) - y_pos
    dz = x_pos * y_pos - beta * z_pos

    return torch.stack([dx, dy, dz], dim=-1)


def _rk4_step(x: np.ndarray, dt: float) -> np.ndarray:
    """Single RK4 step for the Lorenz system."""
    k1 = lorenz_rhs_np(x)
    k2 = lorenz_rhs_np(x + 0.5 * dt * k1)
    k3 = lorenz_rhs_np(x + 0.5 * dt * k2)
    k4 = lorenz_rhs_np(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_lorenz(
    T: int = 2000,
    dt: float = 0.01,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Simulate a single Lorenz trajectory using RK4.

    Parameters
    ----------
    T : int
        Number of time steps.
    dt : float
        Time step size.
    x0 : np.ndarray, optional
        Initial condition of shape (3,). If None, a default is used.

    Returns
    -------
    np.ndarray
        Array of shape (T, 3) with the simulated trajectory.
    """
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    x = np.asarray(x0, dtype=np.float64)
    traj = np.empty((T, 3), dtype=np.float64)
    traj[0] = x

    for t in range(1, T):
        x = _rk4_step(x, dt)
        traj[t] = x

    return traj


def generate_lorenz_dataset(
    n_traj: int = 50,
    T: int = 2000,
    dt: float = 0.01,
    seed: int = 42,
    ic_range: Tuple[float, float] = (-10.0, 10.0),
) -> Tuple[List[np.ndarray], float]:
    """
    Generate a list of Lorenz trajectories with randomized initial conditions.

    Parameters
    ----------
    n_traj : int
        Number of trajectories to generate.
    T : int
        Number of time steps per trajectory.
    dt : float
        Time step size.
    seed : int
        Random seed for reproducibility.
    ic_range : (float, float)
        Uniform range for initial conditions in each dimension.

    Returns
    -------
    (list_of_trajectories, dt)
        list_of_trajectories is a list of np.ndarray each of shape (T, 3).
    """
    rng = np.random.RandomState(seed)
    low, high = ic_range

    trajectories: List[np.ndarray] = []
    for _ in range(n_traj):
        x0 = rng.uniform(low=low, high=high, size=(3,))
        traj = simulate_lorenz(T=T, dt=dt, x0=x0)
        trajectories.append(traj.astype(np.float32))

    return trajectories, dt