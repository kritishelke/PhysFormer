## PhysFormer: Physics-Informed RNNs and Transformers

Minimal research scaffold for physics-informed forecasting on the Lorenz system using PyTorch.

This repo can:
- **Generate Lorenz trajectories** via a simple RK4 simulator.
- **Create sliding-window datasets** for sequence-to-sequence forecasting.
- **Run either an LSTM or Transformer forecaster** with a shared interface.
- **Compute a hybrid loss**: data MSE + finite-difference Lorenz physics residual.
- **Train for a few epochs and print train/val losses** (total / data / physics).

### Repo structure

- `train.py` — entry point; trains LSTM or Transformer on synthetic Lorenz data.
- `requirements.txt` — minimal dependencies.
- `src/`
  - `__init__.py` — package marker.
  - `systems.py` — Lorenz RHS, RK4 simulator, and dataset generation utilities.
  - `dataset.py` — `WindowDataset` for sliding-window forecasting.
  - `models.py` — `LSTMForecaster` and `TransformerForecaster`.
  - `losses.py` — data MSE, Lorenz physics residual, and hybrid loss.

### Installation

From the repo root:

```bash
pip install -r requirements.txt
```

### Running training

From the same repo root:

```bash
python train.py --model lstm --epochs 2
python train.py --model transformer --epochs 2
```

Both commands:
- Use CPU-friendly defaults.
- Generate synthetic Lorenz trajectories.
- Train for a few epochs.
- Print train/val losses with **total / data / physics** components.

### Notes

This is a **minimal scaffold** intended for physics-informed long-horizon forecasting research. It is deliberately small, readable, and free of heavy frameworks so you can quickly prototype new loss terms, model variants, or benchmarking setups. Plotting and logging are intentionally left out to keep the core loop focused and runnable in a short time window.

