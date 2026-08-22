"""
modeling/unsupervised/autoencoder.py
=====================================
PyTorch Autoencoder for unsupervised anomaly detection.

Architecture & Training Philosophy
-----------------------------------
The Autoencoder is trained EXCLUSIVELY on non-fraud (legitimate) transactions.
It learns to reconstruct the manifold of normal transaction patterns.
At inference, fraud transactions produce high reconstruction error (MSE)
because they lie off the learned normal manifold.

Architecture:
  Input(n_features)
    -> Linear(128) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.2)
    -> Linear(64)  -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.2)
    -> Linear(32)  [bottleneck / latent space]
    -> Linear(64)  -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.2)
    -> Linear(128) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.2)
    -> Linear(n_features)  [reconstruction]

Loss: Mean Squared Error (MSE) on reconstruction
Optimiser: Adam (lr=1e-3, weight_decay=1e-5)
LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
Early stopping: patience=10 epochs on validation reconstruction loss

Anomaly score: per-sample MSE reconstruction error
Threshold: Expected Cost minimisation (since we have labels for evaluation)

Scientific justification:
  - Training only on legitimate data prevents the model from learning
    fraud patterns, ensuring high reconstruction error is a meaningful
    anomaly signal (Sakurada & Yairi, 2014; Pimentel et al., 2014).
  - BatchNorm stabilises training with heterogeneous financial features.
  - LeakyReLU avoids dying-ReLU in deep reconstruction layers.
  - Dropout provides regularisation against memorisation of training data.

Outputs (in models/):
  autoencoder_v1.pt              - trained PyTorch state dict
  autoencoder_v1_arch.json       - architecture config for re-instantiation
  ae_y_prob_v1.npy               - reconstruction errors (test set)
  ae_results_v1.json             - all metrics + CIs
  report/figures/models/ae_*     - training curves
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modeling.supervised.model_selection import (
    RANDOM_STATE,
    compute_metrics,
    full_bootstrap_ci,
    optimal_threshold_cost,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
MODEL_FIG_DIR = ROOT / "report" / "figures" / "models"
MODEL_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
LATENT_DIM = 32
HIDDEN_DIMS = [128, 64]
DROPOUT_RATE = 0.20
BATCH_SIZE = 1024
MAX_EPOCHS = 150
LR_INIT = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE_ES = 10       # early stopping patience (epochs)
PATIENCE_LR = 5        # LR scheduler patience


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class FraudAutoencoder(nn.Module):
    """
    Symmetric Autoencoder for fraud anomaly detection.

    Parameters
    ----------
    input_dim : int
        Number of input features (after feature pipeline).
    hidden_dims : list[int]
        Sizes of encoder hidden layers (mirror for decoder).
    latent_dim : int
        Bottleneck dimension.
    dropout_rate : float
        Dropout probability in each hidden block.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        latent_dim: int = LATENT_DIM,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # --- Encoder ---
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Decoder (mirror of encoder) ---
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Weight initialisation (Kaiming for LeakyReLU)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (used as anomaly score)."""
        recon, _ = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)

    def to_config(self) -> dict:
        return {
            "input_dim":   self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim":  self.latent_dim,
            "dropout_rate": self.dropout_rate,
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = PATIENCE_ES, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info(f"Restored best model (val loss: {self.best_loss:.6f})")


def _make_dataloader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.FloatTensor(X)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimiser.zero_grad()
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        # Gradient clipping (max norm=1.0) for training stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        recon, _ = model(batch)
        total_loss += criterion(recon, batch).item() * len(batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def score_samples(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Compute per-sample MSE reconstruction error as anomaly score."""
    model.eval()
    tensor = torch.FloatTensor(X).to(device)
    loader = DataLoader(TensorDataset(tensor), batch_size=4096, shuffle=False)
    scores = []
    for (batch,) in loader:
        err = model.reconstruction_error(batch)
        scores.append(err.cpu().numpy())
    return np.concatenate(scores)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_autoencoder(version: int = 1) -> dict:
    """Full Autoencoder training + evaluation pipeline."""
    log.info("=== Autoencoder Training ===")

    # Set all seeds for reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load data
    X_train_all = np.load(MODELS_DIR / f"X_train_v{version}.npy").astype(np.float32)
    X_test      = np.load(MODELS_DIR / f"X_test_v{version}.npy").astype(np.float32)
    y_train     = np.load(MODELS_DIR / f"y_train_v{version}.npy")
    y_test      = np.load(MODELS_DIR / f"y_test_v{version}.npy")
    log.info(f"Loaded arrays: train={X_train_all.shape}, test={X_test.shape}")

    # KEY: train ONLY on legitimate transactions
    X_legit = X_train_all[y_train == 0]
    log.info(
        f"Training on {len(X_legit):,} legitimate transactions "
        f"(excluded {(y_train == 1).sum():,} fraud transactions)"
    )

    # Split legitimate-only data into train/val for early stopping
    from sklearn.model_selection import train_test_split
    X_legit_tr, X_legit_val = train_test_split(
        X_legit, test_size=0.15, random_state=RANDOM_STATE
    )
    log.info(f"Autoencoder train: {len(X_legit_tr):,} | val: {len(X_legit_val):,}")

    train_loader = _make_dataloader(X_legit_tr, BATCH_SIZE, shuffle=True)
    val_loader   = _make_dataloader(X_legit_val, BATCH_SIZE, shuffle=False)

    # Build model
    input_dim = X_train_all.shape[1]
    model = FraudAutoencoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Autoencoder: {n_params:,} trainable parameters")
    log.info(f"Architecture: {input_dim} -> {HIDDEN_DIMS} -> {LATENT_DIM} -> {list(reversed(HIDDEN_DIMS))} -> {input_dim}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=PATIENCE_LR, min_lr=1e-6
    )
    early_stop = EarlyStopping(patience=PATIENCE_ES)

    # Training loop
    train_losses, val_losses, lr_history = [], [], []
    log.info(f"Starting training (max {MAX_EPOCHS} epochs, early stop patience={PATIENCE_ES})...")
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss  = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                f"train_loss={tr_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}"
            )

        if early_stop.step(val_loss, model):
            log.info(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE_ES} epochs)")
            break

    early_stop.restore_best(model)
    n_epochs_trained = len(train_losses)

    # Save training curves
    _plot_training_curves(train_losses, val_losses, lr_history, version)

    # Score test set
    log.info("Scoring test set...")
    y_prob_raw = score_samples(model, X_test, device)

    # Normalise reconstruction errors to [0, 1]
    y_prob = (y_prob_raw - y_prob_raw.min()) / (y_prob_raw.max() - y_prob_raw.min() + 1e-9)
    np.save(MODELS_DIR / f"ae_y_prob_v{version}.npy", y_prob)
    np.save(MODELS_DIR / f"ae_y_prob_raw_v{version}.npy", y_prob_raw)

    # Threshold selection
    theta_star, min_cost = optimal_threshold_cost(y_test, y_prob)
    log.info(f"Optimal threshold: {theta_star:.4f} (cost/txn: )")

    # Metrics
    metrics = compute_metrics(y_test, y_prob, threshold=theta_star, model_name="Autoencoder")
    ci = full_bootstrap_ci(y_test, y_prob, threshold=theta_star)
    metrics.update(ci)
    metrics.update({
        "epochs_trained":  n_epochs_trained,
        "n_params":        n_params,
        "input_dim":       input_dim,
        "latent_dim":      LATENT_DIM,
        "hidden_dims":     HIDDEN_DIMS,
        "best_val_loss":   round(early_stop.best_loss, 8),
        "n_legit_train":   int(len(X_legit)),
    })
    log.info(f"Test AUPRC: {metrics['auprc']:.4f}  AUROC: {metrics['auroc']:.4f}  MCC: {metrics['mcc']:.4f}")

    # Save model
    torch.save(model.state_dict(), MODELS_DIR / f"autoencoder_v{version}.pt")
    (MODELS_DIR / f"autoencoder_v{version}_arch.json").write_text(
        json.dumps(model.to_config(), indent=2)
    )
    log.info(f"Model saved -> {MODELS_DIR / f'autoencoder_v{version}.pt'}")

    save_results(metrics, f"ae_v{version}")
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    lr_history: list[float],
    version: int,
) -> None:
    """Two-panel figure: reconstruction loss + learning rate schedule."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(epochs, train_losses, label="Train MSE", color="#2c7bb6", linewidth=1.5)
    ax1.plot(epochs, val_losses,   label="Val MSE",   color="#d7191c", linewidth=1.5)
    ax1.set_ylabel("Reconstruction Loss (MSE)")
    ax1.set_title("Autoencoder Training Curves")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.plot(epochs, lr_history, color="#636363", linewidth=1.2)
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_yscale("log")

    plt.tight_layout()
    path = MODEL_FIG_DIR / f"ae_training_curves_v{version}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Training curves saved -> {path}")


# ---------------------------------------------------------------------------
# Inference utility (used by serving/api/predictor.py)
# ---------------------------------------------------------------------------

def load_autoencoder(version: int = 1) -> tuple[FraudAutoencoder, torch.device]:
    """Load trained autoencoder from disk for inference."""
    arch_path = MODELS_DIR / f"autoencoder_v{version}_arch.json"
    config = json.loads(arch_path.read_text())
    model = FraudAutoencoder(**config)
    state = torch.load(
        MODELS_DIR / f"autoencoder_v{version}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    device = torch.device("cpu")
    return model, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    results = train_autoencoder(version=args.version)
    print(json.dumps(results, indent=2))
