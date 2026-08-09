"""Denoising / masked autoencoder for gene-expression data.

Can be used as a pre-training step or as a dimensionality-reduction
layer before the HMC head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ExpressionAutoencoder(nn.Module):
    """Denoising autoencoder with optional masking.

    Architecture::

        Encoder: Linear → ReLU → … → latent
        Decoder: latent → ReLU → … → Linear(reconstruction)

    Parameters
    ----------
    input_dim : int
        Number of gene-expression features.
    latent_dim : int
        Bottleneck dimension.
    hidden_dims : list[int]
        Hidden layer sizes for encoder (reversed for decoder).
    noise_std : float
        Standard deviation of Gaussian noise added during training
        (0 = no noise / standard autoencoder).
    mask_prob : float
        Probability of masking an input feature during training
        (0 = no masking).
    dropout : float
        Dropout probability in hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        hidden_dims: list[int] | None = None,
        noise_std: float = 0.1,
        mask_prob: float = 0.15,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        self.mask_prob = mask_prob

        if hidden_dims is None:
            hidden_dims = [min(1024, input_dim * 2), 512]

        # Encoder
        enc_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (reverse hidden dims)
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def _add_noise_and_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise and create masked version of input.

        Returns:
            ``(noisy_x, mask)`` where mask is 0 for masked positions.
        """
        noisy = x.clone()

        # Gaussian noise
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(x) * self.noise_std
            noisy = noisy + noise

        # Masking
        mask = torch.ones_like(x)
        if self.mask_prob > 0 and self.training:
            mask = (torch.rand_like(x) > self.mask_prob).float()
            noisy = noisy * mask

        return noisy, mask

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            ``(reconstruction, latent, mask)``.
        """
        noisy_x, mask = self._add_noise_and_mask(x)
        latent = self.encoder(noisy_x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent, mask

    def encode(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Encode features to latent representation (numpy in → numpy out)."""
        was_array = isinstance(x, np.ndarray)
        if was_array:
            x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            latent = self.encoder(x)
        if was_array:
            return latent.cpu().numpy()
        return latent

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute masked reconstruction loss (MSE on masked positions only)."""
        recon, _, mask = self.forward(x)
        mse = (recon - x) ** 2
        if mask is not None:
            mse = mse * mask
            return mse.sum() / mask.sum().clamp(min=1)
        return mse.mean()

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> ExpressionAutoencoder:
        """Train the autoencoder on a numpy feature matrix."""
        from torch.utils.data import DataLoader, TensorDataset  # pylint: disable=import-outside-toplevel

        device = next(self.parameters()).device
        X_t = torch.tensor(X.astype(np.float32))
        ds = TensorDataset(X_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                loss = self.reconstruction_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  AE epoch {epoch + 1}/{epochs} loss={total_loss:.4f}")

        self.eval()
        return self
