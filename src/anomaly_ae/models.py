"""Архітектури автоенкодерів: повнозв'язний (Dense) та згортковий (Conv)."""
from __future__ import annotations

import torch
from torch import nn


class DenseAE(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden: tuple[int, ...] = (512, 256), dropout: float = 0.0):
        super().__init__()
        dims = [784, *hidden]
        enc: list[nn.Module] = [nn.Flatten()]
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        enc += [nn.Linear(dims[-1], latent_dim)]
        self.encoder = nn.Sequential(*enc)

        rdims = [latent_dim, *reversed(hidden)]
        dec: list[nn.Module] = []
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        dec += [nn.Linear(rdims[-1], 784), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28))]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ConvAE(nn.Module):
    """28x28 -> 14x14 -> 7x7 -> latent -> 7x7 -> 14x14 -> 28x28."""

    def __init__(self, latent_dim: int = 32, channels: tuple[int, int] = (32, 64), dropout: float = 0.0):
        super().__init__()
        c1, c2 = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(c2 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, c2 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (c2, 7, 7)),
            nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(c1, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m["arch"] == "dense":
        return DenseAE(m["latent_dim"], tuple(m.get("hidden", (512, 256))), m.get("dropout", 0.0))
    if m["arch"] == "conv":
        return ConvAE(m["latent_dim"], tuple(m.get("channels", (32, 64))), m.get("dropout", 0.0))
    raise ValueError(f"Невідома архітектура: {m['arch']}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
