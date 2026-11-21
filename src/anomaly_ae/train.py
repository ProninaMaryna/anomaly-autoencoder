"""Цикл навчання автоенкодера з ранньою зупинкою та логуванням по епохах."""
from __future__ import annotations

import copy
import logging
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import AnomalyData
from .evaluate import compute_metrics, reconstruction_scores


def _loss_fn(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "bce":
        return nn.BCELoss()
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(name)


def train_autoencoder(model: nn.Module, data: AnomalyData, cfg: dict, seed: int,
                      logger: logging.Logger) -> tuple[nn.Module, list[dict]]:
    t = cfg["train"]
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(data.train.x)),
                        batch_size=t["batch_size"], shuffle=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    loss_fn = _loss_fn(t.get("loss", "mse"))
    noise_std = float(t.get("noise_std", 0.0))

    es = t.get("early_stopping", {})
    patience, monitor = es.get("patience", 10), es.get("monitor", "val_loss")
    min_delta = float(es.get("min_delta", 0.0))
    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    history: list[dict] = []

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for (xb,) in loader:
            inp = xb
            if noise_std > 0:  # denoising autoencoder: на вході шум, ціль — чисте зображення
                inp = (xb + noise_std * torch.randn(xb.shape, generator=g)).clamp(0, 1)
            rec = model(inp)
            loss = loss_fn(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        train_time = time.perf_counter() - t0

        val_scores = reconstruction_scores(model, data.val.x)
        val_loss = float(val_scores[data.val.y == 0].mean())  # MSE на нормальних валідаційних
        val_auc = compute_metrics(data.val.y, val_scores)["roc_auc"]
        grad_norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                         if p.grad is not None)))
        row = {"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss, "val_auc": val_auc,
               "last_grad_norm": grad_norm, "epoch_time_s": train_time}
        history.append(row)
        logger.info("epoch %3d | train_loss %.5f | val_loss %.5f | val_auc %.4f | %.1fs",
                    epoch, row["train_loss"], val_loss, val_auc, train_time)

        current = val_loss if monitor == "val_loss" else -val_auc
        tol = max(1e-7, min_delta * abs(best_val)) if np.isfinite(best_val) else 0.0
        if current < best_val - tol:
            best_val, best_epoch, bad = current, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                logger.info("Рання зупинка на епосі %d (найкраща — %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    for row in history:
        row["best_epoch"] = best_epoch
    return model, history
