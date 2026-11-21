"""Запуск одного прогону (метод + конфігурація + сід) та серії прогонів з YAML-експерименту."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import fit_score
from .data import make_data
from .evaluate import compute_metrics, per_digit_detection, reconstruction_scores, threshold_from_normals
from .models import build_model, count_params
from .train import train_autoencoder
from .utils import (PROJECT_ROOT, deep_update, env_info, get_logger, load_yaml, save_json,
                    save_yaml, set_by_path, set_seed)


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return "-".join(map(str, v))
    return str(v)


def run_single(cfg: dict, seed: int, out_dir: Path, force: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))["summary"]

    set_seed(seed)
    torch.set_num_threads(cfg.get("num_threads", 8))
    logger = get_logger(out_dir.name, out_dir / "train.log")
    save_yaml({**cfg, "seed": seed}, out_dir / "config.yaml")
    logger.info("Конфігурація: %s", cfg)

    data = make_data(cfg, seed)
    logger.info("Дані: train=%d (аномалій %d), val=%d (аномалій %d), test=%d (аномалій %d)",
                len(data.train.y), data.train.y.sum(), len(data.val.y), data.val.y.sum(),
                len(data.test.y), data.test.y.sum())

    method = cfg.get("method", "autoencoder")
    t0 = time.perf_counter()
    extra: dict = {}
    if method == "autoencoder":
        model = build_model(cfg)
        extra["n_params"] = count_params(model)
        logger.info("Модель %s, параметрів: %d", cfg["model"]["arch"], extra["n_params"])
        model, history = train_autoencoder(model, data, cfg, seed, logger)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        extra["epochs_run"] = len(history)
        extra["best_epoch"] = history[-1]["best_epoch"]
        val_scores = reconstruction_scores(model, data.val.x)
        test_scores = reconstruction_scores(model, data.test.x)
        if cfg.get("save_model", True):
            torch.save(model.state_dict(), out_dir / "model.pt")
    else:
        val_scores, test_scores = fit_score(method, data.train.x, [data.val.x, data.test.x], seed,
                                            cfg.get("baseline_params", {}).get(method))
    extra["train_time_s"] = time.perf_counter() - t0

    thr = threshold_from_normals(val_scores[data.val.y == 0], cfg["eval"]["threshold_percentile"])
    val_m = compute_metrics(data.val.y, val_scores, thr)
    test_m = compute_metrics(data.test.y, test_scores, thr)
    per_digit = per_digit_detection(data.test.digit, data.test.y, test_scores, thr)
    np.savez_compressed(out_dir / "scores.npz", test_scores=test_scores, test_y=data.test.y,
                        test_digit=data.test.digit, val_scores=val_scores, val_y=data.val.y,
                        threshold=thr)

    d, m, t = cfg["data"], cfg.get("model", {}), cfg.get("train", {})
    summary = {
        "experiment": cfg.get("experiment", ""), "run": out_dir.name, "method": method,
        "arch": m.get("arch") if method == "autoencoder" else "-",
        "latent_dim": m.get("latent_dim") if method == "autoencoder" else None,
        "noise_std": t.get("noise_std") if method == "autoencoder" else None,
        "lr": t.get("lr") if method == "autoencoder" else None,
        "batch_size": t.get("batch_size") if method == "autoencoder" else None,
        "loss": t.get("loss") if method == "autoencoder" else None,
        "normal_classes": _fmt(d["normal_classes"]), "n_train": len(data.train.y),
        "contamination": d.get("contamination", 0.0), "seed": seed, **extra,
        **{f"val_{k}": v for k, v in val_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
    }
    save_json({"summary": summary, "val": val_m, "test": test_m, "per_digit_test": per_digit,
               "env": env_info()}, metrics_path)
    logger.info("TEST: ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f | FPR %.4f",
                test_m["roc_auc"], test_m["pr_auc"], test_m["f1"], test_m["fpr"])
    return summary


def expand_experiment(exp: dict) -> list[tuple[str, dict]]:
    """Розгортає сітку sweep у список (назва_прогону, конфігурація)."""
    base = load_yaml(PROJECT_ROOT / exp.get("base", "configs/base.yaml"))
    base = deep_update(base, exp.get("overrides", {}))
    base["experiment"] = exp["name"]
    sweep: dict = exp.get("sweep", {}) or {}
    keys = list(sweep)
    runs = []
    for values in itertools.product(*[sweep[k] for k in keys]) if keys else [()]:
        cfg = base
        parts = []
        for k, v in zip(keys, values):
            cfg = set_by_path(cfg, k, v)
            parts.append(f"{k.split('.')[-1]}={_fmt(v)}")
        runs.append(("__".join(parts) or "default", cfg))
    return runs


def run_experiment(exp_path: Path, force: bool = False, quick: bool = False,
                   shard: tuple[int, int] = (0, 1)) -> pd.DataFrame:
    """shard=(i, n): виконати лише кожен n-й прогін, починаючи з i (для паралельного запуску)."""
    exp = load_yaml(exp_path)
    results_dir = PROJECT_ROOT / "results" / exp["name"]
    jobs = [(rn, cfg, seed) for rn, cfg in expand_experiment(exp)
            for seed in exp.get("seeds", [cfg.get("seed", 0)])]
    rows = []
    for k, (run_name, cfg, seed) in enumerate(jobs):
        if k % shard[1] != shard[0]:
            continue
        if quick:
            cfg = set_by_path(cfg, "train.epochs", 2)
        out = results_dir / f"{run_name}__seed={seed}"
        print(f"=== {exp['name']} [{k + 1}/{len(jobs)}] :: {out.name}", flush=True)
        rows.append(run_single(cfg, seed, out, force=force))
    df = pd.DataFrame(rows)
    if shard[1] == 1:
        df.to_csv(results_dir / "runs.csv", index=False)
    return df
