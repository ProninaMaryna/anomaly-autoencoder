"""Будує графіки (reports/figures) та markdown-таблиці (reports/tables) з results/.

python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anomaly_ae.data import make_data  # noqa: E402
from anomaly_ae.models import build_model  # noqa: E402
from anomaly_ae.utils import load_yaml  # noqa: E402

RES, FIG, TAB = ROOT / "results", ROOT / "reports" / "figures", ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Кольори категорій у фіксованому порядку.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#8a8985"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.edgecolor": "#8a8985",
    "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
    "axes.prop_cycle": plt.cycler(color=C),
})

df = pd.read_csv(RES / "experiments.csv")
df["normal_classes"] = df["normal_classes"].astype(str)


def ms(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std(ddof=1) if len(s) > 1 else 0:.3f}"


def table(g: pd.DataFrame, by: list[str], name: str, metrics=("val_roc_auc", "test_roc_auc", "test_pr_auc", "test_f1")):
    """Таблиця mean ± std по сідах (і класах, якщо їх не вказано в by)."""
    rows = []
    for key, sub in g.groupby(by, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n_runs"] = len(sub)
        for m in metrics:
            r[m] = ms(sub[m])
        # розкид лише між сідами (усереднений по нормальних класах) — міра стабільності
        r["seed_std_test_auc"] = f"{sub.groupby('normal_classes').test_roc_auc.std().mean():.4f}"
        if "epochs_run" in sub and sub["epochs_run"].notna().any():
            r["best_epoch"] = f"{sub['best_epoch'].mean():.1f}"
            r["time_s"] = f"{sub['train_time_s'].mean():.1f}"
        rows.append(r)
    t = pd.DataFrame(rows)
    (TAB / f"{name}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
    return t


def curve(ax, g, x, label, color, metric="test_roc_auc", logx=False, marker="o"):
    a = g.groupby(x)[metric].agg(["mean", "std"]).reset_index().sort_values(x)
    ax.plot(a[x], a["mean"], marker=marker, color=color, label=label)
    ax.fill_between(a[x], a["mean"] - a["std"], a["mean"] + a["std"], color=color, alpha=0.15, lw=0)
    if logx:
        ax.set_xscale("log")


def by_class_panels(g, x, xlabel, fname, title, logx=False, xticks=None):
    classes = sorted(g["normal_classes"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for i, c in enumerate(classes):
        sub = g[g["normal_classes"] == c]
        curve(ax[0], sub, x, f"норма = {c}", C[i], "test_roc_auc", logx)
        curve(ax[1], sub, x, f"норма = {c}", C[i], "val_roc_auc", logx)
    for a, t in zip(ax, ["TEST ROC-AUC", "VAL ROC-AUC (для вибору)"]):
        a.set_xlabel(xlabel); a.set_ylabel(t)
        if xticks is not None:
            a.set_xticks(xticks); a.set_xticklabels([str(v) for v in xticks])
    ax[0].legend()
    fig.suptitle(y=1.04, t=title + " (середнє ± std по 3 сідах)")
    fig.savefig(FIG / fname); plt.close(fig)


def hist(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "history.csv")


# ---------------- E1: архітектура + криві навчання ----------------
e1 = df[df.experiment == "e1_architecture"]
if len(e1):
    table(e1, ["arch", "normal_classes"], "e1_architecture_by_class")
    table(e1, ["arch"], "e1_architecture")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, arch in enumerate(["dense", "conv"]):
        for s in range(3):
            h = hist(RES / "e1_architecture" / f"normal_classes=8__arch={arch}__seed={s}")
            kw = dict(color=C[i], alpha=1 if s == 0 else 0.45, lw=2 if s == 0 else 1)
            lab = arch if s == 0 else None
            ax[0].plot(h.epoch, h.train_loss, ls="--", **kw)
            ax[0].plot(h.epoch, h.val_loss, label=lab, **kw)
            ax[1].plot(h.epoch, h.val_auc, label=lab, **kw)
            ax[2].plot(h.epoch, h.last_grad_norm, label=lab, **kw)
            ax[1].axvline(h.best_epoch.iloc[0], color=C[i], lw=0.8, ls=":")
    ax[0].set_yscale("log"); ax[0].set_title("Втрати (суцільна — val, пунктир — train)")
    ax[1].set_title("VAL ROC-AUC по епохах"); ax[2].set_title("Норма градієнта (останній батч)")
    ax[2].set_yscale("log")
    for a in ax:
        a.set_xlabel("епоха")
    ax[1].legend()
    fig.suptitle(y=1.04, t="E1: збіжність Dense vs Conv AE, норма = цифра 8, 3 сіди (точкова лінія — найкраща епоха)")
    fig.savefig(FIG / "e1_learning_curves.png"); plt.close(fig)

# ---------------- E2–E6: свіпи гіперпараметрів ----------------
e2 = df[df.experiment == "e2_latent_dim"]
if len(e2):
    table(e2, ["latent_dim"], "e2_latent_dim")
    table(e2, ["latent_dim", "normal_classes"], "e2_latent_dim_by_class")
    by_class_panels(e2, "latent_dim", "розмірність латентного простору", "e2_latent_dim.png",
                    "E2: вплив розміру bottleneck", logx=True, xticks=[2, 8, 16, 32, 64, 128])
    # Компроміс: якість реконструкції нормальних vs розділення
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for i, c in enumerate(sorted(e2.normal_classes.unique())):
        rows = []
        for _, r in e2[e2.normal_classes == c].iterrows():
            h = hist(RES / "e2_latent_dim" / r["run"])
            rows.append((r.latent_dim, h.val_loss.min()))
        a = pd.DataFrame(rows, columns=["ld", "vl"]).groupby("ld").mean()
        ax.plot(a.index, a.vl, marker="o", color=C[i], label=f"норма = {c}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([2, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["2", "8", "16", "32", "64", "128"])
    ax.set_xlabel("розмірність латентного простору"); ax.set_ylabel("мін. val MSE (нормальні)")
    ax.set_title("E2: помилка реконструкції нормальних даних"); ax.legend()
    fig.savefig(FIG / "e2_val_loss_vs_latent.png"); plt.close(fig)

e3 = df[df.experiment == "e3_denoising"]
if len(e3):
    table(e3, ["noise_std"], "e3_denoising")
    by_class_panels(e3, "noise_std", "σ гаусового шуму на вході", "e3_denoising.png", "E3: denoising AE")

e4 = df[df.experiment == "e4_optimization"]
if len(e4):
    table(e4, ["lr"], "e4_optimization")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, c in enumerate(sorted(e4.normal_classes.unique())):
        sub = e4[e4.normal_classes == c]
        curve(ax[0], sub, "lr", f"норма = {c}", C[i], "test_roc_auc", logx=True)
        curve(ax[1], sub, "lr", f"норма = {c}", C[i], "best_epoch", logx=True)
    sd = e4.groupby(["lr", "normal_classes"]).test_roc_auc.std().groupby("lr").mean()
    ax[2].plot(sd.index, sd.values, marker="o", color=C[0]); ax[2].set_xscale("log")
    for a, t in zip(ax, ["TEST ROC-AUC", "найкраща епоха (рання зупинка)", "std ROC-AUC між сідами"]):
        a.set_xlabel("learning rate"); a.set_ylabel(t)
    ax[0].legend()
    fig.suptitle(y=1.04, t="E4: швидкість навчання Adam (середнє ± std по 3 сідах)")
    fig.savefig(FIG / "e4_lr.png"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, lr in enumerate(sorted(e4.lr.unique())):
        h = hist(RES / "e4_optimization" / f"normal_classes=8__lr={lr}__seed=0")
        ax[0].plot(h.epoch, h.val_loss, color=C[i], label=f"lr={lr:g}")
        ax[1].plot(h.epoch, h.val_auc, color=C[i], label=f"lr={lr:g}")
    ax[0].set_yscale("log"); ax[0].set_ylabel("val MSE (нормальні)"); ax[1].set_ylabel("VAL ROC-AUC")
    for a in ax:
        a.set_xlabel("епоха")
    ax[1].legend(); fig.suptitle(y=1.04, t="E4: збіжність при різних learning rate (норма = 8, сід 0)")
    fig.savefig(FIG / "e4_convergence_lr.png"); plt.close(fig)

e4b = df[df.experiment == "e4b_loss"]
if len(e4b) and len(e4):
    both = pd.concat([e4b, e4[np.isclose(e4.lr, 0.001)]])
    table(both, ["loss"], "e4b_loss")

e5 = df[df.experiment == "e5_sample_efficiency"]
if len(e5):
    e5 = e5.copy()
    e5["n_train_cfg"] = e5["n_train"].where(e5["n_train"] <= 3000, -1)   # -1 = уся вибірка (~5000)
    table(e5, ["method", "n_train_cfg"], "e5_sample_efficiency")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for j, c in enumerate(sorted(e5.normal_classes.unique())):
        for i, m in enumerate(["autoencoder", "pca"]):
            curve(ax[j], e5[(e5.normal_classes == c) & (e5.method == m)], "n_train",
                  "Conv AE (latent 32)" if m == "autoencoder" else "PCA (16 комп.)", C[i], logx=True)
        ax[j].set_title(f"норма = {c}"); ax[j].set_xlabel("к-сть нормальних навчальних зразків")
    ax[0].set_ylabel("TEST ROC-AUC"); ax[0].legend()
    fig.suptitle(y=1.04, t="E5: вибіркова ефективність (середнє ± std по 3 сідах)")
    fig.savefig(FIG / "e5_sample_efficiency.png"); plt.close(fig)

e6 = df[df.experiment == "e6_contamination"]
if len(e6):
    table(e6, ["contamination"], "e6_contamination")
    by_class_panels(e6, "contamination", "частка аномалій у навчальній вибірці", "e6_contamination.png",
                    "E6: стійкість до забруднення train")

e7s = df[df.experiment == "e7_select_dense"]
if len(e7s):
    sel = pd.concat([e7s, e1[e1.arch == "dense"], e2[e2.latent_dim.isin([8, 16])]])
    table(sel, ["arch", "latent_dim"], "e7_selection",
          metrics=("val_roc_auc", "test_roc_auc", "test_pr_auc", "test_f1"))

e8 = df[df.experiment == "e8_early_stopping"]
if len(e8):
    base = e1.assign(monitor="val_loss")
    both = pd.concat([base, e8.assign(monitor="val_auc")])
    table(both, ["arch", "monitor"], "e8_early_stopping",
          metrics=("val_roc_auc", "test_roc_auc", "test_pr_auc", "test_f1"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (exp, lab) in enumerate([("e1_architecture", "val_loss (40 епох)"), ("e8_early_stopping", "val_auc (patience 8)")]):
        for j, arch in enumerate(["dense", "conv"]):
            h = hist(RES / exp / f"normal_classes=2__arch={arch}__seed=0")
            ax[j].plot(h.epoch, h.val_auc, color=C[i], label=f"монітор: {lab}")
            ax[j].axvline(h.best_epoch.iloc[0], color=C[i], ls=":", lw=1)
            ax[j].set_title(f"{arch} AE, норма = 2, сід 0"); ax[j].set_xlabel("епоха")
    ax[0].set_ylabel("VAL ROC-AUC"); ax[0].legend()
    fig.suptitle(y=1.04, t="E8: критерій ранньої зупинки (точкова лінія — відновлені ваги)")
    fig.savefig(FIG / "e8_early_stopping.png"); plt.close(fig)

# AUC vs MSE по епохах: «перегенералізація» автоенкодера
if len(e1):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, c in enumerate(["2", "5", "8"]):
        for i, arch in enumerate(["dense", "conv"]):
            hs = [hist(RES / "e1_architecture" / f"normal_classes={c}__arch={arch}__seed={s}") for s in range(3)]
            auc = np.mean([h.val_auc.values for h in hs], 0)
            ax[j].plot(np.arange(1, len(auc) + 1), auc, color=C[i], label=arch)
            ax[j].scatter([auc.argmax() + 1], [auc.max()], color=C[i], s=40, zorder=3, edgecolor="white")
        ax[j].set_title(f"норма = {c}"); ax[j].set_xlabel("епоха")
    ax[0].set_ylabel("VAL ROC-AUC (середнє 3 сідів)"); ax[0].legend()
    fig.suptitle(y=1.04, t="Пік якості виявлення раніше за мінімум val MSE (крапка — максимум AUC)")
    fig.savefig(FIG / "auc_peak_vs_epoch.png"); plt.close(fig)

# ---------------- E9: збіжність без ліміту 40 епох ----------------
e9 = df[df.experiment == "e9_convergence"]
e7ae = df[(df.experiment == "e7_final") & (df.method == "autoencoder")]
if len(e9):
    ref = e7ae[e7ae.normal_classes.isin(e9.normal_classes.unique())]
    rows = []
    for lab, g in [("40 епох (E7)", ref), ("до плато val MSE (E9)", e9)]:
        rows.append({"навчання": lab, "n_runs": len(g),
                     "epochs_run": f"{g.epochs_run.mean():.1f} [{g.epochs_run.min():.0f}–{g.epochs_run.max():.0f}]",
                     "best_epoch": f"{g.best_epoch.mean():.1f}",
                     "val_roc_auc": ms(g.val_roc_auc), "test_roc_auc": ms(g.test_roc_auc),
                     "test_pr_auc": ms(g.test_pr_auc), "test_f1": ms(g.test_f1),
                     "time_s": f"{g.train_time_s.mean():.1f}"})
    (TAB / "e9_convergence.md").write_text(pd.DataFrame(rows).to_markdown(index=False), encoding="utf-8")
    by_cls = []
    for c in sorted(e9.normal_classes.unique(), key=int):
        a, b = ref[ref.normal_classes == c], e9[e9.normal_classes == c]
        by_cls.append({"норма": c, "AUC, 40 епох": ms(a.test_roc_auc), "AUC, до плато": ms(b.test_roc_auc),
                       "епох до плато": f"{b.epochs_run.mean():.0f}"})
    (TAB / "e9_convergence_by_class.md").write_text(pd.DataFrame(by_cls).to_markdown(index=False),
                                                     encoding="utf-8")

    classes = sorted(e9.normal_classes.unique(), key=int)
    fig, ax = plt.subplots(1, len(classes), figsize=(13, 3.6))
    for j, c in enumerate(classes):
        tw = ax[j].twinx(); tw.grid(False); tw.spines["right"].set_visible(True)
        for s in range(3):
            run = RES / "e9_convergence" / f"normal_classes={c}__seed={s}"
            if not (run / "history.csv").exists():
                continue
            h = hist(run)
            ax[j].plot(h.epoch, h.val_loss, color=C[0], lw=1.3, alpha=0.85, label="val MSE" if s == 0 else None)
            tw.plot(h.epoch, h.val_auc, color=C[1], lw=1.3, alpha=0.85, label="val ROC-AUC" if s == 0 else None)
            ax[j].axvline(len(h), color=C[0], ls=":", lw=1)
        ax[j].axvline(40, color=GRAY, ls="--", lw=1.2)
        ax[j].text(41, ax[j].get_ylim()[0] * 1.03, " 40 епох", va="bottom", fontsize=8, color="#52514e")
        ax[j].set_yscale("log"); ax[j].set_xlabel("епоха"); ax[j].set_title(f"норма = {c}")
        tw.set_ylim(0.5, 1.0)
        if j == 0:
            ax[j].set_ylabel("val MSE (норма), log")
        if j == len(classes) - 1:
            tw.set_ylabel("val ROC-AUC")
        if j == 0:
            h1, l1 = ax[j].get_legend_handles_labels(); h2, l2 = tw.get_legend_handles_labels()
            ax[j].legend(h1 + h2, l1 + l2, loc="center right")
    fig.suptitle(y=1.04, t="E9: Dense AE до плато val MSE (3 сіди; пунктир — епоха зупинки)")
    fig.tight_layout(); fig.savefig(FIG / "e9_convergence.png"); plt.close(fig)

# ---------------- E7: фінальне порівняння на всіх 10 класах ----------------
e7 = df[df.experiment == "e7_final"]
if len(e7):
    names = {"autoencoder": "Autoencoder", "pca": "PCA", "iforest": "Isolation Forest", "ocsvm": "One-Class SVM"}
    methods = [m for m in names if m in e7.method.unique()]
    table(e7, ["method"], "e7_final_overall",
          metrics=("test_roc_auc", "test_pr_auc", "test_f1", "test_precision", "test_recall", "test_fpr"))
    piv = e7.groupby(["normal_classes", "method"]).test_roc_auc.agg(["mean", "std"])
    t = piv["mean"].unstack()[methods].round(3)
    t.loc["середнє"] = t.mean().round(3)
    t.rename(columns=names).to_markdown(TAB / "e7_final_by_class.md")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    classes = sorted(e7.normal_classes.unique(), key=int)
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        mm = [piv.loc[(c, m), "mean"] for c in classes]
        ss = [piv.loc[(c, m), "std"] for c in classes]
        ax.bar(np.arange(len(classes)) + (i - (len(methods) - 1) / 2) * w, mm, w * 0.92,
               yerr=ss, color=C[i], label=names[m], error_kw=dict(lw=1, ecolor="#52514e"))
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_xlabel("нормальний клас (цифра)"); ax.set_ylabel("TEST ROC-AUC"); ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color=GRAY, lw=1, ls="--")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("E7: one-vs-rest на MNIST — AE vs класичні методи (середнє ± std, 3 сіди)")
    fig.savefig(FIG / "e7_final_by_class.png"); plt.close(fig)

    # Стабільність по сідах
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sd = e7.groupby(["method", "normal_classes"]).test_roc_auc.std().unstack(0)[methods]
    ax.boxplot([sd[m].values for m in methods], labels=[names[m] for m in methods])
    ax.set_ylabel("std ROC-AUC між сідами"); ax.set_title("E7: розкид результатів між сідами (по 10 класах)")
    fig.savefig(FIG / "e7_seed_std.png"); plt.close(fig)

# ---------------- Аналіз помилок для фінальної AE-моделі ----------------
FOCUS = "8"
run = RES / "e7_final" / f"normal_classes={FOCUS}__method=autoencoder__seed=0"
if (run / "model.pt").exists():
    cfg = load_yaml(run / "config.yaml")
    z = np.load(run / "scores.npz")
    s, y, dg, thr = z["test_scores"], z["test_y"], z["test_digit"], float(z["threshold"])

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 60)
    ax.hist(s[y == 0], bins=bins, color=C[0], alpha=0.7, density=True, label="норма (цифра 8)")
    ax.hist(s[y == 1], bins=bins, color=C[1], alpha=0.6, density=True, label="аномалії (0–7, 9)")
    ax.axvline(thr, color="#0b0b0b", lw=1.2, ls="--", label="поріг (95-й перц. val-норми)")
    ax.set_xscale("log"); ax.set_xlabel("помилка реконструкції (MSE)"); ax.set_ylabel("щільність")
    ax.legend(); ax.set_title("Розподіл anomaly score на тесті")
    fig.savefig(FIG / "score_hist.png"); plt.close(fig)

    import json
    pdg = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["per_digit_test"]
    digs = sorted(pdg, key=int)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    vals = [pdg[d]["flagged_rate"] for d in digs]
    cols = [C[0] if d == FOCUS else C[1] for d in digs]
    ax.bar(digs, vals, color=cols)
    for d, v in zip(digs, vals):
        ax.text(d, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color="#52514e")
    ax.set_xlabel("цифра"); ax.set_ylabel("частка позначених як аномалія")
    ax.set_title("Виявлення за цифрами (синя — нормальний клас: це FPR)")
    fig.savefig(FIG / "per_digit_detection.png"); plt.close(fig)

    model = build_model(cfg)
    model.load_state_dict(torch.load(run / "model.pt", weights_only=True)); model.eval()
    data = make_data(cfg, 0)
    xt = data.test.x
    order = np.argsort(s)
    fn = [i for i in order if y[i] == 1][:8]                  # аномалії з найменшою помилкою
    fp = [i for i in order[::-1] if y[i] == 0][:8]            # норма з найбільшою помилкою
    tn = [i for i in order if y[i] == 0][:8:1]
    tp = [i for i in order[::-1] if y[i] == 1][:8]
    groups = [("Норма, добре реконструйовано", tn), ("Хибно-позитивні (норма, велика помилка)", fp),
              ("Хибно-негативні (аномалія, мала помилка)", fn), ("Найочевидніші аномалії", tp)]
    with torch.no_grad():
        fig, axes = plt.subplots(8, 8, figsize=(9, 12.5))
        for gi, (title, idx) in enumerate(groups):
            rec = model(torch.from_numpy(xt[idx])).numpy()
            for k, i in enumerate(idx):
                for r, img in enumerate([xt[i, 0], rec[k, 0]]):
                    a = axes[gi * 2 + r, k]
                    a.imshow(img, cmap="gray_r", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
                    a.grid(False)
                    if r == 0:
                        a.set_title(f"{dg[i]} | {s[i]:.3f}", fontsize=7)
            axes[gi * 2, 0].set_ylabel("вхід", fontsize=7); axes[gi * 2 + 1, 0].set_ylabel("рекон.", fontsize=7)
            axes[gi * 2, 0].annotate(title, (0, 1.45), xycoords="axes fraction", fontsize=9, weight="bold")
        fig.suptitle("Аналіз помилок: фінальний AE, норма = 8 (підпис: цифра | MSE)", y=0.995)
        fig.subplots_adjust(hspace=0.75, top=0.93); fig.savefig(FIG / "error_analysis.png"); plt.close(fig)

print("Графіки:", sorted(p.name for p in FIG.glob("*.png")))
print("Таблиці:", sorted(p.name for p in TAB.glob("*.md")))
