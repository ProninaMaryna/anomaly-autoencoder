# Виявлення аномальних даних за допомогою автоенкодера

Автоенкодер навчається реконструювати тільки «нормальні» дані. На аномаліях помилка реконструкції
(MSE) виходить більшою, тому саме вона слугує оцінкою аномальності (anomaly score). Дані — MNIST
у постановці **one-vs-rest**: одна цифра вважається нормою, решта дев'ять цифр — аномаліями.

📄 **Короткий технічний звіт:** [REPORT.md](REPORT.md) · 📊 **Таблиця всіх прогонів:** [results/experiments.csv](results/experiments.csv)

## Структура

```
configs/
  base.yaml                 # базова конфігурація (дані, модель, навчання, оцінка)
  experiments/*.yaml        # експерименти: перевизначення + сітка sweep + сіди
src/anomaly_ae/
  data.py                   # розбиття train/val/test, контамінація, обмеження n_train
  models.py                 # DenseAE, ConvAE
  train.py                  # цикл навчання, denoising, рання зупинка, логування по епохах
  evaluate.py               # anomaly score, поріг, ROC-AUC / PR-AUC / F1 / FPR, аналіз за цифрами
  baselines.py              # PCA, Isolation Forest, One-Class SVM
  runner.py                 # один прогін і розгортання сітки експерименту
scripts/
  run_experiment.py         # запуск одного YAML-експерименту (опційно --shard i/n)
  run_all.py                # паралельний запуск кількох експериментів
  collect_results.py        # збирання metrics.json -> results/experiments.csv
  make_figures.py           # графіки reports/figures + таблиці reports/tables
results/<експеримент>/<прогін>/
  config.yaml  train.log  history.csv  metrics.json  scores.npz   (model.pt — не в git)
reports/figures, reports/tables
```

## Відтворення

```bash
pip install -r requirements.txt
# MNIST завантажиться автоматично в data/ під час першого запуску
python scripts/run_experiment.py configs/experiments/e0_smoke.yaml --quick       # перевірка коду (~10 с)
python scripts/run_all.py configs/experiments/e*.yaml --workers 4               # усі експерименти (~3–4 год на CPU)
python scripts/collect_results.py
python scripts/make_figures.py
```

Кожен прогін повністю задається файлом `config.yaml` і сідом. Сід фіксує розбиття даних,
ініціалізацію ваг, порядок батчів і шум. Прогони, які вже мають `metrics.json`, пропускаються
при повторному запуску, тож перерваний запуск можна продовжити.
Один прогін вручну:

```bash
python scripts/run_experiment.py configs/experiments/e7_final.yaml --shard 0/120   # лише перший із 120 прогонів
```

## Експерименти

| ID | Що досліджується | Сітка | Прогонів |
|----|------------------|-------|---------:|
| E1 | архітектура | dense / conv × норма {2,5,8} × 3 сіди | 18 |
| E2 | розмір bottleneck | latent ∈ {2,8,16,32,64,128} | 54 |
| E3 | denoising AE | σ шуму ∈ {0, 0.1, 0.2, 0.4} | 36 |
| E4 | learning rate | {1e-4 … 1e-2} | 45 |
| E4b | функція втрат | MSE / BCE / L1 | 18 (+9 з E4) |
| E5 | вибіркова ефективність | n_train ∈ {50 … усі} × {AE, PCA} | 108 |
| E6 | забруднення train аномаліями | 0 … 20 % | 45 |
| E7-select | добір фінальної конфігурації | Dense, latent ∈ {8,16} | 18 |
| E8 | критерій ранньої зупинки | val_loss vs val_auc | 18 |
| E9 | достатність 40 епох | Dense AE до 200 епох, зупинка на плато val MSE | 9 |
| E7 | фінальне порівняння | 10 класів × {AE, PCA, IF, OC-SVM} × 3 сіди | 120 |

Гіперпараметри обиралися тільки за **валідаційним** ROC-AUC (MNIST-train). Тестовий набір
(MNIST-test) у виборі моделі участі не брав.
