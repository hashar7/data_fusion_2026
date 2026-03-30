# Data Fusion 2026 — Guardian Track

Solution to the [Data Fusion 2026](https://fusioncontest.ru/) hackathon, Guardian track.

**Task:** classify unconfirmed bank transactions as fraud or legitimate.
**Metric:** PR-AUC (`sklearn.metrics.average_precision_score`).

---

## Problem Description

Bank clients sometimes do not confirm transactions for security reasons. The goal is to build a binary classifier that, for each such "unconfirmed" transaction, predicts whether it is fraudulent.

Key challenges:
- **Extreme class imbalance** — fraud transactions are very sparse
- **Temporal realism** — the model must simulate a live classifier; features may only use information available *before* the current transaction
- **Scale** — 200M+ operations across 100K clients spanning 1.5 years
- **Three label classes** — red (fraud), yellow (confirmed non-fraud), green (unlabeled open-loop)

### Data Periods

| Period | Dates | Role |
|--------|-------|------|
| Pre-train | 2023-10-01 – 2024-09-30 | Unlabeled transaction history for feature context |
| Train | 2024-10-01 – 2025-05-31 | Labeled data (red/yellow/green) |
| Pre-test | 2025-06-01 – 2025-08-09 | Unlabeled context preceding test day |
| Test | 2025-06-01 – 2025-08-09 | One final day per client — must be scored |

---

## Repository Structure

```
scripts/
  features/           Feature engineering modules (one file per section A–K)
    _helpers.py       Shared expression builders
    global_stats.py   Population-level stats + Bayesian target encodings
    transaction.py    Section A — datetime, amount flags, tx_type_group, interaction cols
    behavioral.py     Section B — cumulative per-customer history, lags, log-counts
    feedback.py       Section K — dynamic label feedback features
    rolling.py        Section D — rolling window aggregations (9 windows × 14 metrics)
    device.py         Section E — device/session risk flags
    temporal.py       Section F — global frequency lookups, circadian deviation
    zscore.py         Section G — channel/MCC z-scores, composite flags
    category_stats.py Sections H + I — per-category cumulative stats + target encodings
    category_risk.py  Section J — hardcoded binary risk flags
  feature_engineering.py   Public API — runs A → B → K → D → E → F → G → H → I → J
  feature_calculation.py   Public API — partitioned parquet output pipeline

  calculation/        Partitioned feature calculation
    config.py         OUTPUT_DIR, N_PARTITIONS, ID_COLS
    partition.py      process_partition(), _downcast()
    pipeline.py       build_processed_dataset() orchestration loop

  training/           Model training
    config.py         All tuneable constants (dates, params, seeds, paths)
    _utils.py         Progress display helpers
    data.py           load_labels(), build_memmaps(), memmap cache
    train.py          train_model() — LightGBM with negative undersampling
    train_catboost.py train_catboost_model()
    hierarchical.py   Hierarchical CatBoost helpers (train_v2 only)
    evaluate.py       evaluate() — PR-AUC, max-F1, feature importance
    predict.py        score_test() — submission CSV generation
    pipeline.py       train_baseline() orchestrator
    pipeline_v2.py    train_v2() orchestrator (recommended)
  train_eval_model.py Public API shim — re-exports train_baseline and train_v2

notebooks/
  feature_example.ipynb   Main experiment notebook (canonical usage example)
  models/                 Saved model files

misc/
  features.md         Complete feature catalogue (~450 features, organized by section)

data/                 Raw competition parquets (not in repo)
data_processed/       Processed feature parquets (output of build_processed_dataset)
data_splits/          Memmap files + memmap_meta.json (output of build_memmaps)
```

---

## Quick Start

```python
# In the notebook (working directory = notebooks/):
import sys; sys.path.insert(0, "..")

import polars as pl
from scripts.feature_engineering import generate_fraud_features, compute_global_stats
from scripts.feature_calculation import build_processed_dataset
from scripts.train_eval_model import train_v2

# 1. Compute global stats + target encodings from train+pretrain
pretrain_lf = pl.scan_parquet("../../data/pretrain_part_*.parquet")
train_lf    = pl.scan_parquet("../../data/train_part_*.parquet")
labels_lf   = pl.scan_parquet("../../data/train_labels.parquet")
full_lf     = pl.concat([pretrain_lf, train_lf])

global_stats = compute_global_stats(full_lf, labels_lf=labels_lf)

# 2. Build processed feature parquets
all_lf = pl.concat([
    pretrain_lf, train_lf,
    pl.scan_parquet("../../data/pretest.parquet"),
    pl.scan_parquet("../../data/test.parquet"),
])
build_processed_dataset(all_lf, global_stats=global_stats, labels_lf=labels_lf)

# 3. Train hierarchical ensemble + generate submission
train_v2()
```

After step 3, `notebooks/submission.csv` contains the scored test transactions.

> **After any feature engineering change:** delete `../data_splits/memmap_meta.json`
> (and the `.npy` files) to force a memmap rebuild.

---

## Feature Engineering

The pipeline produces ~450 features across 10 sections, all strictly leakage-free:
- Rolling windows use `closed="left"` — interval `[t − period, t)` excludes the current row
- Cumulative features use `cum_count() - 1` / `cum_sum() - current_value`
- Global statistics (target encodings, population means) are frozen on training data only

**Full feature catalogue:** [`misc/features.md`](misc/features.md)

### Pipeline execution order

```
A  transaction.py   — datetime decomposition, amount flags, tx_type_group, interaction cols
B  behavioral.py    — cumulative counts, lags (n=1..5), running-max, log-frequency
K  feedback.py      — dynamic label feedback (fraud/non-fraud history per customer)
D  rolling.py       — 9 windows × 14 metrics + card/P2P splits + derived ratios
E  device.py        — device risk flags, session features, screen dimensions
F  temporal.py      — global freq lookups, circadian deviation, velocity flags
G  zscore.py        — channel/MCC z-scores, combination freq, composite flags
H  category_stats.py— per-(customer,category) cumulative stats for 8 key dimensions
I  category_stats.py— Bayesian-smoothed target encodings (12 fraud rates)
J  category_risk.py — hardcoded binary risk flags from empirical fraud-rate analysis
```

### Key design choices

| Choice | Reason |
|--------|--------|
| `tx_type_group` (0/1/2) splits non-payment / card / P2P | Fraud rate and signal differ dramatically across types |
| `model_group` (0/1/2/3) further splits non-payment by `event_type_nm==7` | type-7 has 70M rows at 58% fraud and overwhelms other types in a single model |
| Interaction columns `channel_type_subtype`, `evtype_channel`, `evtype_subchannel` | Encode Cartesian combinations as single integers; used in rolling stats, log-counts, and category stats |
| Section K feedback features | Propagate prior label signal forward in time; `fb_cust_prev_any_red` and `fb_sec_since_prev_red` are among the strongest features |
| Bayesian target encodings (α=20) | Smooth fraud rates for low-frequency categories; unseen test values filled with global fraud rate |

---

## Training Pipelines

Two complete pipelines are available. **`train_v2()` is recommended** — it consistently outperforms `train_baseline()` on PR-AUC.

### Common infrastructure

Both pipelines share the same memmap build step:

```
build_memmaps()
    ├── Scans processed parquets, filters is_train==1
    ├── Rows with event_dttm < VAL_CUTOFF_DATE (2025-04-01)   → X_train / y_train
    ├── Rows with VAL_CUTOFF_DATE ≤ event_dttm < TRAIN_END_DATE (2025-06-01) → X_val / y_val
    ├── il_train.npy  — marks labeled (red+yellow)=1 vs green=0 in train split
    ├── il_val.npy    — marks labeled rows in val split
    └── tg_train.npy / tg_val.npy  — model_group routing key per row
```

The result is cached in `../data_splits/memmap_meta.json`. Delete it to force a rebuild.

**Label classes in training:**

| Class | `target` | Present in `train_labels.parquet` | `il_train` flag |
|-------|----------|----------------------------------|-----------------|
| Red (fraud) | 1 | Yes | 1 |
| Yellow (confirmed non-fraud) | 0 | Yes | 1 |
| Green (unlabeled open-loop) | 0 | No | 0 |

Yellow rows receive `YELLOW_WEIGHT_MULTIPLIER = 2.0` extra gradient weight on top of the standard undersampling correction weight, because they are rare hard-negatives that sharply define the fraud boundary.

---

### `train_baseline()` — Per-group LightGBM + CatBoost ensemble

```
scripts/training/pipeline.py
```

For each of the 4 model groups (`np_type7`, `np_other`, `card`, `p2p`):

**Step 1 — Multi-seed LightGBM**
- Trains `len(ENSEMBLE_SEEDS)` = 1 LightGBM model (configurable; set to `[42]` for speed)
- Negative undersampling at `NEG_SAMPLE_RATIO_BY_GROUP` (default 5% for all groups)
- Yellow rows get `YELLOW_WEIGHT_MULTIPLIER` extra weight
- Early stopping on labeled val rows (`PRAUC` metric, 150 rounds patience; 200 for `np_other`)
- Per-group parameter overrides via `LGBM_PARAMS_BY_GROUP`:
  - `np_type7`: faster LR (0.01) and fewer estimators (3000) — large homogeneous population
  - `np_other`: finer splits (`min_child_samples=50`, `num_leaves=63`) — small heterogeneous group
  - `p2p`: more estimators (8000) — still improving at default iteration cap

**Step 2 — Multi-seed CatBoost**
- Trains `len(CATBOOST_SEEDS)` = 1 CatBoost model per group
- Same undersampling, `eval_metric=PRAUC`, 100 rounds patience

**Step 3 — Per-group blend weight search**
- Grid searches `w ∈ {0.00, 0.05, …, 0.50}` on labeled val rows
- Final score = `lgbm_avg × (1 − w) + catboost_avg × w`
- `CATBOOST_BLEND_WEIGHT = 0.25` is the fallback default only

**Step 4 — Evaluation**
- PR-AUC per group on labeled val rows
- Combined PR-AUC across all groups (competition metric proxy)
- Top-30 feature importances printed for seed-0 of each group

**Step 5 — Full-data retraining**
- All models retrained on `train + labeled_val` combined
- Rounds = `best_iteration × RETRAIN_FULL_ITER_FACTOR (1.05)`, no early stopping
- Val negatives are undersampled at the same ratio before appending to avoid class imbalance shift

**Step 6 — Test scoring**
- `score_test()` routes each test row by `model_group`, blends per-group LGBM + CatBoost
- Writes `submission.csv`

**Saved model files (retrained):**
```
models/model_np_type7_s0.txt          (LGBM per seed)
models/model_np_type7_catboost_s0.cbm (CatBoost per seed)
models/model_np_other_s0.txt
... (4 groups × 1+ seeds each)
```

---

### `train_v2()` — Hierarchical ensemble (recommended)

```
scripts/training/pipeline_v2.py
```

Replaces per-group CatBoost with a global hierarchical decomposition:

```
P(fraud | tx)  ≈  P(labeled | tx)  ×  P(fraud | labeled, tx)
              =  sigmoid(susp_raw)  ×  sigmoid(rgs_raw)
```

The hierarchical pair provides orthogonal signal to per-group LGBM because it is trained globally (sees cross-group patterns) and uses native CatBoost categorical handling.

#### Step-by-step

**Step 1 — Build memmaps** (same as `train_baseline`)

**Step 2 — Pre-load val from parquet**
- Loads labeled val rows from processed parquets into a pandas DataFrame
- Also loads a 1%-sampled fraction of unlabeled val rows (for all-rows blend optimisation, `BLEND_ON_ALL_ROWS=True`)
- Both labeled and sampled unlabeled rows are used in the blend weight grid search

**Step 3 — Per-group LightGBM** (same as `train_baseline` minus CatBoost)
- 1 seed per group; each booster scores the pre-loaded val DataFrame before being freed
- `lgbm_blend_sum` accumulates scores aligned to the val DataFrame index

**Step 4 — Hierarchical CatBoost data load**
- `load_hier_split()` reads processed parquets, keeps:
  - **All labeled rows** (red + yellow) from the train period
  - **5% sampled green rows** (`HIER_FULL_GREEN_RATIO`) — deterministic hash-based sampling
- Features: when `HIER_USE_ALL_FEATURES=True`, uses the full LGBM feature set + CatBoost categorical columns (`customer_id`, `mcc_code_int`)

**Step 5 — Suspicious CatBoost** (target: `P(labeled | tx)`)
- Training: (red ∪ yellow) as positive, green as negative
- Sample weights:
  - Labeled rows: `SUSPICIOUS_LABELED_WEIGHT = 6.0`
  - Green rows from `RECENT_BORDER (2025-02-01)` onward: `SUSPICIOUS_GREEN_RECENT_W = 1.5`
  - Older green rows: `SUSPICIOUS_GREEN_OLD_W = 3.0` (reliably non-fraud — higher penalty)
- Eval metric: AUC on labeled val
- Params: `SUSPICIOUS_CATBOOST_PARAMS` (3000 iterations, depth 8, l2=6.0)

**Step 6 — RGS CatBoost** (target: `P(fraud | labeled, tx)`)
- Training: labeled rows only — red (fraud) vs yellow (confirmed non-fraud)
- Near-50% fraud rate produces a strong, clean signal
- Sample weights: `RGS_RED_WEIGHT = 2.5`, `RGS_YELLOW_WEIGHT = 1.0`
- Eval metric: PRAUC on labeled val
- Params: `RGS_CATBOOST_PARAMS` (5000 iterations, depth 8, l2=8.0)

**Step 7 — Main CatBoost** (target: direct fraud prediction)
- Global model: red=1 vs yellow+green=0
- Weights: red=10.0, yellow=2.5, green_recent=1.5, green_old=3.0
- Provides additional orthogonal signal via native CatBoost categorical handling
- Params: `MAIN_CATBOOST_PARAMS` (5000 iterations, depth 8, l2=8.0)

**Step 8 — Optional Recent LightGBM** (`TRAIN_RECENT_LGBM=True`)
- Single global LGBM trained on data from `RECENT_BORDER (2025-02-01)` onward
- Captures temporal drift in fraud patterns as the test period approaches
- `LGBM_PARAMS_RECENT`: LR=0.02, 2000 estimators

**Step 9 — Blend weight optimisation**
- Scores all models on the pre-loaded val DataFrame
- Hierarchical score: `sigmoid(susp_raw) × sigmoid(rgs_raw)`
- Blend: `final = w_lgbm * lgbm + w_hier * hier + w_main * main + w_recent * recent`
- When `BLEND_IN_LOGIT_SPACE=True`, blending is done in logit space (better-calibrated)
- When `BLEND_ON_ALL_ROWS=True`, optimises on labeled + sampled unlabeled val rows
- Grid search over the 4-weight simplex; best weights reported

**Step 10 — Full-data retraining**
- Per-group LGBM: retrained from memmaps, `best_iter × RETRAIN_FULL_ITER_FACTOR` rounds
- Hierarchical CatBoost: refitted with `refit_model()` on extended data (`cutoff_hi = TRAIN_END_DATE`)
- Main CatBoost: refitted on full data
- Recent LGBM: retrained on data up to `TRAIN_END_DATE`

**Step 11 — Test scoring**
- For each test parquet chunk:
  - LGBM score: per-group average
  - Hierarchical score: `sigmoid(susp) × sigmoid(rgs)`
  - Main CatBoost score
  - Recent LGBM score
  - Final: blended in logit space using optimised weights
- Writes `submission.csv`

**Saved model files (retrained):**
```
models/model_np_type7_s0.txt    (per-group LGBM)
models/model_np_other_s0.txt
models/model_card_s0.txt
models/model_p2p_s0.txt
models/model_suspicious.cbm    (global Suspicious CatBoost)
models/model_rgs.cbm           (global RGS CatBoost)
models/model_main_catboost.cbm (global Main CatBoost)
models/model_recent.txt        (global Recent LGBM)
```

Old model files are automatically archived as `model_*_ver_N.*` before each training run.

---

## Configuration Reference

All tuneable constants live in `scripts/training/config.py`. Key knobs:

| Constant | Default | Description |
|----------|---------|-------------|
| `VAL_CUTOFF_DATE` | `2025-04-01` | Train/val split boundary |
| `TRAIN_END_DATE` | `2025-06-01` | Val/test boundary |
| `NEG_SAMPLE_RATIO_BY_GROUP` | 0.05 all groups | Negative undersampling ratio |
| `YELLOW_WEIGHT_MULTIPLIER` | 2.0 | Extra weight for labeled negatives |
| `ENSEMBLE_SEEDS` | `[42]` | LGBM seeds (list → multi-seed averaging) |
| `RETRAIN_FULL_ITER_FACTOR` | 1.05 | Extra rounds for full-data retraining |
| `RECENT_BORDER` | `2025-02-01` | Start of "recent" window for Recent LGBM |
| `BLEND_IN_LOGIT_SPACE` | `True` | Blend model outputs in logit space |
| `BLEND_ON_ALL_ROWS` | `True` | Include sampled unlabeled rows in blend optimisation |
| `HIER_USE_ALL_FEATURES` | `True` | Use full LGBM feature set for hierarchical CatBoost |
| `TRAIN_RECENT_LGBM` | `True` | Train a global recent-data LGBM |
| `FEATURE_BLACKLIST` | 15 features | Zero-gain features excluded from all models |

---

## Hardware & Environment

- **CPU only, 32 GB RAM** — no GPU required (GPU support via `train_v2(gpu=True)`)
- Python 3.11, Polars 1.x, LightGBM, CatBoost, scikit-learn, NumPy, pandas
- Feature engineering processes one customer partition at a time (50 partitions)
- Training uses numpy memmaps to avoid loading all features into RAM simultaneously
- Output features are downcast to float32/int32 to halve disk and memory usage
