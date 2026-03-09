"""
All tuneable constants for the training pipeline.
Change values here — no other file needs to be touched for config changes.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_PATH     = "../../data/train_labels.parquet"
FEATURES_DIR    = "../data_processed"   # ALL periods in one directory
STAGING_DIR     = "../data_splits"
MODEL_OUT_PATH  = "baseline_lgbm.txt"
SUBMISSION_PATH = "submission.csv"

# ── Date boundaries ───────────────────────────────────────────────────────────
# Train period : 2024-10-01 → 2025-05-31  (is_train == 1)
# Val   window : VAL_CUTOFF_DATE → TRAIN_END_DATE  (is_train == 1)
# Test  period : 2025-06-01 → 2025-08-09  (is_train == 1)
#
# TRAIN_END_DATE is the exclusive upper boundary of the train+val window.
# Any row with event_dttm >= TRAIN_END_DATE belongs to pretest or test.
VAL_CUTOFF_DATE = "2025-04-01"
TRAIN_END_DATE  = "2025-06-01"   # first date of the test period (exclusive)

# ── Negative undersampling ────────────────────────────────────────────────────
# Fraction of TRAINING negatives (label=0) to keep.
# All positives are always kept.
#
# None  → no undersampling; use full dataset (slowest)
# 0.05  → keep 5% of negatives  → ~1:20  pos:neg ratio  (recommended start)
# 0.02  → keep 2% of negatives  → ~1:8   pos:neg ratio  (fast iteration)
# 0.10  → more negative diversity; slower but less likely to miss neg patterns
NEG_SAMPLE_RATIO = 0.05
UNDERSAMPLE_SEED = 42

# ── Columns that are not model features ───────────────────────────────────────
NON_FEATURE_COLS = {
    "customer_id", "event_id", "event_dttm", "target", "is_train",
    "mcc_code", "accept_language", "browser_language",
    "battery", "device_system_version", "screen_size",
    "developer_tools", "compromised",
}

# ── LightGBM hyper-parameters ─────────────────────────────────────────────────
# is_unbalance is set dynamically in train.py depending on NEG_SAMPLE_RATIO.
LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",
    "verbosity":         -1,
    "device_type":       "cpu",
    "num_threads":       max(1, (os.cpu_count() or 4) - 1),
    # ── Tree structure ────────────────────────────────────────────────────────
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 200,
    # ── Learning rate & iterations ────────────────────────────────────────────
    "learning_rate":     0.005,
    "n_estimators":      3000,
    # ── Memory / speed ────────────────────────────────────────────────────────
    "max_bin":           255,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "seed":              42,
}

EARLY_STOPPING_ROUNDS = 150
LOG_EVAL_PERIOD       = 50
