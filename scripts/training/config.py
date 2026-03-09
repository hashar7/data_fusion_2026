"""
All tuneable constants for the training pipeline.
Change values here — no other file needs to be touched for config changes.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_PATH     = "../../data/train_labels.parquet"
FEATURES_DIR    = "../data_processed"   # ALL periods in one directory
STAGING_DIR     = "../data_splits"
SUBMISSION_PATH = "submission.csv"

# Per-group model output paths (keyed by tx_type_group value)
TX_TYPE_GROUPS  = {0: "nonpayment", 1: "card", 2: "p2p"}
MODEL_OUT_PATHS = {
    0: "model_nonpayment.txt",
    1: "model_card.txt",
    2: "model_p2p.txt",
}

# ── Date boundaries ───────────────────────────────────────────────────────────
VAL_CUTOFF_DATE = "2025-04-01"
TRAIN_END_DATE  = "2025-06-01"

# ── Negative undersampling ────────────────────────────────────────────────────
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
LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",
    "verbosity":         -1,
    "device_type":       "cpu",
    "num_threads":       max(1, (os.cpu_count() or 4) - 1),
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 200,
    "learning_rate":     0.005,
    "n_estimators":      3000,
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
