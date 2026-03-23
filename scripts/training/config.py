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
MODELS_DIR      = "models"             # all model files saved here; created on first run

# Per-group model output paths (keyed by model_group value).
# model_group refines tx_type_group: nonpayment is split into two sub-models
# because event_type_nm=7 (70 M rows) dominates and masks the minority types.
#   0 = nonpayment-type7  (tx_type_group==0 AND event_type_nm==7)
#   1 = nonpayment-other  (tx_type_group==0 AND event_type_nm!=7)
#   2 = card              (tx_type_group==1)
#   3 = p2p               (tx_type_group==2)
TX_TYPE_GROUPS  = {0: "np_type7", 1: "np_other", 2: "card", 3: "p2p"}
MODEL_OUT_PATHS = {
    0: "model_np_type7.txt",
    1: "model_np_other.txt",
    2: "model_card.txt",
    3: "model_p2p.txt",
}

# ── Date boundaries ───────────────────────────────────────────────────────────
VAL_CUTOFF_DATE = "2025-04-01"
TRAIN_END_DATE  = "2025-06-01"

# ── Negative undersampling ────────────────────────────────────────────────────
NEG_SAMPLE_RATIO = 0.05          # default fallback
NEG_SAMPLE_RATIO_BY_GROUP = {
    0: 0.05,   # np_type7 — homogeneous population, standard ratio
    1: 0.05,   # np_other — bumped from 0.03 to give more boundary context (~142K rows)
    2: 0.05,   # card
    3: 0.05,   # p2p
}
UNDERSAMPLE_SEED = 42

# ── Columns that are not model features ───────────────────────────────────────
NON_FEATURE_COLS = {
    "customer_id", "event_id", "event_dttm", "target", "is_train",
    "mcc_code", "accept_language", "browser_language",
    "battery", "device_system_version", "screen_size",
    "developer_tools", "compromised",
    "model_group",   # routing key only — derived from tx_type_group + event_type_nm
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
    "n_estimators":      5000,
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

# ── Per-group LightGBM parameter overrides ────────────────────────────────────
# Merged on top of LGBM_PARAMS at training time.
# Groups 2 (card) and 3 (p2p) use LGBM_PARAMS unchanged.
LGBM_PARAMS_BY_GROUP = {
    0: {  # np_type7 — large dataset with slow tail; 2× faster LR saves ~4 min
        "learning_rate": 0.01,
        "n_estimators":  3000,
    },
    1: {  # np_other — small dataset (86K rows); finer splits expose rare fraud patterns
        "min_child_samples": 50,
        "num_leaves":        63,
    },
}

# Per-group early-stopping patience overrides.
# np_other has only 794 val positives — larger patience reduces false triggers.
EARLY_STOPPING_ROUNDS_BY_GROUP = {
    1: 200,
}

# ── Multi-seed ensemble ────────────────────────────────────────────────────────
# Each group is trained N times with different random seeds; val scores are
# averaged before blending.  5 seeds reduces variance at moderate wall-time cost
# (early stopping keeps individual runs short).
ENSEMBLE_SEEDS = [42, 7, 13, 17, 99]

# CatBoost multi-seed: train this many CatBoost seeds per group and average.
# 3 seeds balances variance reduction against wall-time (CatBoost is slower).
CATBOOST_SEEDS = [42, 7, 13]

# ── CatBoost parameters ────────────────────────────────────────────────────────
# Default fallback weight; overridden per-group via grid search at training time.
CATBOOST_BLEND_WEIGHT = 0.25   # fraction of CatBoost score in LightGBM+CatBoost blend

CATBOOST_PARAMS = {
    "iterations":            2000,
    "learning_rate":         0.05,
    "depth":                 8,
    "l2_leaf_reg":           3.0,
    "loss_function":         "Logloss",
    "eval_metric":           "PRAUC",
    "task_type":             "CPU",
    "thread_count":          -1,
    "random_seed":           42,
    "verbose":               100,
    "early_stopping_rounds": 100,
}

# Per-group CatBoost overrides (same merge pattern as LGBM_PARAMS_BY_GROUP).
CATBOOST_PARAMS_BY_GROUP: dict = {}

# ── Full-data retraining ──────────────────────────────────────────────────────
# After the stacker/blend weights are determined on the val split, retrain
# every base model on train + labeled-val rows combined using
# best_iteration × RETRAIN_FULL_ITER_FACTOR rounds (no early stopping).
# The submission is generated from the retrained models only.
RETRAIN_FULL_ITER_FACTOR = 1.05   # 5 % extra rounds to compensate for loss of val signal

# ── Dead feature blacklist ─────────────────────────────────────────────────────
# Features with exactly zero split gain across all seeds and groups (from
# feature_importances.csv).  Removing them reduces noise and speeds up training.
FEATURE_BLACKLIST = {
    "compromised_and_high_amount_flag",
    "timezone_mismatch",
    "burst_flag_15m",
    "voip_and_new_mcc_flag",
    "suspicious_env_flag",
    "mcc_rare_global_flag",
    "rare_combination_flag",
    "developer_tools_flag",
    "compromised_flag",
    "compromised_x_amount_ratio",
    "pos_cd_is_new",
    "mcc_is_new_for_user",
    "amount_usd_normalized",
    "amount_missing_flag",
    "new_device_and_night_flag",
    "new_mcc_flag",
    "new_channel_flag",
}

# ── Ensemble model path formats ────────────────────────────────────────────────
# {name}     = group name (np_type7, np_other, card, p2p)
# {seed_idx} = 0-based seed index within ENSEMBLE_SEEDS / CATBOOST_SEEDS
LGBM_MODEL_PATH_FMT     = "model_{name}_s{seed_idx}.txt"
CATBOOST_MODEL_PATH_FMT = "model_{name}_catboost_s{seed_idx}.cbm"
