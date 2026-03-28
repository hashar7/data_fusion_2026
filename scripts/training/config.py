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

# ── Yellow row weighting ────────────────────────────────────────────────────────
# Labeled negative rows (yellow = explicitly confirmed non-fraud) receive this
# multiplier on top of the standard 1/neg_sample_ratio weight.  They are rare
# hard-negatives that define the fraud boundary more clearly than unlabeled greens.
YELLOW_WEIGHT_MULTIPLIER = 2.0

# ── Hierarchical ensemble (pipeline_v2) ────────────────────────────────────────
# Two global CatBoost models replace the per-group CatBoost blending:
#   (1) Suspicious detector  : (red | yellow) vs green  — P(labeled | tx)
#   (2) Red|Suspicious model : red vs yellow, labeled only — P(fraud | labeled, tx)
# Final product: sigmoid(susp_raw) × sigmoid(rgs_raw)  ≈ P(fraud | tx)
#
# These models use a curated ~50-feature set plus customer_id as a high-cardinality
# categorical feature (CatBoost handles this natively via ordered target statistics).

# Categorical features for hierarchical CatBoost — int-encoded in the parquet plus
# customer_id (Int64) and mcc_code_int (derived from mcc_code String at load time).
HIERARCHICAL_CAT_FEATURES = [
    "customer_id",                  # 100K unique — CatBoost handles via ordered stats
    "event_type_nm",
    "event_desc",
    "channel_indicator_type",
    "channel_indicator_sub_type",
    "currency_iso_cd",
    "pos_cd",
    "timezone",
    "operating_system_type",
    "phone_voip_call_state",
    "web_rdp_connection",
    "tx_type_group",
    "mcc_code_int",                 # derived: mcc_code (String) cast to Int32
]

# Numerical features — drawn from top-importance features + label-feedback + risk flags.
# All are confirmed present from feature_importances.csv / CLAUDE.md.
HIERARCHICAL_NUM_FEATURES = [
    # Amount
    "operaton_amt",
    "log_amount",
    "amount_diff_from_prev",
    "amount_ratio_prev",
    # Temporal
    "hour",
    "day_of_month",
    "week_of_year",
    "time_since_last_tx_minutes",
    "time_since_last_3_tx_mean",
    "time_gap_mean_30d",
    "time_gap_variance_30d",
    "time_gap_cv_30d",
    "minutes_from_midnight",
    # Per-customer behavioral
    "tx_count_lifetime",
    "event_desc_user_freq",
    "event_desc_share_user",
    "event_type_user_freq",
    "tx_time_zscore_given_user",
    "circadian_deviation_score",
    # Per-channel behavioral
    "spend_in_channel_lifetime",
    "tx_count_in_channel_lifetime",
    "channel_usage_share",
    # Z-scores / anomaly
    "amount_zscore_channel",
    "amount_zscore_given_channel",
    "amount_zscore_mcc",
    "amount_zscore_given_device",
    # Session
    "session_amount_sum",
    "session_duration_minutes",
    "session_avg_amount",
    "session_tx_count",
    # Rolling
    "tx_count_90d",
    "tx_count_30d",
    "amount_zscore_30d",
    "amount_mean_90d",
    "amount_std_90d",
    "cumulative_spend_90d",
    # Previous operations (sequence)
    "prev_1_op_desc",
    "prev_2_op_desc",
    "prev_3_op_desc",
    # Label feedback (Section K — populated when build_processed_dataset called with labels_lf)
    "fb_cust_prev_red_cnt",
    "fb_cust_prev_red_rate",
    "fb_cust_prev_any_red",
    "fb_sec_since_prev_red",
    "fb_sec_since_prev_yellow",
    "fb_cust_prev_labeled_cnt",
    "fb_desc_prev_red_rate",
    "fb_desc_prev_red_cnt",
    # Binary risk flags (Section J)
    "is_very_high_risk_desc",
    "is_near_certain_fraud_pair",
    "is_p2p_danger_channel",
    "is_high_risk_desc",
    "is_high_risk_channel",
]

# Fraction of unlabeled green rows to keep per partition for the suspicious model.
# 10 % gives ~6M green train rows — sufficient diversity, fits in 32 GB RAM.
SUSPICIOUS_GREEN_RATIO = 0.10

# Boundary between "recent" and "old" green transactions for weight assignment.
# Recent greens may include unreported fraud — give them slightly less penalty weight.
RECENT_BORDER = "2025-01-01"

# Sample weights for the suspicious model (is_labeled vs green task)
SUSPICIOUS_LABELED_WEIGHT   = 6.0    # red or yellow row
SUSPICIOUS_GREEN_RECENT_W   = 1.5    # green, event_dttm >= RECENT_BORDER
SUSPICIOUS_GREEN_OLD_W      = 1.0    # green, event_dttm < RECENT_BORDER

# Sample weights for the Red|Suspicious model (red vs yellow, labeled only)
RGS_RED_WEIGHT    = 2.5
RGS_YELLOW_WEIGHT = 1.0

# CatBoost params for the suspicious (P(labeled) detector) model
SUSPICIOUS_CATBOOST_PARAMS = {
    "iterations":     3000,
    "learning_rate":  0.05,
    "depth":          8,
    "l2_leaf_reg":    6.0,
    "loss_function":  "Logloss",
    "eval_metric":    "AUC",
    "task_type":      "CPU",
    "thread_count":   -1,
    "random_seed":    42,
    "od_type":        "Iter",
    "od_wait":        200,
    "verbose":        100,
    "allow_writing_files": False,
}

# CatBoost params for the Red|Suspicious (P(fraud | labeled)) model
RGS_CATBOOST_PARAMS = {
    "iterations":     5000,
    "learning_rate":  0.05,
    "depth":          8,
    "l2_leaf_reg":    8.0,
    "loss_function":  "Logloss",
    "eval_metric":    "PRAUC",
    "task_type":      "CPU",
    "thread_count":   -1,
    "random_seed":    42,
    "od_type":        "Iter",
    "od_wait":        300,
    "verbose":        100,
    "allow_writing_files": False,
}

# Saved model paths for hierarchical models
SUSPICIOUS_MODEL_PATH = "model_suspicious.cbm"
RGS_MODEL_PATH        = "model_rgs.cbm"

# ── Recent LightGBM (pipeline_v2) ──────────────────────────────────────────────
# Optional: train one extra LightGBM on data from RECENT_BORDER onward.
# Captures temporal drift in fraud patterns closer to the test period.
TRAIN_RECENT_LGBM        = True
RECENT_NEG_SAMPLE_RATIO  = 0.05   # negative undersampling for recent dataset

LGBM_PARAMS_RECENT = {
    **{k: v for k, v in {
        "objective":         "binary",
        "metric":            "average_precision",
        "verbosity":         -1,
        "device_type":       "cpu",
        "num_leaves":        127,
        "max_depth":         -1,
        "min_child_samples": 200,
        "learning_rate":     0.02,
        "n_estimators":      2000,
        "max_bin":           255,
        "subsample":         0.8,
        "subsample_freq":    1,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "seed":              42,
    }.items()},
    "num_threads": max(1, (__import__("os").cpu_count() or 4) - 1),
}
RECENT_MODEL_PATH = "model_recent.txt"
