"""
All tuneable constants for the training pipeline.
Change values here — no other file needs to be touched for config changes.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELS_PATH     = "../../data/raw/train_labels.parquet"
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
# averaged before blending and calibration.  3 seeds gives a good bias/variance
# tradeoff without tripling the wall time (early stopping keeps runs short).
ENSEMBLE_SEEDS = [42, 7, 13]

# ── CatBoost parameters ────────────────────────────────────────────────────────
CATBOOST_BLEND_WEIGHT = 0.25   # fraction of CatBoost score in LightGBM+CatBoost blend

CATBOOST_PARAMS = {
    "iterations":            2000,
    "learning_rate":         0.075,
    "depth":                 11,
    "l2_leaf_reg":           3.0,
    "loss_function":         "Logloss",
    "eval_metric":           "AUC",
    "task_type":             "CPU",
    "thread_count":          -1,
    "random_seed":           42,
    "verbose":               100,
    "early_stopping_rounds": 100,
}

# Per-group CatBoost overrides (same merge pattern as LGBM_PARAMS_BY_GROUP).
CATBOOST_PARAMS_BY_GROUP: dict = {}

# ── Ensemble model path formats ────────────────────────────────────────────────
# {name}     = group name (np_type7, np_other, card, p2p)
# {seed_idx} = 0-based seed index within ENSEMBLE_SEEDS
LGBM_MODEL_PATH_FMT     = "model_{name}_s{seed_idx}.txt"
CATBOOST_MODEL_PATH_FMT = "model_{name}_catboost.cbm"
CALIBRATOR_PATH_FMT     = "calibrator_{name}.pkl"


# ── F vs G random forrest config ────────────────────────────────────────────────

RF_N_JOBS = -1
RF_CV_N_SPLITS = 8
RF_CV_SEED = 42
RF_MODEL_FILENAME = "model_rf_fg.pkl"

# Fixed RF settings
RF_BASE_PARAMS = {
    "bootstrap": True,
    "class_weight": "balanced_subsample",
    "random_state": RF_CV_SEED,
    "n_jobs": RF_N_JOBS
}

# Hyperparameter plane for 6-fold CV
RF_PARAM_GRID = {
    "n_estimators": [1500],
    "max_depth": [20],
    "min_samples_split": [20],
    "min_samples_leaf": [1],
    "max_features": ['sqrt'],
}


# ── F vs U catboost model ────────────────────────────────────────────────––––––

CATBOOST_FU_MODEL_FILENAME = "model_catboost_fu.cbm"
