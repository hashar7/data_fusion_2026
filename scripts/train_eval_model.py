"""
Thin entry point — kept for backward compatibility.
All logic lives in scripts/training/:
  config.py   — tuneable constants (paths, dates, LightGBM params)
  data.py     — load_labels, count_rows, build_memmaps, memmap cache
  train.py    — train_model (undersampling + lgb.train)
  evaluate.py — evaluate   (PR-AUC, max-F1, feature importance)
  predict.py  — score_test (submission CSV)
  pipeline.py — train_baseline (orchestrator)
"""
from scripts.training.pipeline import train_baseline, train_rf_fg_by_group, train_catboost_fu_by_group, train_final_ensemble_by_group  # noqa: F401 (re-exported)
from scripts.training import (                         # noqa: F401
    train_model, 
    evaluate, 
    score_test, score_test_final_ensemble_by_group,
    load_labels, build_memmaps,
    build_labeled_train_indices, 
    train_rf_model, sanitize_rf_features,
    LGBM_PARAMS, NON_FEATURE_COLS,
    RF_BASE_PARAMS, RF_CV_SEED,
    CATBOOST_FU_MODEL_PATH_FMT, 
)
