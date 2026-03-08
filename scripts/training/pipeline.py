"""Top-level orchestrator: glues data → train → evaluate → predict."""
import gc
import time
from datetime import datetime

from scripts.training.config import (
    FEATURES_DIR, STAGING_DIR, MODEL_OUT_PATH, SUBMISSION_PATH,
    VAL_CUTOFF_DATE, TRAIN_END_DATE,
)
from scripts.training._utils import _fmt, _parquet_files
from scripts.training.data import load_labels, count_rows, build_memmaps, _load_cache
from scripts.training.train import train_model
from scripts.training.evaluate import evaluate
from scripts.training.predict import score_test


def train_baseline() -> None:
    total_start = time.perf_counter()
    cutoff    = datetime.fromisoformat(VAL_CUTOFF_DATE)
    train_end = datetime.fromisoformat(TRAIN_END_DATE)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")
    print(f"Found {len(files)} parquet partitions in {FEATURES_DIR!r}\n", flush=True)

    # 1. Build (or reload) memmap split files
    # If memmap_meta.json + the five .npy files already exist in STAGING_DIR,
    # the full streaming pass is skipped. Delete any of those files to rebuild.
    cached = _load_cache(STAGING_DIR)
    if cached is not None:
        X_train, y_train, X_val, y_val, il_val, feature_cols = cached
    else:
        labels = load_labels()
        n_train, n_val = count_rows(files, cutoff, train_end)
        X_train, y_train, X_val, y_val, il_val, feature_cols = build_memmaps(
            files, labels, cutoff, train_end, n_train, n_val, STAGING_DIR
        )
        del labels
        gc.collect()

    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}\n")

    # 2. Train
    booster = train_model(X_train, y_train, X_val, y_val, il_val, feature_cols)

    # 3. Evaluate
    pr_auc = evaluate(booster, X_val, y_val, il_val, feature_cols)

    # 4. Save model
    booster.save_model(MODEL_OUT_PATH)
    print(f"\n  Model saved → {MODEL_OUT_PATH}")

    # 5. Score test set
    score_test(booster, feature_cols, SUBMISSION_PATH)

    print(f"\n{'─' * 60}")
    print(f"Total wall time : {_fmt(time.perf_counter() - total_start)}")
    print(f"Final PR-AUC    : {pr_auc:.6f}")
    print("─" * 60)
