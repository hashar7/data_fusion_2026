"""LightGBM training with optional negative undersampling."""
import gc
import time

import lightgbm as lgb
import numpy as np

from scripts.training.config import (
    LGBM_PARAMS, EARLY_STOPPING_ROUNDS, LOG_EVAL_PERIOD,
    NEG_SAMPLE_RATIO, UNDERSAMPLE_SEED,
)
from scripts.training._utils import _fmt


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    il_val: np.ndarray,
    feature_cols: list,
) -> lgb.Booster:
    """
    Build LightGBM Datasets from memmap arrays and train.

    X_train / y_train : full train split (labeled + open-loop rows, label=0)
    X_val / y_val     : full val split   (labeled + open-loop rows, label=0)
    il_val            : int8 mask — 1 = this val row has a real label

    Early stopping uses ONLY labeled val rows (il_val == 1) so that the
    average_precision metric tracked during training is not diluted by the
    millions of open-loop rows assigned a synthetic label=0.
    """
    rng = np.random.default_rng(UNDERSAMPLE_SEED)
    y_train_arr = np.array(y_train)
    pos_idx = np.where(y_train_arr == 1)[0]
    neg_idx = np.where(y_train_arr == 0)[0]

    if NEG_SAMPLE_RATIO is not None:
        n_neg_keep      = max(int(len(neg_idx) * NEG_SAMPLE_RATIO), len(pos_idx))
        neg_idx_sampled = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        neg_idx_sampled.sort()   # sorted → sequential memmap reads (faster)
        train_idx = np.sort(np.concatenate([pos_idx, neg_idx_sampled]))

        n_pos  = len(pos_idx)
        n_neg  = len(neg_idx_sampled)
        print(f"  Undersampling: kept {n_pos:,} pos + {n_neg:,} neg "
              f"(1:{n_neg/max(n_pos,1):.0f} ratio, "
              f"{len(train_idx):,} / {len(y_train_arr):,} total rows)", flush=True)

        # Class ratio already corrected by undersampling — is_unbalance would
        # double-correct and make the model over-aggressive.
        params = {**LGBM_PARAMS, "is_unbalance": False}
        X_train_used = np.array(X_train[train_idx])
        y_train_used = y_train_arr[train_idx]
    else:
        print(f"  No undersampling — using full {len(y_train_arr):,} train rows",
              flush=True)
        params = {**LGBM_PARAMS, "is_unbalance": True}
        X_train_used = X_train
        y_train_used = y_train_arr

    del y_train_arr, pos_idx, neg_idx
    gc.collect()

    # Early stopping: labeled val rows only
    labeled_mask  = np.array(il_val) == 1
    X_val_labeled = np.array(X_val[labeled_mask])
    y_val_labeled = np.array(y_val[labeled_mask])
    print(f"  Labeled val rows for early stopping: {labeled_mask.sum():,} / {len(il_val):,}\n",
          flush=True)
    del labeled_mask
    gc.collect()

    print("Training LightGBM …", flush=True)
    for k, v in params.items():
        print(f"  {k:25s}: {v}")
    print(flush=True)

    free_train = NEG_SAMPLE_RATIO is not None
    dtrain = lgb.Dataset(
        X_train_used, label=y_train_used,
        feature_name=feature_cols,
        free_raw_data=free_train,
    )
    dval = lgb.Dataset(
        X_val_labeled, label=y_val_labeled,
        feature_name=feature_cols,
        reference=dtrain,
        free_raw_data=True,
    )
    del X_val_labeled, y_val_labeled
    if free_train:
        del X_train_used, y_train_used
    gc.collect()

    t0 = time.perf_counter()
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=LOG_EVAL_PERIOD),
        ],
    )

    print(f"\n  Training time   : {_fmt(time.perf_counter() - t0)}")
    print(f"  Best iteration  : {booster.best_iteration}")
    print(f"  Best val PR-AUC : "
          f"{booster.best_score['val']['average_precision']:.6f}\n")
    return booster
