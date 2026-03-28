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
    train_row_mask: np.ndarray | None = None,
    neg_sample_ratio: float | None = None,
    lgbm_params: dict | None = None,
    early_stopping_rounds: int | None = None,
    seed_override: int | None = None,
    retrain_extra: tuple[np.ndarray, np.ndarray] | None = None,
    n_rounds_fixed: int | None = None,
    il_train: np.ndarray | None = None,
    yellow_weight_multiplier: float = 1.0,
) -> lgb.Booster:
    """
    Train a LightGBM binary classifier.

    X_train / y_train : full train split memmaps (or any array).
    X_val / y_val     : val rows used for early stopping.
                        If il_val is all-ones, all rows are labeled.
    il_val            : int8 mask — 1 = row has a real label.
    train_row_mask    : optional boolean array (len = len(y_train)).
                        When provided, only rows where mask is True are used
                        for training. Group filtering + undersampling are
                        applied jointly so only the final sample is loaded
                        into RAM from the memmap.
    retrain_extra     : optional (X_extra, y_extra) arrays to append to
                        training data after undersampling. Used for
                        full-data retraining (appending labeled-val rows).
    n_rounds_fixed    : when set, train for exactly this many rounds with
                        no early stopping. Used for full-data retraining.
    """
    rng = np.random.default_rng(seed_override if seed_override is not None else UNDERSAMPLE_SEED)
    if neg_sample_ratio is None:
        neg_sample_ratio = NEG_SAMPLE_RATIO
    base_params = lgbm_params if lgbm_params is not None else LGBM_PARAMS
    es_rounds   = early_stopping_rounds if early_stopping_rounds is not None else EARLY_STOPPING_ROUNDS

    # ── Select group rows (or all rows) ──────────────────────────────────────
    # y is always small (Int8), so loading it for the full train split is fine.
    y_full = np.asarray(y_train)
    if train_row_mask is not None:
        # group_global_idx: global indices of rows in this group (sorted)
        group_global_idx = np.where(np.asarray(train_row_mask))[0]
        y_group = y_full[group_global_idx]
    else:
        group_global_idx = None
        y_group = y_full

    pos_local = np.where(y_group == 1)[0]
    neg_local = np.where(y_group == 0)[0]

    if neg_sample_ratio is not None:
        n_neg_keep = max(int(len(neg_local) * neg_sample_ratio), len(pos_local))
        neg_sampled = rng.choice(neg_local, size=n_neg_keep, replace=False)
        neg_sampled.sort()
        sample_local = np.sort(np.concatenate([pos_local, neg_sampled]))

        n_pos = len(pos_local)
        n_neg = len(neg_sampled)
        print(f"  Undersampling: kept {n_pos:,} pos + {n_neg:,} neg "
              f"(1:{n_neg/max(n_pos,1):.0f} ratio, "
              f"{len(sample_local):,} total rows)", flush=True)
        params = {**base_params, "is_unbalance": False}
    else:
        sample_local = np.arange(len(y_group))
        print(f"  No undersampling — using {len(y_group):,} rows", flush=True)
        params = {**base_params, "is_unbalance": True}

    # Override random seed for ensemble diversity
    if seed_override is not None:
        params = {**params, "seed": seed_override}

    # Map local indices back to global X_train indices
    if group_global_idx is not None:
        sample_global = group_global_idx[sample_local]
        sample_global.sort()   # sequential memmap reads
    else:
        sample_global = sample_local

    # Load only the final sampled subset into RAM
    X_train_used = np.array(X_train[sample_global])
    y_train_used = y_group[sample_local]

    # Compute yellow mask BEFORE releasing group arrays.
    # Yellow rows = labeled negatives (y=0 AND il_train=1).  They carry clean signal
    # and deserve a higher gradient weight than unlabeled green open-loop transactions.
    yellow_sample_mask = None
    if il_train is not None and yellow_weight_multiplier > 1.0 and neg_sample_ratio is not None:
        il_full_np = np.asarray(il_train)
        if group_global_idx is not None:
            il_group = il_full_np[group_global_idx]
        else:
            il_group = il_full_np
        il_sampled = il_group[sample_local]
        yellow_sample_mask = (y_group[sample_local] == 0) & (il_sampled == 1)
        del il_group, il_sampled, il_full_np

    del y_full, y_group, pos_local, neg_local, sample_local, sample_global
    if group_global_idx is not None:
        del group_global_idx
    gc.collect()

    # ── Append extra rows (full-data retraining) ──────────────────────────────
    if retrain_extra is not None:
        X_extra, y_extra = retrain_extra
        print(f"  Appending {len(y_extra):,} extra rows (retrain_extra)", flush=True)
        X_train_used = np.concatenate([X_train_used, X_extra], axis=0)
        y_train_used = np.concatenate([y_train_used, y_extra], axis=0)
        del X_extra, y_extra
        gc.collect()

    num_rounds = n_rounds_fixed if n_rounds_fixed is not None else base_params.get("n_estimators", LGBM_PARAMS["n_estimators"])
    use_early_stopping = n_rounds_fixed is None

    # ── Labeled val rows for early stopping ───────────────────────────────────
    if use_early_stopping:
        labeled_mask  = np.asarray(il_val) == 1
        X_val_labeled = np.array(X_val[labeled_mask])
        y_val_labeled = np.array(y_val[labeled_mask])
        print(f"  Labeled val rows for early stopping: {labeled_mask.sum():,} / {len(il_val):,}\n",
              flush=True)
        del labeled_mask
        gc.collect()
    else:
        print(f"  Fixed-round training: {num_rounds} rounds (no early stopping)\n",
              flush=True)

    print("Training LightGBM …", flush=True)
    for k, v in params.items():
        print(f"  {k:25s}: {v}")
    print(flush=True)

    # ── Sample weights: correct for undersampling bias + up-weight yellow rows ─
    # Retained negatives are upweighted to 1/neg_sample_ratio so that the loss
    # gradient landscape matches the true class distribution.  Positives keep
    # weight = 1.  When no undersampling is used all weights are 1.
    # Yellow rows (labeled negatives with a real 0-label) are explicitly confirmed
    # non-fraud and carry cleaner signal than unlabeled green rows, so we scale
    # their weight further by yellow_weight_multiplier.
    if neg_sample_ratio is not None:
        weights = np.ones(len(y_train_used), dtype=np.float32)
        weights[y_train_used == 0] = 1.0 / neg_sample_ratio
        if yellow_sample_mask is not None and yellow_sample_mask.any():
            weights[yellow_sample_mask] *= yellow_weight_multiplier
            print(f"  Yellow rows in sample : {yellow_sample_mask.sum():,} "
                  f"→ weight × {yellow_weight_multiplier:.1f}", flush=True)
    else:
        weights = None

    free_train = neg_sample_ratio is not None
    dtrain = lgb.Dataset(
        X_train_used, label=y_train_used,
        weight=weights,
        feature_name=feature_cols,
        free_raw_data=free_train,
    )

    if use_early_stopping:
        dval = lgb.Dataset(
            X_val_labeled, label=y_val_labeled,
            feature_name=feature_cols,
            reference=dtrain,
            free_raw_data=True,
        )
        del X_val_labeled, y_val_labeled
        valid_sets  = [dval]
        valid_names = ["val"]
        callbacks   = [
            lgb.early_stopping(stopping_rounds=es_rounds, verbose=True),
            lgb.log_evaluation(period=LOG_EVAL_PERIOD),
        ]
    else:
        valid_sets  = []
        valid_names = []
        callbacks   = [lgb.log_evaluation(period=LOG_EVAL_PERIOD)]

    if free_train:
        del X_train_used, y_train_used
    gc.collect()

    t0 = time.perf_counter()
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    print(f"\n  Training time   : {_fmt(time.perf_counter() - t0)}")
    print(f"  Best iteration  : {booster.best_iteration}")
    if use_early_stopping:
        print(f"  Best val PR-AUC : "
              f"{booster.best_score['val']['average_precision']:.6f}\n")
    else:
        print()
    return booster
