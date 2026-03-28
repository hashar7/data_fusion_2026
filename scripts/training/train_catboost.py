"""CatBoostClassifier training with optional negative undersampling."""
import gc
import time

import numpy as np
from catboost import CatBoostClassifier

from scripts.training.config import CATBOOST_PARAMS, UNDERSAMPLE_SEED, NEG_SAMPLE_RATIO
from scripts.training._utils import _fmt


def train_catboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    il_val: np.ndarray,
    feature_cols: list,
    train_row_mask=None,
    neg_sample_ratio: float | None = None,
    catboost_params: dict | None = None,
    seed_override: int | None = None,
    retrain_extra: tuple[np.ndarray, np.ndarray] | None = None,
    n_rounds_fixed: int | None = None,
) -> CatBoostClassifier:
    """
    Train a CatBoostClassifier binary classifier.

    Interface mirrors train_model(): X_val / y_val / il_val are the labeled val
    rows only (caller typically passes il_val = all-ones for the filtered subset).

    train_row_mask : optional boolean array — restricts X_train to this group.
    retrain_extra  : optional (X_extra, y_extra) to append after undersampling.
    n_rounds_fixed : when set, train for exactly this many iterations with no
                     early stopping (used for full-data retraining).
    """
    rng = np.random.default_rng(seed_override if seed_override is not None else UNDERSAMPLE_SEED)
    if neg_sample_ratio is None:
        neg_sample_ratio = NEG_SAMPLE_RATIO

    # Work on a copy so we can pop keys without mutating the config dict.
    params = {**(catboost_params if catboost_params is not None else CATBOOST_PARAMS)}
    if seed_override is not None:
        params = {**params, "random_seed": seed_override}
    es_rounds   = params.pop("early_stopping_rounds", 100)
    verbose_lvl = params.pop("verbose", 100)

    # ── Select and undersample training rows ──────────────────────────────────
    y_full = np.asarray(y_train)
    if train_row_mask is not None:
        group_global_idx = np.where(np.asarray(train_row_mask))[0]
        y_group = y_full[group_global_idx]
    else:
        group_global_idx = None
        y_group = y_full

    pos_local = np.where(y_group == 1)[0]
    neg_local = np.where(y_group == 0)[0]

    n_neg_keep  = max(int(len(neg_local) * neg_sample_ratio), len(pos_local))
    neg_sampled = rng.choice(neg_local, size=n_neg_keep, replace=False)
    neg_sampled.sort()
    sample_local = np.sort(np.concatenate([pos_local, neg_sampled]))

    print(f"  CatBoost undersampling: {len(pos_local):,} pos + {len(neg_sampled):,} neg "
          f"({len(sample_local):,} total)", flush=True)

    if group_global_idx is not None:
        sample_global = group_global_idx[sample_local]
        sample_global.sort()
    else:
        sample_global = sample_local

    X_train_used = np.array(X_train[sample_global])
    y_train_used = y_group[sample_local]

    del y_full, y_group, pos_local, neg_local, neg_sampled, sample_local, sample_global
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

    use_early_stopping = n_rounds_fixed is None

    # Override iterations when doing fixed-round retraining
    if n_rounds_fixed is not None:
        params["iterations"] = n_rounds_fixed

    # ── Labeled val rows for early stopping ───────────────────────────────────
    if use_early_stopping:
        labeled_mask  = np.asarray(il_val) == 1
        X_val_labeled = np.array(X_val[labeled_mask])
        y_val_labeled = np.array(y_val[labeled_mask])
        print(f"  CatBoost labeled val  : {labeled_mask.sum():,}", flush=True)
        del labeled_mask
        gc.collect()
    else:
        print(f"  Fixed-round training: {n_rounds_fixed} iterations (no early stopping)",
              flush=True)

    # ── Sample weights: correct for undersampling bias ────────────────────────
    train_weights = np.ones(len(y_train_used), dtype=np.float32)
    train_weights[y_train_used == 0] = 1.0 / neg_sample_ratio

    # ── Build and fit ──────────────────────────────────────────────────────────
    model = CatBoostClassifier(**params)

    print("Training CatBoost …", flush=True)
    for k, v in params.items():
        print(f"  {k:25s}: {v}")
    print(flush=True)

    t0 = time.perf_counter()
    if use_early_stopping:
        model.fit(
            X_train_used, y_train_used,
            sample_weight=train_weights,
            eval_set=(X_val_labeled, y_val_labeled),
            early_stopping_rounds=es_rounds,
            verbose=verbose_lvl,
        )
        del X_val_labeled, y_val_labeled
    else:
        model.fit(
            X_train_used, y_train_used,
            sample_weight=train_weights,
            verbose=verbose_lvl,
        )
    print(f"\n  CatBoost training time : {_fmt(time.perf_counter() - t0)}")
    print(f"  Best iteration         : {model.best_iteration_}\n")

    del X_train_used, y_train_used
    gc.collect()
    return model
