"""Top-level orchestrator: trains one model per tx_type_group."""
import gc
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import average_precision_score

from scripts.training.config import (
    FEATURES_DIR, STAGING_DIR, MODEL_OUT_PATHS, SUBMISSION_PATH,
    VAL_CUTOFF_DATE, TRAIN_END_DATE, TX_TYPE_GROUPS,
)
from scripts.training._utils import _fmt, _parquet_files
from scripts.training.data import load_labels, count_rows, build_memmaps, _load_cache
from scripts.training.train import train_model
from scripts.training.evaluate import evaluate
from scripts.training.predict import score_test


def train_baseline() -> None:
    """
    Train one LightGBM model per tx_type_group (non-payment / card / P2P),
    evaluate each model independently, compute the combined PR-AUC, save all
    three models, then score the test set using the appropriate model per row.
    """
    total_start = time.perf_counter()
    cutoff    = datetime.fromisoformat(VAL_CUTOFF_DATE)
    train_end = datetime.fromisoformat(TRAIN_END_DATE)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")
    print(f"Found {len(files)} parquet partitions in {FEATURES_DIR!r}\n", flush=True)

    # ── Step 1: Build or reload memmap split files ────────────────────────────
    cached = _load_cache(STAGING_DIR)
    if cached is not None:
        X_train, y_train, X_val, y_val, il_val, tg_train, tg_val, feature_cols = cached
    else:
        labels = load_labels()
        n_train, n_val = count_rows(files, cutoff, train_end)
        (X_train, y_train, X_val, y_val, il_val,
         tg_train, tg_val, feature_cols) = build_memmaps(
            files, labels, cutoff, train_end, n_train, n_val, STAGING_DIR
        )
        del labels
        gc.collect()

    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}\n")

    # Convert tx_type_group arrays to plain numpy (small, Int8)
    tg_train_np = np.asarray(tg_train)
    tg_val_np   = np.asarray(tg_val)
    il_val_np   = np.asarray(il_val)
    y_val_np    = np.asarray(y_val)

    # Accumulate val scores per row across models for the combined metric
    all_val_scores = np.full(len(y_val_np), np.nan, dtype=np.float32)

    boosters: dict = {}
    pr_aucs:  dict = {}

    # ── Step 2: Train and evaluate one model per group ────────────────────────
    for group_id, group_name in TX_TYPE_GROUPS.items():
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  Group {group_id} — {group_name.upper()}  model")
        print(f"{sep}\n")

        train_mask = tg_train_np == group_id
        val_mask   = tg_val_np   == group_id
        n_train_g  = int(train_mask.sum())
        n_val_g    = int(val_mask.sum())

        # Labeled val rows for this group — used for early stopping.
        # This set is tiny (few thousand rows at most) so loading to RAM is fine.
        labeled_val_mask = val_mask & (il_val_np == 1)
        n_labeled_g = int(labeled_val_mask.sum())
        n_pos_train_g = int(y_train[train_mask].sum()) if n_train_g > 0 else 0
        n_pos_val_g   = int(y_val_np[labeled_val_mask].sum()) if n_labeled_g > 0 else 0

        print(f"  Train rows  : {n_train_g:,}  ({n_pos_train_g:,} positives)")
        print(f"  Val rows    : {n_val_g:,}  ({n_labeled_g:,} labeled,  "
              f"{n_pos_val_g:,} positives)\n")

        if n_train_g == 0:
            print(f"  SKIP: no training rows for group {group_id}.\n")
            continue
        if n_pos_train_g == 0:
            print(f"  SKIP: no positive training rows for group {group_id}.\n")
            continue
        if n_labeled_g == 0 or n_pos_val_g == 0:
            print(f"  WARNING: no labeled/positive val rows for group {group_id}. "
                  "Early stopping disabled for this group.\n")

        # Extract labeled val rows for early stopping (tiny subset → in RAM)
        labeled_val_idx = np.where(labeled_val_mask)[0]
        X_val_labeled_g = np.array(X_val[labeled_val_idx])
        y_val_labeled_g = np.array(y_val[labeled_val_idx])
        il_val_ones_g   = np.ones(len(labeled_val_idx), dtype=np.int8)

        # Train — train_row_mask restricts X_train to this group;
        # undersampling is applied within the group to keep RAM usage low.
        booster = train_model(
            X_train, y_train,
            X_val_labeled_g, y_val_labeled_g, il_val_ones_g,
            feature_cols,
            train_row_mask=train_mask,
        )
        del X_val_labeled_g, y_val_labeled_g, il_val_ones_g
        gc.collect()

        # Per-group validation evaluation (chunks over val rows of this group)
        pr_auc = evaluate(
            booster, X_val, y_val_np, il_val_np,
            feature_cols,
            row_mask=val_mask,
            label=group_name,
        )
        pr_aucs[group_id] = pr_auc

        # Collect val scores for combined metric (process in chunks)
        val_idx_g  = np.where(val_mask)[0]
        chunk_size = 500_000
        for cs in range(0, len(val_idx_g), chunk_size):
            ce  = min(cs + chunk_size, len(val_idx_g))
            idx = val_idx_g[cs:ce]
            X_c = np.array(X_val[idx])
            s   = booster.predict(X_c, num_iteration=booster.best_iteration).astype(np.float32)
            all_val_scores[idx] = s
            del X_c, s
            gc.collect()

        # Save model
        model_path = MODEL_OUT_PATHS[group_id]
        booster.save_model(model_path)
        print(f"\n  Model saved → {model_path}")
        boosters[group_id] = booster

    # ── Step 3: Combined PR-AUC ───────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Combined performance  (all groups, labeled val rows)")
    print(f"{'=' * 65}\n")

    il_labeled = il_val_np == 1
    scored_mask = il_labeled & ~np.isnan(all_val_scores)
    if scored_mask.sum() > 0 and y_val_np[scored_mask].sum() > 0:
        combined_pr_auc = average_precision_score(
            y_val_np[scored_mask], all_val_scores[scored_mask]
        )
        print(f"  Combined PR-AUC (labeled val, competition metric): {combined_pr_auc:.6f}")
    else:
        combined_pr_auc = 0.0
        print("  WARNING: could not compute combined PR-AUC.")

    print()
    for g, name in TX_TYPE_GROUPS.items():
        if g in pr_aucs:
            print(f"  {name:12s} model PR-AUC : {pr_aucs[g]:.6f}")

    # ── Step 4: Score test set ────────────────────────────────────────────────
    if boosters:
        score_test(boosters, feature_cols, SUBMISSION_PATH)
    else:
        print("\nWARNING: no models were trained — submission skipped.")

    print(f"\n{'─' * 65}")
    print(f"Total wall time  : {_fmt(time.perf_counter() - total_start)}")
    print(f"Combined PR-AUC  : {combined_pr_auc:.6f}")
    print("─" * 65)
