"""Top-level orchestrator: trains one ensemble per tx_type_group."""
import gc
import glob
import os
import re
import time
from datetime import datetime
import joblib

import numpy as np
from sklearn.metrics import average_precision_score

# ── tx_type_group based models ────────────────────────────────────────────────––––
from scripts.training.config import (
    FEATURES_DIR, STAGING_DIR, SUBMISSION_PATH, MODELS_DIR,
    VAL_CUTOFF_DATE, TRAIN_END_DATE, TX_TYPE_GROUPS,
    NEG_SAMPLE_RATIO, NEG_SAMPLE_RATIO_BY_GROUP,
    LGBM_PARAMS, LGBM_PARAMS_BY_GROUP,
    EARLY_STOPPING_ROUNDS_BY_GROUP,
    ENSEMBLE_SEEDS,
    CATBOOST_PARAMS, CATBOOST_PARAMS_BY_GROUP, CATBOOST_BLEND_WEIGHT,
    LGBM_MODEL_PATH_FMT, CATBOOST_MODEL_PATH_FMT,
)
from scripts.training._utils import _fmt, _parquet_files, _progress
from scripts.training.data import load_labels, count_rows, build_memmaps, _load_cache
from scripts.training.train import train_model
from scripts.training.train_catboost import train_catboost_model
from scripts.training.evaluate import evaluate
from scripts.training.predict import score_test

# ── F vs G Random Forrest ────────────────────────────────────────────────–––––––––
from scripts.training.train_rf import (
    build_labeled_train_indices,
    tune_rf_hyperparameters,
    train_rf_model,
    sanitize_rf_features,
)
from scripts.training.config import (
    RF_MODEL_FILENAME,
)


def _prepare_models_dir(models_dir: str) -> None:
    """
    Create models_dir if it does not exist.
    If unversioned model files (model_*.txt / model_*.cbm) are already present,
    rename them all with a _ver_N suffix so the new run's files don't overwrite them.
    All files from the same previous run share the same version number.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Collect unversioned model files — anything that does NOT already have _ver_N
    all_files = (
        glob.glob(os.path.join(models_dir, "model_*.txt")) +
        glob.glob(os.path.join(models_dir, "model_*.cbm")) +
        glob.glob(os.path.join(models_dir, "model_*.pkl"))
    )
    unversioned = [
        f for f in all_files
        if not re.search(r"_ver_\d+\.(txt|cbm|pkl)$", f)
    ]

    if not unversioned:
        return

    # Find the highest existing version number so we don't collide
    versioned = (
        glob.glob(os.path.join(models_dir, "model_*_ver_*.txt")) +
        glob.glob(os.path.join(models_dir, "model_*_ver_*.cbm")) +
        glob.glob(os.path.join(models_dir, "model_*_ver_*.pkl"))
    )
    max_ver = 0
    for f in versioned:
        m = re.search(r"_ver_(\d+)\.(txt|cbm|pkl)$", f)
        if m:
            max_ver = max(max_ver, int(m.group(1)))

    next_ver = max_ver + 1
    print(f"  Found {len(unversioned)} existing model file(s) - "
          f"archiving as _ver_{next_ver} ...")
    for f in sorted(unversioned):
        base, ext = os.path.splitext(f)
        dest = f"{base}_ver_{next_ver}{ext}"
        os.rename(f, dest)
        print(f"    {os.path.basename(f)}  ->  {os.path.basename(dest)}")
    print()


def _score_val_group(
    model,
    X_val: np.ndarray,
    global_indices: np.ndarray,
    chunk_size: int = 500_000,
    is_catboost: bool = False,
    suffix: str = "scoring val",
) -> np.ndarray:
    """
    Score a subset of val rows (identified by global_indices) without loading
    the full val split into RAM.  Returns float32 scores array of len(global_indices).
    """
    n = len(global_indices)
    scores = np.empty(n, dtype=np.float32)
    n_chunks   = max(1, (n + chunk_size - 1) // chunk_size)
    chunk_times: list = []
    for chunk_i, cs in enumerate(range(0, n, chunk_size)):
        t0  = time.perf_counter()
        ce  = min(cs + chunk_size, n)
        idx = global_indices[cs:ce]
        X_c = np.array(X_val[idx])
        if is_catboost:
            s = model.predict_proba(X_c)[:, 1].astype(np.float32)
        else:
            s = model.predict(X_c, num_iteration=model.best_iteration).astype(np.float32)
        scores[cs:ce] = s
        del X_c, s
        gc.collect()
        chunk_times.append(time.perf_counter() - t0)
        _progress(chunk_i + 1, n_chunks, chunk_times, suffix=suffix)
    print()
    return scores


def train_baseline() -> None:
    """
    Train one LightGBM ensemble (multi-seed) + one CatBoost per tx_type_group,
    blend and calibrate scores, evaluate each group independently, compute the
    combined PR-AUC, save all models, then score the test set.

    Pipeline per group:
        1. Train LGBM N times with different seeds → average val scores
        2. Train CatBoost → blend with LGBM avg
        3. Fit Platt calibrator on labeled val rows → calibrate blended scores
        4. Compute ensemble PR-AUC on calibrated scores
    """
    total_start = time.perf_counter()
    cutoff    = datetime.fromisoformat(VAL_CUTOFF_DATE)
    train_end = datetime.fromisoformat(TRAIN_END_DATE)

    _prepare_models_dir(MODELS_DIR)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")
    print(f"Found {len(files)} parquet partitions in {FEATURES_DIR!r}\n", flush=True)
    print(f"Ensemble seeds   : {ENSEMBLE_SEEDS}")
    print(f"CatBoost weight  : {CATBOOST_BLEND_WEIGHT}\n", flush=True)

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

    tg_train_np = np.asarray(tg_train)
    tg_val_np   = np.asarray(tg_val)
    il_val_np   = np.asarray(il_val)
    y_val_np    = np.asarray(y_val)

    all_val_scores = np.full(len(y_val_np), np.nan, dtype=np.float32)

    lgbm_boosters:   dict = {}   # group_id → list[lgb.Booster]
    catboost_models: dict = {}   # group_id → CatBoostClassifier
    pr_aucs:         dict = {}

    # ── Step 2: Train and evaluate one ensemble per group ────────────────────
    for group_id, group_name in TX_TYPE_GROUPS.items():
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  Group {group_id} — {group_name.upper()}  ensemble")
        print(f"{sep}\n")

        train_mask = tg_train_np == group_id
        val_mask   = tg_val_np   == group_id
        n_train_g  = int(train_mask.sum())
        n_val_g    = int(val_mask.sum())

        labeled_val_mask = val_mask & (il_val_np == 1)
        n_labeled_g  = int(labeled_val_mask.sum())
        n_pos_train_g = int(y_train[train_mask].sum()) if n_train_g > 0 else 0
        n_pos_val_g   = int(y_val_np[labeled_val_mask].sum()) if n_labeled_g > 0 else 0

        print(f"  Train rows  : {n_train_g:,}  ({n_pos_train_g:,} positives)")
        print(f"  Val rows    : {n_val_g:,}  ({n_labeled_g:,} labeled,  "
              f"{n_pos_val_g:,} positives)\n")

        if n_train_g == 0 or n_pos_train_g == 0:
            print(f"  SKIP: no training data for group {group_id}.\n")
            continue
        if n_labeled_g == 0 or n_pos_val_g == 0:
            print(f"  WARNING: no labeled/positive val rows for group {group_id}.\n")

        # Pre-extract labeled val subset for early stopping (tiny, fits in RAM).
        labeled_val_idx = np.where(labeled_val_mask)[0]
        X_val_labeled_g = np.array(X_val[labeled_val_idx])
        y_val_labeled_g = np.array(y_val[labeled_val_idx])
        il_val_ones_g   = np.ones(len(labeled_val_idx), dtype=np.int8)

        val_idx_g       = np.where(val_mask)[0]   # global indices in X_val
        group_neg_ratio = NEG_SAMPLE_RATIO_BY_GROUP.get(group_id, NEG_SAMPLE_RATIO)
        group_lgbm_params = {**LGBM_PARAMS, **LGBM_PARAMS_BY_GROUP.get(group_id, {})}
        group_es_rounds   = EARLY_STOPPING_ROUNDS_BY_GROUP.get(group_id, None)
        group_cb_params   = {**CATBOOST_PARAMS, **CATBOOST_PARAMS_BY_GROUP.get(group_id, {})}

        # ── 2a. Multi-seed LightGBM ──────────────────────────────────────────
        lgbm_val_sum  = np.zeros(len(val_idx_g), dtype=np.float64)
        seed_boosters: list = []

        for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
            print(f"\n  ── LGBM seed {seed_idx + 1}/{len(ENSEMBLE_SEEDS)} "
                  f"(seed={seed}) ──────────────────────────────────")
            booster = train_model(
                X_train, y_train,
                X_val_labeled_g, y_val_labeled_g, il_val_ones_g,
                feature_cols,
                train_row_mask=train_mask,
                neg_sample_ratio=group_neg_ratio,
                lgbm_params=group_lgbm_params,
                early_stopping_rounds=group_es_rounds,
                seed_override=seed,
            )
            path = os.path.join(MODELS_DIR, LGBM_MODEL_PATH_FMT.format(name=group_name, seed_idx=seed_idx))
            booster.save_model(path)
            print(f"  LGBM model saved → {path}")

            # Diagnostic evaluation (feature importance) on first seed only.
            if seed_idx == 0:
                evaluate(
                    booster, X_val, y_val_np, il_val_np, feature_cols,
                    row_mask=val_mask, label=f"{group_name} seed-0",
                )

            # Accumulate val scores (chunked, leakage-free).
            print(f"  Scoring val rows (LGBM seed {seed_idx}) …", flush=True)
            s = _score_val_group(
                booster, X_val, val_idx_g,
                suffix=f"lgbm s{seed_idx} {group_name}",
            )
            lgbm_val_sum += s.astype(np.float64)
            seed_boosters.append(booster)
            del s
            gc.collect()

        lgbm_boosters[group_id] = seed_boosters
        lgbm_val_scores = (lgbm_val_sum / len(ENSEMBLE_SEEDS)).astype(np.float32)
        del lgbm_val_sum
        gc.collect()

        # ── 2b. CatBoost ────────────────────────────────────────────────────
        print(f"\n  ── CatBoost ────────────────────────────────────────────────")
        cb_model = train_catboost_model(
            X_train, y_train,
            X_val_labeled_g, y_val_labeled_g, il_val_ones_g,
            feature_cols,
            train_row_mask=train_mask,
            neg_sample_ratio=group_neg_ratio,
            catboost_params=group_cb_params,
        )
        cb_path = os.path.join(MODELS_DIR, CATBOOST_MODEL_PATH_FMT.format(name=group_name))
        cb_model.save_model(cb_path)
        print(f"  CatBoost model saved → {cb_path}")
        catboost_models[group_id] = cb_model

        print(f"  Scoring val rows (CatBoost) …", flush=True)
        cb_val_scores = _score_val_group(
            cb_model, X_val, val_idx_g,
            is_catboost=True, suffix=f"catboost {group_name}",
        )

        del X_val_labeled_g, y_val_labeled_g, il_val_ones_g
        gc.collect()

        # ── 2c. Blend ────────────────────────────────────────────────────────
        w = CATBOOST_BLEND_WEIGHT
        blended_scores = (
            lgbm_val_scores * (1.0 - w) + cb_val_scores * w
        ).astype(np.float32)
        del lgbm_val_scores, cb_val_scores
        gc.collect()

        # ── 2d. Per-group ensemble PR-AUC ────────────────────────────────────
        # Raw blended scores are used directly — no Platt calibration.
        # Calibration on labeled val rows (50% positive) would inflate test scores
        # to 0.35-0.5 for every row, as the calibrated prior doesn't match the
        # true 0.06% positive rate in the competition.
        il_g          = il_val_np[val_idx_g]
        labeled_local = np.where(il_g == 1)[0]
        y_labeled     = y_val_np[val_idx_g][labeled_local]
        blended_lbl   = blended_scores[labeled_local]

        if len(labeled_local) > 0 and y_labeled.sum() > 0:
            pr_auc_ens = average_precision_score(y_labeled, blended_lbl)
            print(f"\n  Ensemble PR-AUC [{group_name}]: {pr_auc_ens:.6f}")
            pr_aucs[group_id] = pr_auc_ens

        # Store blended scores in the combined array for joint metric.
        all_val_scores[val_idx_g] = blended_scores

        del blended_scores, il_g, labeled_local, y_labeled, blended_lbl
        gc.collect()

    # ── Step 3: Combined PR-AUC ───────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  Combined performance  (all groups, labeled val rows)")
    print(f"{'=' * 65}\n")

    il_labeled  = il_val_np == 1
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
            print(f"  {name:12s} ensemble PR-AUC : {pr_aucs[g]:.6f}")

    # ── Step 4: Score test set ────────────────────────────────────────────────
    if lgbm_boosters:
        score_test(lgbm_boosters, catboost_models, feature_cols, SUBMISSION_PATH)
    else:
        print("\nWARNING: no models were trained — submission skipped.")

    print(f"\n{'─' * 65}")
    print(f"Total wall time  : {_fmt(time.perf_counter() - total_start)}")
    print(f"Combined PR-AUC  : {combined_pr_auc:.6f}")
    print("─" * 65)


def train_rf_fg() -> None:
    """
    Train RandomForest on labeled F/G rows only.

    Reuses existing memmaps:
        X_train, y_train, X_val, y_val, is_labeled_val, memmap_meta

    Steps:
        1. Reload or build memmaps
        2. Reconstruct labeled-train row indices by rescanning processed parquet partitions
        3. Load only labeled train/val rows into RAM
        4. Tune RF with 6-fold CV on labeled-train rows
        5. Fit best RF on labeled-train rows and evaluate on labeled val rows
        6. Refit RF on labeled train + labeled val rows
        7. Report optimistic PR-AUC on val after refit
        8. Save final model bundle to MODELS_DIR
    """
    total_start = time.perf_counter()
    cutoff = datetime.fromisoformat(VAL_CUTOFF_DATE)
    train_end = datetime.fromisoformat(TRAIN_END_DATE)

    # _prepare_models_dir(MODELS_DIR)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")
    print(f"Found {len(files)} parquet partitions in {FEATURES_DIR!r}\n", flush=True)

    cached = _load_cache(STAGING_DIR)
    if cached is not None:
        X_train, y_train, X_val, y_val, il_val, tg_train, tg_val, feature_cols = cached
    else:
        labels = load_labels()
        n_train, n_val = count_rows(files, cutoff, train_end)
        (
            X_train, y_train, X_val, y_val, il_val,
            tg_train, tg_val, feature_cols
        ) = build_memmaps(
            files, labels, cutoff, train_end, n_train, n_val, STAGING_DIR
        )
        del labels
        gc.collect()

    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}\n")

    # We need original labels again to identify F/G rows inside train.
    labels = load_labels()

    # ── Build train labeled mapping (F/G only) ──────────────────────────────
    train_labeled_idx = build_labeled_train_indices(files, labels, cutoff)
    if len(train_labeled_idx) == 0:
        raise RuntimeError("No labeled train rows found for RF training.")

    # ── Val labeled mapping already exists in cache as il_val ───────────────
    il_val_np = np.asarray(il_val)
    val_labeled_idx = np.flatnonzero(il_val_np == 1)

    if len(val_labeled_idx) == 0:
        raise RuntimeError("No labeled val rows found for RF validation.")

    print("Loading labeled train/val subsets into RAM …", flush=True)

    X_train_fg = np.asarray(X_train[train_labeled_idx], dtype=np.float32)
    y_train_fg = np.asarray(y_train[train_labeled_idx], dtype=np.int8)

    X_val_fg = np.asarray(X_val[val_labeled_idx], dtype=np.float32)
    y_val_fg = np.asarray(y_val[val_labeled_idx], dtype=np.int8)

    X_train_fg = sanitize_rf_features(
        X_train_fg,
        feature_cols=feature_cols,
        stage="X_train_fg",
    )
    X_val_fg = sanitize_rf_features(
        X_val_fg,
        feature_cols=feature_cols,
        stage="X_val_fg",
    )

    print(f"  Train labeled rows : {len(y_train_fg):,}")
    print(f"    positives (F)    : {int(y_train_fg.sum()):,}")
    print(f"    negatives (G)    : {int((y_train_fg == 0).sum()):,}")
    print(f"  Val labeled rows   : {len(y_val_fg):,}")
    print(f"    positives (F)    : {int(y_val_fg.sum()):,}")
    print(f"    negatives (G)    : {int((y_val_fg == 0).sum()):,}\n")

    # ── CV tuning on train only ─────────────────────────────────────────────
    best_params, cv_results = tune_rf_hyperparameters(
        X_train_fg,
        y_train_fg,
    )

    # ── Best model on train only → true holdout val metric ──────────────────
    print("Training best RF on labeled train rows …", flush=True)
    best_train_model = train_rf_model(
        X_train_fg,
        y_train_fg,
        rf_params=best_params,
    )

    val_scores = best_train_model.predict_proba(X_val_fg)[:, 1].astype(np.float32)
    val_pr_auc = average_precision_score(y_val_fg, val_scores)

    print(f"\nHoldout val PR-AUC (best RF trained on train only): {val_pr_auc:.6f}\n", flush=True)

    # ── Refit on train + val labeled rows ───────────────────────────────────
    print("Refitting RF on labeled train + labeled val rows …", flush=True)
    X_full_fg = np.concatenate([X_train_fg, X_val_fg], axis=0)
    y_full_fg = np.concatenate([y_train_fg, y_val_fg], axis=0)

    X_full_fg = sanitize_rf_features(
        X_full_fg,
        feature_cols=feature_cols,
        stage="X_full_fg",
    )

    final_model = train_rf_model(
        X_full_fg,
        y_full_fg,
        rf_params=best_params,
    )

    final_val_scores = final_model.predict_proba(X_val_fg)[:, 1].astype(np.float32)
    final_val_pr_auc = average_precision_score(y_val_fg, final_val_scores)

    print(
        f"PR-AUC after refit on train+val: {final_val_pr_auc:.6f}",
        flush=True,
    )
    # ── Save final model bundle ──────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, RF_MODEL_FILENAME)
    bundle = {
        "model": final_model,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "cv_results": cv_results,
        "holdout_val_pr_auc": float(val_pr_auc),
        "optimistic_val_pr_auc_after_refit": float(final_val_pr_auc),
    }
    joblib.dump(bundle, model_path)
    print(f"Final RF model bundle saved → {model_path}\n", flush=True)

    del (
        labels,
        train_labeled_idx,
        val_labeled_idx,
        X_train_fg,
        y_train_fg,
        X_val_fg,
        y_val_fg,
        X_full_fg,
        y_full_fg,
        val_scores,
        final_val_scores,
    )
    gc.collect()

    print(f"{'─' * 65}")
    print(f"Total wall time : {_fmt(time.perf_counter() - total_start)}")
    print(f"Holdout val AP  : {val_pr_auc:.6f}")
    print(f"Refit val AP    : {final_val_pr_auc:.6f}")
    print(f"{'─' * 65}")

