"""Top-level orchestrator: trains one ensemble per tx_type_group."""
import gc
import os
import time
from datetime import datetime
import joblib

import numpy as np
import polars as pl

from sklearn.metrics import average_precision_score

from scripts.training._utils import (
    _fmt, _parquet_files, _progress, _estimate_matrix_ram_gb, _prepare_models_dir
)


# ── tx_type_group based models ────────────────────────────────────────────────––––
import lightgbm as lgb
from scripts.training.config import (
    FEATURES_DIR, STAGING_DIR, SUBMISSION_PATH, MODELS_DIR,
    VAL_CUTOFF_DATE, TRAIN_END_DATE, TX_TYPE_GROUPS,
    NEG_SAMPLE_RATIO, NEG_SAMPLE_RATIO_BY_GROUP,
    LGBM_PARAMS, LGBM_PARAMS_BY_GROUP,
    EARLY_STOPPING_ROUNDS_BY_GROUP,
    ENSEMBLE_SEEDS,
    CATBOOST_PARAMS, CATBOOST_PARAMS_BY_GROUP, CATBOOST_BLEND_WEIGHT,
    LGBM_MODEL_PATH_FMT, CATBOOST_MODEL_PATH_FMT, UNDERSAMPLE_SEED,
)
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

# ── F vs U catboost ────────────────────────────────────────────────–––––––––––––
from scripts.training.config import (
    CATBOOST_FU_MODEL_FILENAME,
)

# ── Final ensemble ────────────────────────────────────────────────–––––––––––––
from catboost import CatBoostClassifier
import lightgbm as lgb
from scripts.training.data import (
    _load_cache,
    build_memmaps,
    count_rows,
    load_labels,
)
from scripts.training.train_catboost import train_catboost_model
from scripts.training.train_rf import sanitize_rf_features
from scripts.training._utils import _load_tx_type_catboost_models, _load_rf_bundle, _load_fu_catboost_model, _load_tx_type_lgbm_models
from scripts.training.predict import score_test_final_ensemble, _score_tx_group_ensemble_chunk
from scripts.training.config import (
    VAL_CUTOFF_DATE, TRAIN_END_DATE,

    MODELS_DIR, FEATURES_DIR, STAGING_DIR, 
    CATBOOST_FU_MODEL_FILENAME, RF_MODEL_FILENAME,
    FINAL_ENSEMBLE_MODEL_FILENAME, 
    SUBMISSION_PATH, 
    LGBM_MODEL_PATH_FMT,

    TX_TYPE_GROUPS, 
    CATBOOST_PARAMS, CATBOOST_BLEND_WEIGHT,
    NEG_SAMPLE_RATIO,
)


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


def _build_fu_train_mask(
    files: list[str],
    labels: pl.DataFrame,
    cutoff: datetime,
) -> np.ndarray:
    """
    Reconstruct a boolean train mask aligned with X_train / y_train memmaps.

    Keep:
        F rows  -> target == 1
        U rows  -> target is null after label join
    Drop:
        G rows  -> target == 0 in original labels
    """
    print("Building F/U train mask from processed parquet partitions …", flush=True)

    labels_small = labels.select(["customer_id", "event_id", "target"])
    mask_parts: list[np.ndarray] = []
    wall_times: list[float] = []

    for i, f in enumerate(files):
        t0 = time.perf_counter()

        chunk = pl.read_parquet(
            f,
            columns=["customer_id", "event_id", "event_dttm", "is_train"],
        )

        if chunk["event_dttm"].dtype == pl.Utf8:
            chunk = chunk.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )

        chunk = chunk.join(
            labels_small,
            on=["customer_id", "event_id"],
            how="left",
        )

        train_chunk = chunk.filter(
            (pl.col("is_train") == 1) & (pl.col("event_dttm") < cutoff)
        )

        fu_mask_chunk = (
            train_chunk["target"].is_null() | (train_chunk["target"] == 1)
        ).to_numpy()

        mask_parts.append(fu_mask_chunk.astype(bool, copy=False))

        del chunk, train_chunk, fu_mask_chunk
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(
            i + 1,
            len(files),
            wall_times,
            suffix=f"assembled train F/U mask rows {sum(len(x) for x in mask_parts):,}",
        )

    print()

    if not mask_parts:
        return np.empty(0, dtype=bool)

    train_fu_mask = np.concatenate(mask_parts).astype(bool, copy=False)
    print(f"  Train F/U mask built: {len(train_fu_mask):,} rows\n", flush=True)
    return train_fu_mask


def _score_meta_features_chunked(
    X_mm: np.ndarray,
    global_indices: np.ndarray,
    model_groups: np.ndarray,
    tx_lgbm_models: dict[int, list[lgb.Booster]],
    tx_cb_models: dict[int, CatBoostClassifier],
    rf_bundle: dict,
    fu_cb_model: CatBoostClassifier,
    chunk_size: int = 250_000,
    suffix: str = "meta scoring",
) -> np.ndarray:
    """
    Build meta-features from previously trained models.

    Meta-feature order:
        0 -> tx_type_group LightGBM average score
        1 -> tx_type_group CatBoost score
        2 -> tx_type_group blended ensemble score
        3 -> RF F/G score
        4 -> CatBoost F/U score
    """
    rf_model = rf_bundle["model"]
    rf_feature_cols = rf_bundle["feature_cols"]

    n = len(global_indices)
    X_meta = np.empty((n, 5), dtype=np.float32)

    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    chunk_times: list[float] = []

    for chunk_i, cs in enumerate(range(0, n, chunk_size)):
        t0 = time.perf_counter()
        ce = min(cs + chunk_size, n)

        idx = global_indices[cs:ce]
        X_c = np.asarray(X_mm[idx], dtype=np.float32)
        mg_c = np.asarray(model_groups[idx], dtype=np.int8)

        tx_lgbm_scores, tx_cb_scores, tx_blend_scores = _score_tx_group_ensemble_chunk(
            X_c,
            mg_c,
            tx_lgbm_models,
            tx_cb_models,
            blend_weight=CATBOOST_BLEND_WEIGHT,
        )

        X_rf = sanitize_rf_features(
            X_c,
            feature_cols=rf_feature_cols,
            stage=f"{suffix} rf chunk {chunk_i}",
        )
        rf_scores = rf_model.predict_proba(X_rf)[:, 1].astype(np.float32)
        fu_scores = fu_cb_model.predict_proba(X_c)[:, 1].astype(np.float32)

        X_meta[cs:ce, 0] = tx_lgbm_scores
        X_meta[cs:ce, 1] = tx_cb_scores
        X_meta[cs:ce, 2] = tx_blend_scores
        X_meta[cs:ce, 3] = rf_scores
        X_meta[cs:ce, 4] = fu_scores

        del (
            X_c,
            X_rf,
            mg_c,
            tx_lgbm_scores,
            tx_cb_scores,
            tx_blend_scores,
            rf_scores,
            fu_scores,
        )
        gc.collect()

        chunk_times.append(time.perf_counter() - t0)
        _progress(chunk_i + 1, n_chunks, chunk_times, suffix=suffix)

    print()
    return X_meta


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


def train_catboost_fu() -> None:
    """
    Train CatBoost on F/U events only.

    Train rows:
        keep F (target=1) and U (not present in original labels)
        drop G (target=0 in original labels)

    Val rows:
        keep F and U
        drop G

    Steps:
        1. Reload or build memmaps
        2. Reconstruct F/U train mask from processed parquet partitions
        3. Build F/U val mask using y_val + is_labeled_val
        4. Estimate RAM for masked + undersampled train matrix and adjust ratio if needed
        5. Train CatBoost on train split
        6. Evaluate on F/U validation subset
        7. Retrain on train+val combined F/U subset
        8. Evaluate final model on validation subset
        9. Save model to MODELS_DIR
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
            X_train,
            y_train,
            X_val,
            y_val,
            il_val,
            tg_train,
            tg_val,
            feature_cols,
        ) = build_memmaps(
            files, labels, cutoff, train_end, n_train, n_val, STAGING_DIR
        )
        del labels
        gc.collect()

    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}\n")

    labels = load_labels()

    # ── Step 1: Build F/U masks ──────────────────────────────────────────────
    train_fu_mask = _build_fu_train_mask(files, labels, cutoff)

    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)
    il_val_np = np.asarray(il_val)

    # Keep F or U on val, drop G.
    val_fu_mask = (y_val_np == 1) | (il_val_np == 0)

    n_train_fu = int(train_fu_mask.sum())
    n_val_fu = int(val_fu_mask.sum())

    n_train_pos = int(y_train_np[train_fu_mask].sum()) if n_train_fu > 0 else 0
    n_train_neg = n_train_fu - n_train_pos  # U only
    n_val_pos = int(y_val_np[val_fu_mask].sum()) if n_val_fu > 0 else 0
    n_val_neg = n_val_fu - n_val_pos        # U only

    print("F/U subset summary:")
    print(f"  Train rows : {n_train_fu:,}  |  F positives: {n_train_pos:,}  |  U negatives: {n_train_neg:,}")
    print(f"  Val rows   : {n_val_fu:,}  |  F positives: {n_val_pos:,}  |  U negatives: {n_val_neg:,}\n")

    if n_train_fu == 0 or n_train_pos == 0:
        raise RuntimeError("F/U train subset is empty or has zero positives.")
    if n_val_fu == 0 or n_val_pos == 0:
        raise RuntimeError("F/U val subset is empty or has zero positives.")

    # ── Step 2: Decide negative sampling ratio based on RAM estimate ────────
    neg_ratio = NEG_SAMPLE_RATIO
    n_neg_keep_est = min(n_train_neg, int(n_train_neg * neg_ratio))
    n_neg_keep_est = max(n_neg_keep_est, n_train_pos)
    n_rows_fit_est = n_train_pos + n_neg_keep_est
    est_ram_gb = _estimate_matrix_ram_gb(n_rows_fit_est, len(feature_cols), dtype_bytes=4)

    print("Train matrix RAM estimate after mask + undersampling:")
    print(f"  NEG_SAMPLE_RATIO candidate : {neg_ratio}")
    print(f"  Positives kept             : {n_train_pos:,}")
    print(f"  Negatives kept             : {n_neg_keep_est:,}")
    print(f"  Final train rows           : {n_rows_fit_est:,}")
    print(f"  Estimated X RAM            : {est_ram_gb:.3f} GB")

    # Threshold for CatBoost on a 16 GB machine.
    if est_ram_gb > 10.0:
        neg_ratio = neg_ratio / 5
        n_neg_keep_est = min(n_train_neg, int(n_train_pos * neg_ratio))
        n_rows_fit_est = n_train_pos + n_neg_keep_est
        est_ram_gb = _estimate_matrix_ram_gb(n_rows_fit_est, len(feature_cols), dtype_bytes=4)

        print("\n  Estimated RAM is heavy - switching NEG_SAMPLE_RATIO to 0.01")
        print(f"  Adjusted negatives kept    : {n_neg_keep_est:,}")
        print(f"  Adjusted final train rows  : {n_rows_fit_est:,}")
        print(f"  Adjusted estimated X RAM   : {est_ram_gb:.3f} GB")
    print()

    # ── Step 3: Prepare validation subset in RAM ─────────────────────────────
    print("Loading F/U validation subset into RAM …", flush=True)
    val_pos_idx = np.flatnonzero(y_val_np == 1)
    val_u_idx_all = np.flatnonzero((y_val_np == 0) & (il_val_np == 0))

    n_val_pos = len(val_pos_idx)
    n_val_u = len(val_u_idx_all)

    n_val_u_keep = min(n_val_u, int(n_val_u * neg_ratio))
    n_val_u_keep = max(n_val_u_keep, n_val_pos)

    rng = np.random.default_rng(UNDERSAMPLE_SEED)
    if n_val_u_keep < n_val_u:
        val_u_idx = rng.choice(val_u_idx_all, size=n_val_u_keep, replace=False)
    else:
        val_u_idx = val_u_idx_all

    val_fu_idx = np.concatenate([val_pos_idx, val_u_idx])
    rng.shuffle(val_fu_idx)

    X_val_fu = np.asarray(X_val[val_fu_idx], dtype=np.float32)
    y_val_fu = np.asarray(y_val_np[val_fu_idx], dtype=np.int8)
    il_val_ones_fu = np.ones(len(val_fu_idx), dtype=np.int8)

    print("Validation subset after negative subsampling:")
    print(f"  F positives kept : {n_val_pos:,}")
    print(f"  U negatives kept : {len(val_u_idx):,}")
    print(f"  X_val_fu shape : {X_val_fu.shape}")
    print(f"  y_val_fu rows  : {len(y_val_fu):,}\n")

    # ── Step 4: Train on train split ─────────────────────────────────────────
    print("=" * 65)
    print("  CatBoost F/U model - train split")
    print("=" * 65)
    print()

    cb_model = train_catboost_model(
        X_train,
        y_train,
        X_val_fu,
        y_val_fu,
        il_val_ones_fu,
        feature_cols,
        train_row_mask=train_fu_mask,
        neg_sample_ratio=neg_ratio,
        catboost_params=CATBOOST_PARAMS,
    )

    # model_path = os.path.join(MODELS_DIR, CATBOOST_FU_MODEL_FILENAME)
    # cb_model.save_model(model_path)
    # print(f"\nCatBoost F/U model saved → {model_path}\n", flush=True)

    print("Scoring F/U validation rows …", flush=True)
    val_scores = _score_val_group(
        cb_model,
        X_val,
        val_fu_idx,
        is_catboost=True,
        suffix="catboost f/u val",
    )
    val_pr_auc = average_precision_score(y_val_fu, val_scores)
    print(f"\nValidation PR-AUC [F/U CatBoost]: {val_pr_auc:.6f}\n", flush=True)

    # ── Step 5: Refit on train + val combined F/U subset ────────────────────
    print("=" * 65)
    print("  CatBoost F/U model - train + val combined")
    print("=" * 65)
    print()

    train_pos_idx = np.flatnonzero(train_fu_mask & (y_train_np == 1))
    train_neg_idx_all = np.flatnonzero(train_fu_mask & (y_train_np == 0))

    val_pos_idx = np.flatnonzero(val_fu_mask & (y_val_np == 1))
    val_neg_idx_all = np.flatnonzero(val_fu_mask & (y_val_np == 0))

    n_pos_full = len(train_pos_idx) + len(val_pos_idx)
    n_neg_full = len(train_neg_idx_all) + len(val_neg_idx_all)
    n_neg_keep_full = min(n_neg_full, int(n_neg_full * neg_ratio))

    if n_neg_full > 0 and n_neg_keep_full > 0:
        share_train = len(train_neg_idx_all) / n_neg_full
        n_neg_keep_train = min(len(train_neg_idx_all), int(round(n_neg_keep_full * share_train)))
        n_neg_keep_val = min(len(val_neg_idx_all), n_neg_keep_full - n_neg_keep_train)

        # Fix any rounding remainder.
        shortfall = n_neg_keep_full - (n_neg_keep_train + n_neg_keep_val)
        if shortfall > 0:
            extra_train = min(shortfall, len(train_neg_idx_all) - n_neg_keep_train)
            n_neg_keep_train += extra_train
            shortfall -= extra_train
        if shortfall > 0:
            extra_val = min(shortfall, len(val_neg_idx_all) - n_neg_keep_val)
            n_neg_keep_val += extra_val

        rng = np.random.default_rng(UNDERSAMPLE_SEED)
        train_neg_idx = (
            rng.choice(train_neg_idx_all, size=n_neg_keep_train, replace=False)
            if n_neg_keep_train > 0 else np.empty(0, dtype=np.int64)
        )
        val_neg_idx = (
            rng.choice(val_neg_idx_all, size=n_neg_keep_val, replace=False)
            if n_neg_keep_val > 0 else np.empty(0, dtype=np.int64)
        )
    else:
        train_neg_idx = np.empty(0, dtype=np.int64)
        val_neg_idx = np.empty(0, dtype=np.int64)

    print("Combined F/U refit subset:")
    print(f"  Positives kept : {n_pos_full:,}")
    print(f"  Negatives kept : {len(train_neg_idx) + len(val_neg_idx):,}")
    print(f"  Total rows     : {n_pos_full + len(train_neg_idx) + len(val_neg_idx):,}\n")

    X_full_fu = np.concatenate(
        [
            np.asarray(X_train[train_pos_idx], dtype=np.float32),
            np.asarray(X_val[val_pos_idx], dtype=np.float32),
            np.asarray(X_train[train_neg_idx], dtype=np.float32),
            np.asarray(X_val[val_neg_idx], dtype=np.float32),
        ],
        axis=0,
    )

    y_full_fu = np.concatenate(
        [
            np.ones(len(train_pos_idx), dtype=np.int8),
            np.ones(len(val_pos_idx), dtype=np.int8),
            np.zeros(len(train_neg_idx), dtype=np.int8),
            np.zeros(len(val_neg_idx), dtype=np.int8),
        ],
        axis=0,
    )

    rng = np.random.default_rng(UNDERSAMPLE_SEED)
    perm = rng.permutation(len(y_full_fu))
    X_full_fu = X_full_fu[perm]
    y_full_fu = y_full_fu[perm]

    final_cb_model = train_catboost_model(
        X_full_fu,
        y_full_fu,
        X_val_fu,
        y_val_fu,
        il_val_ones_fu,
        feature_cols,
        train_row_mask=None,
        neg_sample_ratio=None,
        catboost_params=CATBOOST_PARAMS,
    )

    final_model_path = os.path.join(MODELS_DIR, CATBOOST_FU_MODEL_FILENAME)
    final_cb_model.save_model(final_model_path)
    print(f"\nFinal CatBoost F/U model saved → {final_model_path}\n", flush=True)

    print("Scoring F/U validation rows with final model …", flush=True)
    final_val_scores = final_cb_model.predict_proba(X_val_fu)[:, 1].astype(np.float32)
    final_val_pr_auc = average_precision_score(y_val_fu, final_val_scores)
    print(f"\nFinal validation PR-AUC [F/U CatBoost]: {final_val_pr_auc:.6f}\n", flush=True)

    del (
        labels,
        train_fu_mask,
        val_fu_mask,
        X_val_fu,
        y_val_fu,
        il_val_ones_fu,
        val_scores,
        train_pos_idx,
        train_neg_idx_all,
        val_pos_idx,
        val_neg_idx_all,
        train_neg_idx,
        val_neg_idx,
        X_full_fu,
        y_full_fu,
        final_val_scores,
        y_train_np,
        y_val_np,
        il_val_np,
    )
    gc.collect()

    print("─" * 65)
    print(f"Total wall time           : {_fmt(time.perf_counter() - total_start)}")
    print(f"Validation PR-AUC         : {val_pr_auc:.6f}")
    print(f"Final validation PR-AUC   : {final_val_pr_auc:.6f}")
    print("─" * 65)


def train_final_ensemble() -> None:
    """
    Train final CatBoost ensemble on outputs of previous models:

        1. tx_type_group LightGBM score
        2. tx_type_group CatBoost score
        3. tx_type_group blended score (same as train_baseline)
        4. RF score trained on F vs G
        5. CatBoost score trained on F vs U

    Final ensemble target:
        1 = F
        0 = G + U

    Negative subsampling:
        n_neg_keep = min(n_neg, int(n_neg * NEG_SAMPLE_RATIO))
        n_neg_keep = max(n_neg_keep, n_pos)
        n_neg_keep = min(n_neg_keep, n_neg)
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
            X_train,
            y_train,
            X_val,
            y_val,
            il_val,
            tg_train,
            tg_val,
            feature_cols,
        ) = build_memmaps(
            files, labels, cutoff, train_end, n_train, n_val, STAGING_DIR
        )
        del labels
        gc.collect()

    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}\n")

    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)
    tg_train_np = np.asarray(tg_train)
    tg_val_np = np.asarray(tg_val)

    train_idx = np.arange(len(y_train_np), dtype=np.int64)
    val_idx = np.arange(len(y_val_np), dtype=np.int64)

    n_train_pos = int(y_train_np.sum())
    n_train_neg = len(y_train_np) - n_train_pos
    n_val_pos = int(y_val_np.sum())
    n_val_neg = len(y_val_np) - n_val_pos

    print("Final ensemble full-data summary:")
    print(
        f"  Train rows : {len(y_train_np):,}  |  "
        f"F positives: {n_train_pos:,}  |  non-fraud negatives (G+U): {n_train_neg:,}"
    )
    print(
        f"  Val rows   : {len(y_val_np):,}  |  "
        f"F positives: {n_val_pos:,}  |  non-fraud negatives (G+U): {n_val_neg:,}\n"
    )

    neg_ratio = NEG_SAMPLE_RATIO
    n_neg_keep_est = min(n_train_neg, int(n_train_neg * neg_ratio))
    n_neg_keep_est = max(n_neg_keep_est, n_train_pos)

    n_rows_fit_est = n_train_pos + n_neg_keep_est
    est_ram_gb = _estimate_matrix_ram_gb(n_rows_fit_est, 5, dtype_bytes=4)

    print("Final ensemble train matrix RAM estimate after negative subsampling:")
    print(f"  NEG_SAMPLE_RATIO candidate : {neg_ratio}")
    print(f"  Positives kept             : {n_train_pos:,}")
    print(f"  Negatives kept             : {n_neg_keep_est:,}")
    print(f"  Final train rows           : {n_rows_fit_est:,}")
    print(f"  Estimated X RAM            : {est_ram_gb:.6f} GB\n")

    tx_lgbm_models = _load_tx_type_lgbm_models(MODELS_DIR)
    tx_cb_models = _load_tx_type_catboost_models(MODELS_DIR)
    rf_bundle = _load_rf_bundle(MODELS_DIR)
    fu_cb_model = _load_fu_catboost_model(MODELS_DIR)

    print("=" * 65)
    print("  Building train meta-features")
    print("=" * 65)
    print()

    X_meta_train = _score_meta_features_chunked(
        X_train,
        train_idx,
        tg_train_np,
        tx_lgbm_models,
        tx_cb_models,
        rf_bundle,
        fu_cb_model,
        suffix="meta train",
    )
    y_meta_train = y_train_np.astype(np.int8, copy=False)

    print("=" * 65)
    print("  Building validation meta-features")
    print("=" * 65)
    print()

    X_meta_val = _score_meta_features_chunked(
        X_val,
        val_idx,
        tg_val_np,
        tx_lgbm_models,
        tx_cb_models,
        rf_bundle,
        fu_cb_model,
        suffix="meta val",
    )
    y_meta_val = y_val_np.astype(np.int8, copy=False)
    il_meta_val = np.ones(len(y_meta_val), dtype=np.int8)

    print("Meta-feature matrices:")
    print(f"  X_meta_train shape : {X_meta_train.shape}")
    print(f"  X_meta_val shape   : {X_meta_val.shape}\n")

    meta_feature_cols = [
        "score_tx_group_lgbm",
        "score_tx_group_catboost",
        "score_tx_group_blend",
        "score_rf_fg",
        "score_catboost_fu",
    ]

    print("=" * 65)
    print("  Final ensemble CatBoost - train split")
    print("=" * 65)
    print()

    final_model = train_catboost_model(
        X_meta_train,
        y_meta_train,
        X_meta_val,
        y_meta_val,
        il_meta_val,
        meta_feature_cols,
        train_row_mask=None,
        neg_sample_ratio=neg_ratio,
        catboost_params=CATBOOST_PARAMS,
    )

    model_path = os.path.join(MODELS_DIR, FINAL_ENSEMBLE_MODEL_FILENAME)
    final_model.save_model(model_path)
    print(f"\nFinal ensemble model saved → {model_path}\n", flush=True)

    val_scores = final_model.predict_proba(X_meta_val)[:, 1].astype(np.float32)
    val_pr_auc = average_precision_score(y_meta_val, val_scores)
    print(f"Validation PR-AUC [final ensemble]: {val_pr_auc:.6f}\n", flush=True)

    print("=" * 65)
    print("  Final ensemble CatBoost - train + val combined")
    print("=" * 65)
    print()

    X_meta_full = np.concatenate([X_meta_train, X_meta_val], axis=0)
    y_meta_full = np.concatenate([y_meta_train, y_meta_val], axis=0)

    n_pos_full = int(y_meta_full.sum())
    n_neg_full = len(y_meta_full) - n_pos_full
    n_neg_keep_full = min(n_neg_full, int(n_neg_full * neg_ratio))
    n_neg_keep_full = max(n_neg_keep_full, n_pos_full)
    n_neg_keep_full = min(n_neg_keep_full, n_neg_full)

    print("Combined refit subset before internal subsampling:")
    print(f"  Positives total           : {n_pos_full:,}")
    print(f"  Negatives total           : {n_neg_full:,}")
    print(f"  Target negatives after ratio : {n_neg_keep_full:,}")
    print(f"  Total rows available      : {len(y_meta_full):,}\n")

    final_model_refit = train_catboost_model(
        X_meta_full,
        y_meta_full,
        X_meta_val,
        y_meta_val,
        il_meta_val,
        meta_feature_cols,
        train_row_mask=None,
        neg_sample_ratio=neg_ratio,
        catboost_params=CATBOOST_PARAMS,
    )

    final_model_refit.save_model(model_path)
    print(f"\nRefit final ensemble model saved → {model_path}\n", flush=True)

    final_val_scores = final_model_refit.predict_proba(X_meta_val)[:, 1].astype(np.float32)
    final_val_pr_auc = average_precision_score(y_meta_val, final_val_scores)
    print(f"Final validation PR-AUC [final ensemble]: {final_val_pr_auc:.6f}\n", flush=True)

    score_test_final_ensemble(
        final_model_refit,
        tx_lgbm_models,
        tx_cb_models,
        rf_bundle,
        fu_cb_model,
        feature_cols,
        SUBMISSION_PATH,
    )

    del (
        y_train_np,
        y_val_np,
        tg_train_np,
        tg_val_np,
        train_idx,
        val_idx,
        X_meta_train,
        y_meta_train,
        X_meta_val,
        y_meta_val,
        il_meta_val,
        val_scores,
        X_meta_full,
        y_meta_full,
        final_val_scores,
    )
    gc.collect()

    print("─" * 65)
    print(f"Total wall time           : {_fmt(time.perf_counter() - total_start)}")
    print(f"Validation PR-AUC         : {val_pr_auc:.6f}")
    print(f"Final validation PR-AUC   : {final_val_pr_auc:.6f}")
    print("─" * 65)
