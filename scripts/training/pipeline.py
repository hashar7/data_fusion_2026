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
import joblib

from scripts.training.train_rf import (
    build_labeled_fg_train_mapping,
    sanitize_rf_features,
    train_rf_model,
)
from scripts.training.config import RF_MODEL_PATH_FMT, RF_BASE_PARAMS

# ── F vs U catboost ────────────────────────────────────────────────–––––––––––––
from scripts.training.config import (
    CATBOOST_FU_MODEL_PATH_FMT
)

# ── Final ensemble ────────────────────────────────────────────────–––––––––––––
import gc
import glob
import os
import time
import joblib
from datetime import datetime

import lightgbm as lgb
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from scripts.training.config import (
    CATBOOST_BLEND_WEIGHT,
    CATBOOST_FU_MODEL_PATH_FMT,
    CATBOOST_PARAMS,
    FEATURES_DIR,
    FINAL_ENSEMBLE_MODEL_PATH_FMT,
    LGBM_MODEL_PATH_FMT,
    MODELS_DIR,
    NEG_SAMPLE_RATIO,
    RF_MODEL_PATH_FMT,
    STAGING_DIR,
    SUBMISSION_PATH,
    TRAIN_END_DATE,
    TX_TYPE_GROUPS,
    VAL_CUTOFF_DATE,
)
from scripts.training.data import (
    _load_cache,
    build_memmaps,
    count_rows,
    load_labels,
)
from scripts.training.train_catboost import train_catboost_model
from scripts.training.train_rf import sanitize_rf_features
from scripts.training._utils import (
    _submission_with_suffix, 
    _load_group_tx_catboost_models, _load_group_tx_lgbm_models, 
    _load_group_rf_bundles, _load_group_fu_catboost_models,
)
from scripts.training.predict import _score_tx_group_ensemble_single_group, score_test_final_ensemble_by_group


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


def _sample_binary_indices(y: np.ndarray, neg_ratio: float, seed: int = 42) -> np.ndarray:
    """
    Binary target sampling:
        positives = y == 1
        negatives = y == 0

    Keep all positives.
    Keep negatives by ratio of all negatives, but at least as many as positives.
    """
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if len(pos_idx) == 0:
        raise RuntimeError("Sampling subset has zero positive rows.")
    if len(neg_idx) == 0:
        raise RuntimeError("Sampling subset has zero negative rows.")

    n_neg_keep = min(len(neg_idx), int(len(neg_idx) * neg_ratio))
    n_neg_keep = max(n_neg_keep, len(pos_idx))
    # n_neg_keep = min(n_neg_keep, len(neg_idx))

    rng = np.random.default_rng(seed)
    if n_neg_keep < len(neg_idx):
        neg_keep = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    else:
        neg_keep = neg_idx.copy()

    sampled_idx = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(sampled_idx)
    return sampled_idx.astype(np.int64, copy=False)


def _build_meta_features_for_group(
    X_mm: np.ndarray,
    global_indices: np.ndarray,
    group_id: int,
    tx_lgbm_models: dict[int, list[lgb.Booster]],
    tx_cb_models: dict[int, CatBoostClassifier],
    rf_bundles: dict[int, dict],
    fu_cb_models: dict[int, CatBoostClassifier],
    chunk_size: int = 250_000,
    suffix: str = "meta group",
) -> np.ndarray:
    """
    Build meta-features for one model_group only.

    Meta-feature order:
        0 -> tx_type_group LightGBM average score
        1 -> tx_type_group CatBoost score
        2 -> tx_type_group blended score
        3 -> RF skeptic score (per-group RF)
        4 -> CatBoost F/U score (per-group)
    """
    rf_bundle = rf_bundles[group_id]
    rf_model = rf_bundle["model"]
    rf_feature_cols = rf_bundle["feature_cols"]
    fu_cb_model = fu_cb_models[group_id]

    n = len(global_indices)
    X_meta = np.empty((n, 5), dtype=np.float32)

    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    chunk_times: list[float] = []

    for chunk_i, cs in enumerate(range(0, n, chunk_size)):
        t0 = time.perf_counter()
        ce = min(cs + chunk_size, n)

        idx = global_indices[cs:ce]
        X_c = np.asarray(X_mm[idx], dtype=np.float32)

        tx_lgbm_scores, tx_cb_scores, tx_blend_scores = _score_tx_group_ensemble_single_group(
            X_c,
            group_id,
            tx_lgbm_models,
            tx_cb_models,
        )

        # RF sanitization happens only after row sampling, on selected rows only.
        X_rf = sanitize_rf_features(
            X_c,
            feature_cols=rf_feature_cols,
            stage=f"{suffix} rf group={group_id} chunk={chunk_i}",
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


def train_rf_fg_by_group() -> None:
    """
    Train one RandomForest skeptic model per model_group on labeled F/G rows only.

    Target is reversed relative to fraud:
        1 = G  (complex non-fraud)
        0 = F  (fraud)

    Uses fixed RF params from config, without CV.
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

    train_labeled_idx, train_orig_target = build_labeled_fg_train_mapping(files, labels, cutoff)

    if len(train_labeled_idx) == 0:
        raise RuntimeError("No labeled F/G train rows found.")

    tg_train_np = np.asarray(tg_train)
    tg_val_np = np.asarray(tg_val)
    y_val_np = np.asarray(y_val)
    il_val_np = np.asarray(il_val)

    # Reverse target for RF skeptic model:
    #   original target: 1 = F, 0 = G
    #   RF target      : 1 = G, 0 = F
    y_train_rf_all = (train_orig_target == 0).astype(np.int8)

    # On val memmap, labeled rows are already F/G only.
    val_labeled_idx = np.flatnonzero(il_val_np == 1)
    y_val_rf_all = (y_val_np[val_labeled_idx] == 0).astype(np.int8)
    tg_val_labeled = tg_val_np[val_labeled_idx]

    group_pr_aucs: dict[int, float] = {}

    print("Fixed RF params:")
    print(f"  {RF_BASE_PARAMS}\n")

    for group_id, group_name in TX_TYPE_GROUPS.items():
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  RF skeptic model — Group {group_id} — {group_name.upper()}")
        print(f"{sep}\n")

        train_group_mask = tg_train_np[train_labeled_idx] == group_id
        val_group_mask = tg_val_labeled == group_id

        train_idx_g = train_labeled_idx[train_group_mask]
        y_train_g = y_train_rf_all[train_group_mask]

        val_idx_g = val_labeled_idx[val_group_mask]
        y_val_g = y_val_rf_all[val_group_mask]

        n_train_g = len(train_idx_g)
        n_val_g = len(val_idx_g)
        n_train_pos_g = int(y_train_g.sum()) if n_train_g > 0 else 0
        n_train_neg_g = n_train_g - n_train_pos_g
        n_val_pos_g = int(y_val_g.sum()) if n_val_g > 0 else 0
        n_val_neg_g = n_val_g - n_val_pos_g

        print(f"  Train labeled rows : {n_train_g:,}  |  G positives: {n_train_pos_g:,}  |  F negatives: {n_train_neg_g:,}")
        print(f"  Val labeled rows   : {n_val_g:,}  |  G positives: {n_val_pos_g:,}  |  F negatives: {n_val_neg_g:,}\n")

        if n_train_g == 0 or n_val_g == 0:
            print("  SKIP: no train/val data for this group.\n")
            continue
        if n_train_pos_g == 0 or n_train_neg_g == 0:
            print("  SKIP: train subset has only one class.\n")
            continue
        if n_val_pos_g == 0 or n_val_neg_g == 0:
            print("  SKIP: val subset has only one class.\n")
            continue

        print("Loading group train/val subsets into RAM …", flush=True)

        X_train_g = np.asarray(X_train[train_idx_g], dtype=np.float32)
        X_val_g = np.asarray(X_val[val_idx_g], dtype=np.float32)

        X_train_g = sanitize_rf_features(
            X_train_g,
            feature_cols=feature_cols,
            stage=f"X_train_rf_group_{group_id}",
        )
        X_val_g = sanitize_rf_features(
            X_val_g,
            feature_cols=feature_cols,
            stage=f"X_val_rf_group_{group_id}",
        )

        print(f"  X_train_g shape : {X_train_g.shape}")
        print(f"  X_val_g shape   : {X_val_g.shape}\n")

        print("Training RF with fixed params …", flush=True)
        best_train_model = train_rf_model(
            X_train_g,
            y_train_g,
            rf_params=RF_BASE_PARAMS,
        )

        val_scores = best_train_model.predict_proba(X_val_g)[:, 1].astype(np.float32)
        val_pr_auc = average_precision_score(y_val_g, val_scores)
        group_pr_aucs[group_id] = val_pr_auc

        print(f"\nValidation PR-AUC [{group_name} RF skeptic]: {val_pr_auc:.6f}\n", flush=True)

        print("Refitting RF on train + val labeled rows …", flush=True)
        X_full_g = np.concatenate([X_train_g, X_val_g], axis=0)
        y_full_g = np.concatenate([y_train_g, y_val_g], axis=0)

        final_model = train_rf_model(
            X_full_g,
            y_full_g,
            rf_params=RF_BASE_PARAMS,
        )

        final_val_scores = final_model.predict_proba(X_val_g)[:, 1].astype(np.float32)
        final_val_pr_auc = average_precision_score(y_val_g, final_val_scores)

        model_path = os.path.join(
            MODELS_DIR,
            RF_MODEL_PATH_FMT.format(name=group_name),
        )
        bundle = {
            "model": final_model,
            "feature_cols": feature_cols,
            "rf_params": RF_BASE_PARAMS.copy(),
            "group_id": group_id,
            "group_name": group_name,
            "holdout_val_pr_auc": float(val_pr_auc),
            "final_val_pr_auc_after_refit": float(final_val_pr_auc),
            "target_definition": {
                "1": "G_complex_non_fraud",
                "0": "F_fraud",
            },
        }
        joblib.dump(bundle, model_path)
        print(f"Final RF skeptic model bundle saved → {model_path}\n", flush=True)

        del (
            X_train_g,
            X_val_g,
            y_train_g,
            y_val_g,
            best_train_model,
            val_scores,
            X_full_g,
            y_full_g,
            final_model,
            final_val_scores,
        )
        gc.collect()

    print(f"\n{'=' * 65}")
    print("  RF skeptic validation summary")
    print(f"{'=' * 65}\n")
    for group_id, group_name in TX_TYPE_GROUPS.items():
        if group_id in group_pr_aucs:
            print(f"  {group_name:12s} RF PR-AUC : {group_pr_aucs[group_id]:.6f}")

    print(f"\n{'─' * 65}")
    print(f"Total wall time : {_fmt(time.perf_counter() - total_start)}")
    print("─" * 65)


def train_catboost_fu_by_group() -> None:
    """
    Train one CatBoost model per model_group on F vs U rows only.

    Definitions:
        F = target == 1
        U = unlabeled rows (target == 0 in memmap and il_val == 0 on val)
        G = excluded

    Per group:
        1. Build F/U train subset
        2. Subsample U negatives
        3. Build F/U validation subset
        4. Subsample U negatives in validation with same ratio
        5. Train CatBoost
        6. Evaluate on validation subset
        7. Refit on train + val combined subset
        8. Save final model
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

    train_fu_mask = _build_fu_train_mask(files, labels, cutoff)

    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)
    il_val_np = np.asarray(il_val)
    tg_train_np = np.asarray(tg_train)
    tg_val_np = np.asarray(tg_val)

    val_fu_mask = (y_val_np == 1) | (il_val_np == 0)

    group_pr_aucs: dict[int, float] = {}

    for group_id, group_name in TX_TYPE_GROUPS.items():
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  CatBoost F/U model — Group {group_id} — {group_name.upper()}")
        print(f"{sep}\n")

        train_group_mask = train_fu_mask & (tg_train_np == group_id)
        val_group_mask = val_fu_mask & (tg_val_np == group_id)

        train_idx_all = np.flatnonzero(train_group_mask)
        val_idx_all = np.flatnonzero(val_group_mask)

        if len(train_idx_all) == 0 or len(val_idx_all) == 0:
            print("  SKIP: no train/val rows for this group.\n")
            continue

        y_train_g_all = y_train_np[train_idx_all].astype(np.int8, copy=False)
        y_val_g_all = y_val_np[val_idx_all].astype(np.int8, copy=False)

        train_pos_idx_local = np.flatnonzero(y_train_g_all == 1)
        train_u_idx_local = np.flatnonzero(y_train_g_all == 0)

        val_pos_idx_local = np.flatnonzero(y_val_g_all == 1)
        val_u_idx_local = np.flatnonzero(y_val_g_all == 0)

        n_train_pos = len(train_pos_idx_local)
        n_train_u = len(train_u_idx_local)
        n_val_pos = len(val_pos_idx_local)
        n_val_u = len(val_u_idx_local)

        print(f"  Train rows before sampling : {len(train_idx_all):,}  |  F positives: {n_train_pos:,}  |  U negatives: {n_train_u:,}")
        print(f"  Val rows before sampling   : {len(val_idx_all):,}  |  F positives: {n_val_pos:,}  |  U negatives: {n_val_u:,}\n")

        if n_train_pos == 0 or n_train_u == 0:
            print("  SKIP: train subset has only one class.\n")
            continue
        if n_val_pos == 0 or n_val_u == 0:
            print("  SKIP: val subset has only one class.\n")
            continue

        neg_ratio = NEG_SAMPLE_RATIO

        n_train_u_keep = min(n_train_u, int(n_train_u * neg_ratio))
        n_train_u_keep = max(n_train_u_keep, n_train_pos)
        # n_train_u_keep = min(n_train_u_keep, n_train_u)

        train_rows_fit_est = n_train_pos + n_train_u_keep
        est_ram_gb = _estimate_matrix_ram_gb(train_rows_fit_est, len(feature_cols), dtype_bytes=4)

        print("  Train RAM estimate after negative subsampling:")
        print(f"    NEG_SAMPLE_RATIO : {neg_ratio}")
        print(f"    Positives kept   : {n_train_pos:,}")
        print(f"    Negatives kept   : {n_train_u_keep:,}")
        print(f"    Final train rows : {train_rows_fit_est:,}")
        print(f"    Estimated X RAM  : {est_ram_gb:.3f} GB")

        if est_ram_gb > 6.0 and neg_ratio > 0.01:
            neg_ratio = 0.01
            n_train_u_keep = min(n_train_u, int(n_train_u * neg_ratio))
            n_train_u_keep = max(n_train_u_keep, n_train_pos)
            # n_train_u_keep = min(n_train_u_keep, n_train_u)

            train_rows_fit_est = n_train_pos + n_train_u_keep
            est_ram_gb = _estimate_matrix_ram_gb(train_rows_fit_est, len(feature_cols), dtype_bytes=4)

            print("    Estimated RAM is heavy - switching NEG_SAMPLE_RATIO to 0.01")
            print(f"    Adjusted negatives kept : {n_train_u_keep:,}")
            print(f"    Adjusted train rows     : {train_rows_fit_est:,}")
            print(f"    Adjusted X RAM          : {est_ram_gb:.3f} GB")
        print()

        rng = np.random.default_rng(UNDERSAMPLE_SEED)

        if n_train_u_keep < n_train_u:
            train_u_keep_local = rng.choice(train_u_idx_local, size=n_train_u_keep, replace=False)
        else:
            train_u_keep_local = train_u_idx_local.copy()

        train_idx_local = np.concatenate([train_pos_idx_local, train_u_keep_local])
        rng.shuffle(train_idx_local)
        train_idx_g = train_idx_all[train_idx_local]

        n_val_u_keep = min(n_val_u, int(n_val_u * neg_ratio))
        n_val_u_keep = max(n_val_u_keep, n_val_pos)
        # n_val_u_keep = min(n_val_u_keep, n_val_u)

        if n_val_u_keep < n_val_u:
            val_u_keep_local = rng.choice(val_u_idx_local, size=n_val_u_keep, replace=False)
        else:
            val_u_keep_local = val_u_idx_local.copy()

        val_idx_local = np.concatenate([val_pos_idx_local, val_u_keep_local])
        rng.shuffle(val_idx_local)
        val_idx_g = val_idx_all[val_idx_local]

        print("  Validation subset after negative subsampling:")
        print(f"    F positives kept : {n_val_pos:,}")
        print(f"    U negatives kept : {len(val_u_keep_local):,}")
        print(f"    Final val rows   : {len(val_idx_g):,}\n")

        X_train_g = np.asarray(X_train[train_idx_g], dtype=np.float32)
        y_train_g = y_train_np[train_idx_g].astype(np.int8, copy=False)

        X_val_g = np.asarray(X_val[val_idx_g], dtype=np.float32)
        y_val_g = y_val_np[val_idx_g].astype(np.int8, copy=False)
        il_val_ones_g = np.ones(len(y_val_g), dtype=np.int8)

        print(f"  X_train_g shape : {X_train_g.shape}")
        print(f"  X_val_g shape   : {X_val_g.shape}\n")

        print("  Training CatBoost …", flush=True)
        cb_model = train_catboost_model(
            X_train_g,
            y_train_g,
            X_val_g,
            y_val_g,
            il_val_ones_g,
            feature_cols,
            train_row_mask=None,
            neg_sample_ratio=None,
            catboost_params=CATBOOST_PARAMS,
        )

        model_path = os.path.join(
            MODELS_DIR,
            CATBOOST_FU_MODEL_PATH_FMT.format(name=group_name),
        )
        cb_model.save_model(model_path)
        print(f"\n  CatBoost F/U model saved → {model_path}\n", flush=True)

        val_scores = cb_model.predict_proba(X_val_g)[:, 1].astype(np.float32)
        val_pr_auc = average_precision_score(y_val_g, val_scores)
        group_pr_aucs[group_id] = val_pr_auc

        print(f"  Validation PR-AUC [{group_name} CatBoost F/U]: {val_pr_auc:.6f}\n", flush=True)

        print("  Refit on train + val combined …", flush=True)

        X_full_g = np.concatenate([X_train_g, X_val_g], axis=0)
        y_full_g = np.concatenate([y_train_g, y_val_g], axis=0)

        final_cb_model = train_catboost_model(
            X_full_g,
            y_full_g,
            X_val_g,
            y_val_g,
            il_val_ones_g,
            feature_cols,
            train_row_mask=None,
            neg_sample_ratio=None,
            catboost_params=CATBOOST_PARAMS,
        )

        final_cb_model.save_model(model_path)
        print(f"  Final CatBoost F/U model saved → {model_path}\n", flush=True)

        final_val_scores = final_cb_model.predict_proba(X_val_g)[:, 1].astype(np.float32)
        final_val_pr_auc = average_precision_score(y_val_g, final_val_scores)
        print(f"  Final validation PR-AUC [{group_name} CatBoost F/U]: {final_val_pr_auc:.6f}\n", flush=True)

        del (
            train_idx_all,
            val_idx_all,
            y_train_g_all,
            y_val_g_all,
            train_pos_idx_local,
            train_u_idx_local,
            val_pos_idx_local,
            val_u_idx_local,
            train_u_keep_local,
            val_u_keep_local,
            train_idx_local,
            val_idx_local,
            train_idx_g,
            val_idx_g,
            X_train_g,
            y_train_g,
            X_val_g,
            y_val_g,
            il_val_ones_g,
            cb_model,
            val_scores,
            X_full_g,
            y_full_g,
            final_cb_model,
            final_val_scores,
        )
        gc.collect()

    print(f"\n{'=' * 65}")
    print("  CatBoost F/U validation summary")
    print(f"{'=' * 65}\n")
    for group_id, group_name in TX_TYPE_GROUPS.items():
        if group_id in group_pr_aucs:
            print(f"  {group_name:12s} CatBoost F/U PR-AUC : {group_pr_aucs[group_id]:.6f}")

    print(f"\n{'─' * 65}")
    print(f"Total wall time : {_fmt(time.perf_counter() - total_start)}")
    print("─" * 65)


def train_final_ensemble_by_group() -> None:
    """
    Train one final CatBoost ensemble per model_group on outputs of previous models:

        1. tx_type_group LightGBM score
        2. tx_type_group CatBoost score
        3. tx_type_group blended score
        4. RF skeptic score trained on G vs F
        5. CatBoost score trained on F vs U

    Final ensemble target:
        1 = F
        0 = G + U

    Negative subsampling is done before meta-feature construction.
    RF sanitization therefore happens only on selected sampled rows.

    Writes two submissions:
        - before refit on train+val
        - after refit on train+val
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

    tx_lgbm_models = _load_group_tx_lgbm_models(MODELS_DIR)
    tx_cb_models = _load_group_tx_catboost_models(MODELS_DIR)
    rf_bundles = _load_group_rf_bundles(MODELS_DIR)
    fu_cb_models = _load_group_fu_catboost_models(MODELS_DIR)

    meta_feature_cols = [
        "score_tx_group_lgbm",
        "score_tx_group_catboost",
        "score_tx_group_blend",
        "score_rf_skeptic",
        "score_catboost_fu",
    ]

    final_models_pre_refit: dict[int, CatBoostClassifier] = {}
    final_models_post_refit: dict[int, CatBoostClassifier] = {}
    group_pr_aucs: dict[int, float] = {}

    for group_id, group_name in TX_TYPE_GROUPS.items():
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  Final ensemble — Group {group_id} — {group_name.upper()}")
        print(f"{sep}\n")

        train_group_idx_all = np.flatnonzero(tg_train_np == group_id)
        val_group_idx_all = np.flatnonzero(tg_val_np == group_id)

        if len(train_group_idx_all) == 0 or len(val_group_idx_all) == 0:
            print("  SKIP: no train/val rows for this group.\n")
            continue

        y_train_group_all = y_train_np[train_group_idx_all].astype(np.int8, copy=False)
        y_val_group_all = y_val_np[val_group_idx_all].astype(np.int8, copy=False)

        n_train_pos = int(y_train_group_all.sum())
        n_train_neg = len(y_train_group_all) - n_train_pos
        n_val_pos = int(y_val_group_all.sum())
        n_val_neg = len(y_val_group_all) - n_val_pos

        print(f"  Train rows before sampling : {len(train_group_idx_all):,}  |  F positives: {n_train_pos:,}  |  non-fraud negatives: {n_train_neg:,}")
        print(f"  Val rows before sampling   : {len(val_group_idx_all):,}  |  F positives: {n_val_pos:,}  |  non-fraud negatives: {n_val_neg:,}\n")

        if n_train_pos == 0 or n_train_neg == 0:
            print("  SKIP: train subset has only one class.\n")
            continue
        if n_val_pos == 0 or n_val_neg == 0:
            print("  SKIP: val subset has only one class.\n")
            continue

        train_sample_local = _sample_binary_indices(
            y_train_group_all,
            neg_ratio=NEG_SAMPLE_RATIO,
            seed=42 + group_id,
        )
        val_sample_local = _sample_binary_indices(
            y_val_group_all,
            neg_ratio=NEG_SAMPLE_RATIO,
            seed=42 + group_id,
        )

        train_idx_g = train_group_idx_all[train_sample_local]
        val_idx_g = val_group_idx_all[val_sample_local]

        y_train_g = y_train_np[train_idx_g].astype(np.int8, copy=False)
        y_val_g = y_val_np[val_idx_g].astype(np.int8, copy=False)

        n_train_pos_kept = int(y_train_g.sum())
        n_train_neg_kept = len(y_train_g) - n_train_pos_kept
        n_val_pos_kept = int(y_val_g.sum())
        n_val_neg_kept = len(y_val_g) - n_val_pos_kept

        est_ram_gb = _estimate_matrix_ram_gb(len(train_idx_g), 5, dtype_bytes=4)

        print("  Sampled subset summary:")
        print(f"    Train positives kept : {n_train_pos_kept:,}")
        print(f"    Train negatives kept : {n_train_neg_kept:,}")
        print(f"    Train rows kept      : {len(train_idx_g):,}")
        print(f"    Train meta X RAM est : {est_ram_gb:.6f} GB")
        print(f"    Val positives kept   : {n_val_pos_kept:,}")
        print(f"    Val negatives kept   : {n_val_neg_kept:,}")
        print(f"    Val rows kept        : {len(val_idx_g):,}\n")

        print("  Building sampled train meta-features …", flush=True)
        X_meta_train = _build_meta_features_for_group(
            X_train,
            train_idx_g,
            group_id,
            tx_lgbm_models,
            tx_cb_models,
            rf_bundles,
            fu_cb_models,
            suffix=f"meta train {group_name}",
        )

        print("  Building sampled validation meta-features …", flush=True)
        X_meta_val = _build_meta_features_for_group(
            X_val,
            val_idx_g,
            group_id,
            tx_lgbm_models,
            tx_cb_models,
            rf_bundles,
            fu_cb_models,
            suffix=f"meta val {group_name}",
        )
        il_meta_val = np.ones(len(y_val_g), dtype=np.int8)

        print(f"  X_meta_train shape : {X_meta_train.shape}")
        print(f"  X_meta_val shape   : {X_meta_val.shape}\n")

        print("  Training final CatBoost on train split …", flush=True)
        final_model = train_catboost_model(
            X_meta_train,
            y_train_g,
            X_meta_val,
            y_val_g,
            il_meta_val,
            meta_feature_cols,
            train_row_mask=None,
            neg_sample_ratio=None,
            catboost_params=CATBOOST_PARAMS,
        )

        model_path = os.path.join(
            MODELS_DIR,
            FINAL_ENSEMBLE_MODEL_PATH_FMT.format(name=group_name),
        )
        final_model.save_model(model_path)
        final_models_pre_refit[group_id] = final_model
        print(f"  Final ensemble model saved → {model_path}\n", flush=True)

        val_scores = final_model.predict_proba(X_meta_val)[:, 1].astype(np.float32)
        val_pr_auc = average_precision_score(y_val_g, val_scores)
        group_pr_aucs[group_id] = val_pr_auc
        print(f"  Validation PR-AUC [{group_name} final ensemble]: {val_pr_auc:.6f}\n", flush=True)

        print("  Refit on sampled train + val combined …", flush=True)
        X_meta_full = np.concatenate([X_meta_train, X_meta_val], axis=0)
        y_meta_full = np.concatenate([y_train_g, y_val_g], axis=0)

        final_model_refit = train_catboost_model(
            X_meta_full,
            y_meta_full,
            X_meta_val,
            y_val_g,
            il_meta_val,
            meta_feature_cols,
            train_row_mask=None,
            neg_sample_ratio=None,
            catboost_params=CATBOOST_PARAMS,
        )

        final_model_refit.save_model(model_path)
        final_models_post_refit[group_id] = final_model_refit
        print(f"  Refit final ensemble model saved → {model_path}\n", flush=True)

        del (
            train_group_idx_all,
            val_group_idx_all,
            y_train_group_all,
            y_val_group_all,
            train_sample_local,
            val_sample_local,
            train_idx_g,
            val_idx_g,
            y_train_g,
            y_val_g,
            X_meta_train,
            X_meta_val,
            X_meta_full,
            y_meta_full,
            il_meta_val,
            val_scores,
        )
        gc.collect()

    print(f"\n{'=' * 65}")
    print("  Final ensemble validation summary")
    print(f"{'=' * 65}\n")
    for group_id, group_name in TX_TYPE_GROUPS.items():
        if group_id in group_pr_aucs:
            print(f"  {group_name:12s} Final ensemble PR-AUC : {group_pr_aucs[group_id]:.6f}")

    submission_pre_refit = _submission_with_suffix(SUBMISSION_PATH, "pre_refit")
    submission_post_refit = _submission_with_suffix(SUBMISSION_PATH, "post_refit")

    score_test_final_ensemble_by_group(
        final_models_pre_refit,
        tx_lgbm_models,
        tx_cb_models,
        rf_bundles,
        fu_cb_models,
        feature_cols,
        submission_pre_refit,
    )

    score_test_final_ensemble_by_group(
        final_models_post_refit,
        tx_lgbm_models,
        tx_cb_models,
        rf_bundles,
        fu_cb_models,
        feature_cols,
        submission_post_refit,
    )

    print(f"\n{'─' * 65}")
    print(f"Total wall time : {_fmt(time.perf_counter() - total_start)}")
    print(f"Submission before refit : {submission_pre_refit}")
    print(f"Submission after refit  : {submission_post_refit}")
    print("─" * 65)
