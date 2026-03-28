"""Top-level orchestrator: trains one ensemble per tx_type_group."""
import gc
import glob
import os
import re
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import average_precision_score

from scripts.training.config import (
    FEATURES_DIR, STAGING_DIR, SUBMISSION_PATH, MODELS_DIR,
    VAL_CUTOFF_DATE, TRAIN_END_DATE, TX_TYPE_GROUPS,
    NEG_SAMPLE_RATIO, NEG_SAMPLE_RATIO_BY_GROUP,
    LGBM_PARAMS, LGBM_PARAMS_BY_GROUP,
    EARLY_STOPPING_ROUNDS_BY_GROUP,
    ENSEMBLE_SEEDS, CATBOOST_SEEDS,
    CATBOOST_PARAMS, CATBOOST_PARAMS_BY_GROUP, CATBOOST_BLEND_WEIGHT,
    LGBM_MODEL_PATH_FMT, CATBOOST_MODEL_PATH_FMT,
    RETRAIN_FULL_ITER_FACTOR, UNDERSAMPLE_SEED,
    YELLOW_WEIGHT_MULTIPLIER,
)
from scripts.training._utils import _fmt, _parquet_files, _progress
from scripts.training.data import load_labels, count_rows, build_memmaps, _load_cache
from scripts.training.train import train_model
from scripts.training.train_catboost import train_catboost_model
from scripts.training.evaluate import evaluate
from scripts.training.predict import score_test


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
        glob.glob(os.path.join(models_dir, "model_*.cbm"))
    )
    unversioned = [f for f in all_files
                   if not re.search(r"_ver_\d+\.(txt|cbm)$", f)]

    if not unversioned:
        return

    # Find the highest existing version number so we don't collide
    versioned = (
        glob.glob(os.path.join(models_dir, "model_*_ver_*.txt")) +
        glob.glob(os.path.join(models_dir, "model_*_ver_*.cbm"))
    )
    max_ver = 0
    for f in versioned:
        m = re.search(r"_ver_(\d+)\.(txt|cbm)$", f)
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
    blend scores, evaluate each group independently, compute the combined
    PR-AUC, then retrain all models on full data (train + labeled val) and
    use the retrained models to score the test set.

    Pipeline per group:
        1. Train LGBM N times with different seeds → average val scores
        2. Train CatBoost → blend with LGBM avg
        3. Compute ensemble PR-AUC on blended scores
        4. Retrain all models on train + labeled-val with fixed rounds
        5. Score test set with retrained models
    """
    total_start = time.perf_counter()
    cutoff    = datetime.fromisoformat(VAL_CUTOFF_DATE)
    train_end = datetime.fromisoformat(TRAIN_END_DATE)

    _prepare_models_dir(MODELS_DIR)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")
    print(f"Found {len(files)} parquet partitions in {FEATURES_DIR!r}\n", flush=True)
    print(f"Ensemble seeds          : {ENSEMBLE_SEEDS}")
    print(f"CatBoost seeds          : {CATBOOST_SEEDS}")
    print(f"CatBoost default weight : {CATBOOST_BLEND_WEIGHT}")
    print(f"Retrain iter factor     : {RETRAIN_FULL_ITER_FACTOR}\n", flush=True)

    # ── Step 1: Build or reload memmap split files ────────────────────────────
    cached = _load_cache(STAGING_DIR)
    if cached is not None:
        X_train, y_train, X_val, y_val, il_val, tg_train, tg_val, il_train, feature_cols = cached
    else:
        labels = load_labels()
        n_train, n_val = count_rows(files, cutoff, train_end)
        (X_train, y_train, X_val, y_val, il_val,
         tg_train, tg_val, il_train, feature_cols) = build_memmaps(
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

    pr_aucs:         dict = {}
    blend_weights:   dict = {}   # group_id → best CatBoost blend weight (from grid search)
    # Per-group best iterations from initial training (used for retraining).
    best_iters_lgbm:  dict = {}   # group_id → list[int]  (one per LGBM seed)
    best_iters_cb:    dict = {}   # group_id → list[int]  (one per CatBoost seed)

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
        seed_best_iters: list[int] = []

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
                il_train=il_train,
                yellow_weight_multiplier=YELLOW_WEIGHT_MULTIPLIER,
            )
            seed_best_iters.append(booster.best_iteration)

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
            del booster, s
            gc.collect()

        best_iters_lgbm[group_id] = seed_best_iters
        lgbm_val_scores = (lgbm_val_sum / len(ENSEMBLE_SEEDS)).astype(np.float32)
        del lgbm_val_sum
        gc.collect()

        # ── 2b. CatBoost multi-seed ──────────────────────────────────────────
        cb_val_sum       = np.zeros(len(val_idx_g), dtype=np.float64)
        cb_seed_best_iters: list[int] = []

        for cb_seed_idx, cb_seed in enumerate(CATBOOST_SEEDS):
            print(f"\n  ── CatBoost seed {cb_seed_idx + 1}/{len(CATBOOST_SEEDS)} "
                  f"(seed={cb_seed}) ────────────────────────────────────")
            cb_model = train_catboost_model(
                X_train, y_train,
                X_val_labeled_g, y_val_labeled_g, il_val_ones_g,
                feature_cols,
                train_row_mask=train_mask,
                neg_sample_ratio=group_neg_ratio,
                catboost_params=group_cb_params,
                seed_override=cb_seed,
            )
            cb_seed_best_iters.append(cb_model.best_iteration_)

            print(f"  Scoring val rows (CatBoost seed {cb_seed_idx}) …", flush=True)
            s = _score_val_group(
                cb_model, X_val, val_idx_g,
                is_catboost=True, suffix=f"catboost s{cb_seed_idx} {group_name}",
            )
            cb_val_sum += s.astype(np.float64)
            del cb_model, s
            gc.collect()

        best_iters_cb[group_id] = cb_seed_best_iters
        cb_val_scores = (cb_val_sum / len(CATBOOST_SEEDS)).astype(np.float32)
        del cb_val_sum

        del X_val_labeled_g, y_val_labeled_g, il_val_ones_g
        gc.collect()

        # ── 2c. Per-group blend weight grid search ────────────────────────────
        il_g          = il_val_np[val_idx_g]
        labeled_local = np.where(il_g == 1)[0]
        y_labeled     = y_val_np[val_idx_g][labeled_local]
        lgbm_lbl      = lgbm_val_scores[labeled_local]
        cb_lbl        = cb_val_scores[labeled_local]

        best_w      = CATBOOST_BLEND_WEIGHT
        best_prauc  = 0.0
        if len(labeled_local) > 0 and y_labeled.sum() > 0:
            for w_cand in np.arange(0.0, 0.55, 0.05):
                blended_cand = lgbm_lbl * (1.0 - w_cand) + cb_lbl * w_cand
                p = average_precision_score(y_labeled, blended_cand)
                if p > best_prauc:
                    best_prauc, best_w = p, float(w_cand)
        blend_weights[group_id] = best_w
        print(f"\n  Blend weight grid search [{group_name}]: "
              f"best_w={best_w:.2f}  PR-AUC={best_prauc:.6f}")
        del lgbm_lbl, cb_lbl

        # ── 2d. Blend with optimised weight ──────────────────────────────────
        w = best_w
        blended_scores = (
            lgbm_val_scores * (1.0 - w) + cb_val_scores * w
        ).astype(np.float32)
        del lgbm_val_scores, cb_val_scores
        gc.collect()

        # ── 2e. Per-group ensemble PR-AUC ─────────────────────────────────────
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

    del all_val_scores
    gc.collect()

    # ── Step 4: Retrain on full data (train + labeled val) ────────────────────
    print(f"\n{'=' * 65}")
    print("  Full-data retraining  (train + labeled val, no early stopping)")
    print(f"{'=' * 65}\n")

    lgbm_boosters_full:   dict = {}   # group_id → list[lgb.Booster]
    catboost_models_full: dict = {}   # group_id → CatBoostClassifier

    for group_id, group_name in TX_TYPE_GROUPS.items():
        if group_id not in best_iters_lgbm:
            continue

        sep = "-" * 65
        print(f"\n{sep}")
        print(f"  Full-data retrain — Group {group_id} — {group_name.upper()}")
        print(f"{sep}\n")

        train_mask       = tg_train_np == group_id
        labeled_val_mask = (tg_val_np == group_id) & (il_val_np == 1)

        # Extract labeled val rows for this group.
        # Item 7 fix: undersample val negatives at the same ratio as train negatives
        # before appending, so the combined pool has a consistent class balance.
        labeled_val_idx = np.where(labeled_val_mask)[0]
        X_val_extra_all = np.array(X_val[labeled_val_idx])
        y_val_extra_all = np.array(y_val[labeled_val_idx])

        val_pos_idx = np.where(y_val_extra_all == 1)[0]
        val_neg_idx = np.where(y_val_extra_all == 0)[0]
        if len(val_neg_idx) > 0:
            n_val_neg_keep  = max(int(len(val_neg_idx) * group_neg_ratio), len(val_pos_idx))
            n_val_neg_keep  = min(n_val_neg_keep, len(val_neg_idx))
            _rng_val        = np.random.default_rng(UNDERSAMPLE_SEED)
            val_neg_sampled = _rng_val.choice(val_neg_idx, size=n_val_neg_keep, replace=False)
            val_keep        = np.sort(np.concatenate([val_pos_idx, val_neg_sampled]))
        else:
            val_keep = val_pos_idx

        X_val_extra = X_val_extra_all[val_keep]
        y_val_extra = y_val_extra_all[val_keep]
        del X_val_extra_all, y_val_extra_all, val_pos_idx, val_neg_idx, val_keep
        gc.collect()

        print(f"  Extra val rows for retraining: {len(y_val_extra):,} "
              f"({int(y_val_extra.sum())} pos, {int((y_val_extra==0).sum())} neg "
              f"— undersampled at {group_neg_ratio:.0%})")

        group_neg_ratio   = NEG_SAMPLE_RATIO_BY_GROUP.get(group_id, NEG_SAMPLE_RATIO)
        group_lgbm_params = {**LGBM_PARAMS, **LGBM_PARAMS_BY_GROUP.get(group_id, {})}
        group_cb_params   = {**CATBOOST_PARAMS, **CATBOOST_PARAMS_BY_GROUP.get(group_id, {})}

        # Dummy val arrays (not used when n_rounds_fixed is set).
        _dummy_X = np.empty((0, len(feature_cols)), dtype=np.float32)
        _dummy_y = np.empty((0,), dtype=np.int8)
        _dummy_il = np.empty((0,), dtype=np.int8)

        # ── 4a. Multi-seed LightGBM (full data) ─────────────────────────────
        seed_boosters_full: list = []
        for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
            bi = best_iters_lgbm[group_id][seed_idx]
            n_rounds = int(bi * RETRAIN_FULL_ITER_FACTOR)
            print(f"\n  ── LGBM full retrain seed {seed_idx + 1}/{len(ENSEMBLE_SEEDS)} "
                  f"(seed={seed}, rounds={n_rounds} from best_iter={bi}) ──")
            booster = train_model(
                X_train, y_train,
                _dummy_X, _dummy_y, _dummy_il,
                feature_cols,
                train_row_mask=train_mask,
                neg_sample_ratio=group_neg_ratio,
                lgbm_params=group_lgbm_params,
                seed_override=seed,
                retrain_extra=(X_val_extra, y_val_extra),
                n_rounds_fixed=n_rounds,
            )
            path = os.path.join(MODELS_DIR, LGBM_MODEL_PATH_FMT.format(name=group_name, seed_idx=seed_idx))
            booster.save_model(path)
            print(f"  LGBM full model saved → {path}")
            seed_boosters_full.append(booster)
            del booster
            gc.collect()

        lgbm_boosters_full[group_id] = seed_boosters_full

        # ── 4b. CatBoost multi-seed (full data) ──────────────────────────────
        cb_seed_models_full: list = []
        for cb_seed_idx, cb_seed in enumerate(CATBOOST_SEEDS):
            cb_bi       = best_iters_cb[group_id][cb_seed_idx]
            cb_n_rounds = int(cb_bi * RETRAIN_FULL_ITER_FACTOR)
            print(f"\n  ── CatBoost full retrain seed {cb_seed_idx + 1}/{len(CATBOOST_SEEDS)} "
                  f"(seed={cb_seed}, rounds={cb_n_rounds} from best_iter={cb_bi}) ──")
            cb_model = train_catboost_model(
                X_train, y_train,
                _dummy_X, _dummy_y, _dummy_il,
                feature_cols,
                train_row_mask=train_mask,
                neg_sample_ratio=group_neg_ratio,
                catboost_params=group_cb_params,
                seed_override=cb_seed,
                retrain_extra=(X_val_extra, y_val_extra),
                n_rounds_fixed=cb_n_rounds,
            )
            cb_path = os.path.join(
                MODELS_DIR,
                CATBOOST_MODEL_PATH_FMT.format(name=group_name, seed_idx=cb_seed_idx),
            )
            cb_model.save_model(cb_path)
            print(f"  CatBoost full model saved → {cb_path}")
            cb_seed_models_full.append(cb_model)
            del cb_model
            gc.collect()

        catboost_models_full[group_id] = cb_seed_models_full

        del X_val_extra, y_val_extra
        gc.collect()

    # ── Step 5: Score test set with retrained models ─────────────────────────
    if lgbm_boosters_full:
        score_test(
            lgbm_boosters_full, catboost_models_full, feature_cols, SUBMISSION_PATH,
            blend_weights=blend_weights,
        )
    else:
        print("\nWARNING: no models were trained — submission skipped.")

    print(f"\n{'─' * 65}")
    print(f"Total wall time  : {_fmt(time.perf_counter() - total_start)}")
    print(f"Combined PR-AUC  : {combined_pr_auc:.6f}  (val, before full retrain)")
    print("─" * 65)
