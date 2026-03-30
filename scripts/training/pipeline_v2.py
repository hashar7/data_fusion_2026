"""
pipeline_v2.py — Hierarchical ensemble training pipeline.

Architecture:
    1. Per-group LightGBM  (memmaps, yellow-weighted negatives)
    2. Hierarchical CatBoost pair (global, from processed parquet):
         - Suspicious  : P(labeled | tx)      — (red | yellow) vs green
         - RGS         : P(fraud | labeled)   — red vs yellow, labeled only
         - Product     : sigmoid(susp) × sigmoid(rgs)  ≈ P(fraud | tx)
    3. Main CatBoost       (direct fraud prediction, global)
    4. Optional: Recent LightGBM    (global, data from RECENT_BORDER onward)
    5. Blend weight optimisation (logit-space, on all val rows or labeled only)
    6. Full-data retraining of all models
    7. Test scoring → submission.csv

Entry point:  train_v2(gpu=False)
"""
import gc
import os
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

from scripts.training.config import (
    FEATURES_DIR, STAGING_DIR, SUBMISSION_PATH, MODELS_DIR,
    VAL_CUTOFF_DATE, TRAIN_END_DATE,
    TX_TYPE_GROUPS,
    NEG_SAMPLE_RATIO, NEG_SAMPLE_RATIO_BY_GROUP,
    LGBM_PARAMS, LGBM_PARAMS_BY_GROUP,
    EARLY_STOPPING_ROUNDS, EARLY_STOPPING_ROUNDS_BY_GROUP,
    ENSEMBLE_SEEDS,
    LGBM_MODEL_PATH_FMT,
    RETRAIN_FULL_ITER_FACTOR, UNDERSAMPLE_SEED,
    YELLOW_WEIGHT_MULTIPLIER,
    HIERARCHICAL_CAT_FEATURES, HIERARCHICAL_NUM_FEATURES,
    SUSPICIOUS_GREEN_RATIO, RECENT_BORDER,
    SUSPICIOUS_LABELED_WEIGHT, SUSPICIOUS_GREEN_RECENT_W, SUSPICIOUS_GREEN_OLD_W,
    RGS_RED_WEIGHT, RGS_YELLOW_WEIGHT,
    SUSPICIOUS_CATBOOST_PARAMS, RGS_CATBOOST_PARAMS,
    SUSPICIOUS_MODEL_PATH, RGS_MODEL_PATH,
    TRAIN_RECENT_LGBM, RECENT_NEG_SAMPLE_RATIO, LGBM_PARAMS_RECENT, RECENT_MODEL_PATH,
    HIER_USE_ALL_FEATURES, HIER_FULL_GREEN_RATIO,
    MAIN_CATBOOST_PARAMS, MAIN_RED_WEIGHT, MAIN_YELLOW_WEIGHT,
    MAIN_GREEN_RECENT_W, MAIN_GREEN_OLD_W, MAIN_MODEL_PATH,
    BLEND_IN_LOGIT_SPACE,
    BLEND_ON_ALL_ROWS, BLEND_UNLABELED_SAMPLE_RATIO,
)
from scripts.training._utils import _fmt, _parquet_files, _progress
from scripts.training.data import load_labels, count_rows, build_memmaps, _load_cache
from scripts.training.train import train_model
from scripts.training.evaluate import evaluate
from scripts.training.hierarchical import (
    load_hier_split, train_suspicious, train_rgs, train_main_catboost,
    score_hierarchical, refit_model, _sigmoid, _logit,
)
from scripts.training.pipeline import _prepare_models_dir, _score_val_group


# ── Derived config ──────────────────────────────────────────────────────────────
# These will be overwritten at runtime in train_v2() when HIER_USE_ALL_FEATURES
_ALL_HIER_FEATURES_ORDERED = list(dict.fromkeys(
    HIERARCHICAL_NUM_FEATURES + HIERARCHICAL_CAT_FEATURES
))

# "mcc_code_int" is derived at load time from "mcc_code" (String) — not in parquet directly
_NEEDS_MCC_INT      = "mcc_code_int" in HIERARCHICAL_CAT_FEATURES
_HIER_PARQUET_EXTRA = [c for c in HIERARCHICAL_CAT_FEATURES if c != "mcc_code_int"]


# ── Helper: load labeled val rows (+ optional unlabeled sample) from parquet ──

def _load_labeled_val(
    files: list,
    labels_df: pl.DataFrame,
    cutoff: datetime,
    train_end: datetime,
    feature_cols: list,
    hier_cat_features: list | None = None,
    hier_all_features: list | None = None,
    unlabeled_sample_ratio: float = 0.0,
) -> pd.DataFrame:
    """
    Load labeled val rows (cutoff <= dttm < train_end, is_train==1, in labels_df)
    with LGBM feature columns + hierarchical extra columns.

    When unlabeled_sample_ratio > 0, also loads a deterministic hash-sampled
    fraction of unlabeled val rows (for all-rows blend optimisation).  These
    rows have raw_target = -1 and target = 0.

    Returns a pandas DataFrame with:
        event_id, target, raw_target, model_group, _is_labeled,
        <feature_cols>, <hier extra cols not in feature_cols>
    """
    if hier_cat_features is None:
        hier_cat_features = HIERARCHICAL_CAT_FEATURES
    if hier_all_features is None:
        hier_all_features = _ALL_HIER_FEATURES_ORDERED

    needs_mcc = "mcc_code_int" in hier_cat_features
    hier_parquet_extra = [c for c in hier_cat_features if c != "mcc_code_int"]

    # Build the set of parquet columns to read
    feat_set  = set(feature_cols)
    hier_extra_set = set(hier_parquet_extra)
    base_cols  = {"event_id", "event_dttm", "is_train", "model_group"}
    if needs_mcc:
        base_cols.add("mcc_code")
    all_needed = feat_set | hier_extra_set | base_cols

    sample_unlabeled = unlabeled_sample_ratio > 0.0
    green_denom = max(1, round(1.0 / unlabeled_sample_ratio)) if sample_unlabeled else 0

    parts: list[pd.DataFrame] = []
    for f in files:
        schema     = pl.scan_parquet(f).collect_schema()
        avail_cols = set(schema.names())
        cols_to_read = list(all_needed & avail_cols)

        lf = pl.scan_parquet(f).select(cols_to_read)

        if schema.get("event_dttm") == pl.Utf8:
            lf = lf.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )

        # Filter to val date range
        lf = lf.filter(
            (pl.col("is_train") == 1)
            & (pl.col("event_dttm") >= cutoff)
            & (pl.col("event_dttm") < train_end)
        )

        # Left join with labels → labeled rows get target, rest get null
        lf = lf.join(
            labels_df.lazy().select(["event_id", "target"]),
            on="event_id",
            how="left",
        )

        # is_labeled flag
        lf = lf.with_columns(
            pl.col("target").is_not_null().cast(pl.Int8).alias("_is_labeled")
        )

        # Keep labeled rows + optionally sampled unlabeled
        if sample_unlabeled:
            lf = lf.filter(
                (pl.col("_is_labeled") == 1)
                | ((pl.col("event_id").hash(seed=UNDERSAMPLE_SEED + 77) % green_denom) == 0)
            )
        else:
            lf = lf.filter(pl.col("_is_labeled") == 1)

        # Fill target for unlabeled rows → 0 (for PR-AUC: treat unlabeled as negative)
        lf = lf.with_columns(pl.col("target").fill_null(0).cast(pl.Int8))

        # Derive mcc_code_int
        if needs_mcc and "mcc_code" in avail_cols:
            lf = lf.with_columns(
                pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_int")
            )

        # Fill nulls
        fill_exprs = []
        for c in hier_all_features:
            if c not in avail_cols and c != "mcc_code_int":
                continue
            if c in hier_cat_features:
                fill_exprs.append(pl.col(c).fill_null(-1))
            else:
                fill_exprs.append(pl.col(c).fill_null(0.0))
        if fill_exprs:
            lf = lf.with_columns(fill_exprs)

        chunk = lf.collect()
        drop_c = [c for c in ("is_train", "mcc_code") if c in chunk.columns]
        if drop_c:
            chunk = chunk.drop(drop_c)

        parts.append(chunk.to_pandas())
        del chunk, lf
        gc.collect()

    if not parts:
        raise RuntimeError("No val rows found — check date boundaries.")

    for p in parts:
        p["raw_target"] = np.where(p["_is_labeled"] == 1, p["target"].values, -1).astype(np.int8)

    result = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    n_labeled = int((result["_is_labeled"] == 1).sum())
    n_unlabeled = int((result["_is_labeled"] == 0).sum())
    n_pos = int((result["raw_target"] == 1).sum())
    print(f"  Val loaded: {len(result):,} rows  "
          f"({n_labeled:,} labeled [{n_pos:,} pos], "
          f"{n_unlabeled:,} unlabeled)\n", flush=True)
    return result


# ── Helper: train Recent LightGBM from parquet ────────────────────────────────

def _train_recent_lgbm(
    files: list,
    labels_df: pl.DataFrame,
    recent_border: datetime,
    cutoff: datetime,
    feature_cols: list,
    neg_sample_ratio: float,
    lgbm_params: dict,
    early_stopping_rounds: int,
    val_df: pd.DataFrame | None = None,
    n_rounds_fixed: int | None = None,
) -> tuple:
    """
    Load recent train data (recent_border <= dttm < cutoff, is_train==1)
    from processed parquets and train a single global LightGBM.

    val_df         : labeled val rows — used for early stopping when provided.
    n_rounds_fixed : if set, train for exactly this many rounds (no early stopping).

    Returns (booster, best_iteration).
    """
    print(f"\nLoading recent train data ({recent_border.date()} – {cutoff.date()}) …",
          flush=True)

    # --- Pass 1: scan partitions to find avail_feats and count pos / neg ------
    avail_feats: list[str] | None = None
    total_pos = 0
    total_neg = 0
    wall_times: list = []

    for i, f in enumerate(files):
        t0 = time.perf_counter()
        schema = pl.scan_parquet(f).collect_schema()
        avail  = set(schema.names())

        if avail_feats is None:
            avail_feats = [c for c in feature_cols if c in avail]

        # Light scan: only event_id + date filters to count pos/neg
        meta_cols = [c for c in ("event_id", "event_dttm", "is_train") if c in avail]
        lf = pl.scan_parquet(f).select(meta_cols)
        if schema.get("event_dttm") == pl.Utf8:
            lf = lf.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )
        lf = lf.filter(
            (pl.col("is_train") == 1)
            & (pl.col("event_dttm") >= recent_border)
            & (pl.col("event_dttm") < cutoff)
        )
        lf = (
            lf.join(labels_df.lazy().select(["event_id", "target"]),
                    on="event_id", how="left")
              .with_columns(pl.col("target").fill_null(0).cast(pl.Int8))
        )
        counts = lf.group_by("target").len().collect()
        for row in counts.iter_rows(named=True):
            if row["target"] == 1:
                total_pos += row["len"]
            else:
                total_neg += row["len"]

        del lf, counts
        gc.collect()
        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(files), wall_times, suffix="(counting)")

    print(flush=True)

    if avail_feats is None:
        avail_feats = list(feature_cols)

    n_neg_keep = max(int(total_neg * neg_sample_ratio), total_pos)
    n_neg_keep = min(n_neg_keep, total_neg)
    effective_neg_ratio = n_neg_keep / total_neg if total_neg > 0 else 1.0

    # --- Pass 2: load partitions, undersample per-partition, keep features ----
    green_denom = max(1, round(1.0 / effective_neg_ratio))

    print(f"  Counts : {total_pos:,} pos, {total_neg:,} neg → keep ~1/{green_denom} neg",
          flush=True)
    print(f"\nLoading + undersampling …", flush=True)

    sampled_parts: list[pd.DataFrame] = []
    kept_pos = 0
    kept_neg = 0
    wall_times = []

    for i, f in enumerate(files):
        t0 = time.perf_counter()
        schema = pl.scan_parquet(f).collect_schema()
        avail  = set(schema.names())
        cols   = [c for c in (avail_feats + ["event_id", "event_dttm", "is_train"]) if c in avail]

        lf = pl.scan_parquet(f).select(cols)
        if schema.get("event_dttm") == pl.Utf8:
            lf = lf.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )
        lf = lf.filter(
            (pl.col("is_train") == 1)
            & (pl.col("event_dttm") >= recent_border)
            & (pl.col("event_dttm") < cutoff)
        )
        lf = (
            lf.join(labels_df.lazy().select(["event_id", "target"]),
                    on="event_id", how="left")
              .with_columns(pl.col("target").fill_null(0).cast(pl.Int8))
        )

        # Keep all positives + deterministic subsample of negatives
        lf = lf.filter(
            (pl.col("target") == 1)
            | ((pl.col("event_id").hash(seed=UNDERSAMPLE_SEED) % green_denom) == 0)
        )

        # Only keep feature columns + target (drop event_id, event_dttm, is_train)
        keep_cols = [c for c in avail_feats if c in avail] + ["target"]
        chunk = lf.select(keep_cols).collect()
        pdf = chunk.to_pandas()

        kept_pos += int((pdf["target"] == 1).sum())
        kept_neg += int((pdf["target"] == 0).sum())

        sampled_parts.append(pdf)
        del chunk, lf, pdf
        gc.collect()
        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(files), wall_times)

    print(flush=True)

    df = pd.concat(sampled_parts, ignore_index=True)
    del sampled_parts
    gc.collect()

    y_tr = df["target"].values.astype(np.int8)
    X_tr = df[avail_feats].to_numpy().astype(np.float32)
    del df
    gc.collect()

    w_tr = np.ones(len(y_tr), dtype=np.float32)
    w_tr[y_tr == 0] = 1.0 / neg_sample_ratio

    print(f"  Recent train : {kept_pos:,} pos + {kept_neg:,} neg kept "
          f"(of {total_neg:,}, ratio {neg_sample_ratio:.0%})", flush=True)

    use_early_stopping = (n_rounds_fixed is None) and (val_df is not None)
    num_rounds = n_rounds_fixed if n_rounds_fixed is not None else lgbm_params.get("n_estimators", 2000)

    params = {**lgbm_params, "seed": 42, "n_estimators": num_rounds}
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                         feature_name=avail_feats, free_raw_data=True)
    del X_tr, y_tr, w_tr

    if use_early_stopping:
        avail_val = [c for c in avail_feats if c in val_df.columns]
        # Use only labeled rows for early stopping
        if "_is_labeled" in val_df.columns:
            val_sub = val_df[val_df["_is_labeled"] == 1]
        else:
            val_sub = val_df
        X_val_lbl = val_sub[avail_val].to_numpy().astype(np.float32)
        y_val_lbl = val_sub["target"].values.astype(np.int8)
        dval = lgb.Dataset(X_val_lbl, label=y_val_lbl,
                           feature_name=avail_val, reference=dtrain, free_raw_data=True)
        valid_sets  = [dval]
        valid_names = ["val"]
        callbacks   = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ]
    else:
        valid_sets  = []
        valid_names = []
        callbacks   = [lgb.log_evaluation(period=50)]
        dval        = None

    print("\nTraining Recent LightGBM …", flush=True)
    t0 = time.perf_counter()
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    print(f"\n  Recent LGBM | training time : {_fmt(time.perf_counter() - t0)}")
    print(f"  Recent LGBM | best_iter     : {booster.best_iteration}")
    if use_early_stopping:
        print(f"  Recent LGBM | val PR-AUC    : "
              f"{booster.best_score['val']['average_precision']:.6f}")
    print(flush=True)

    best_iter = booster.best_iteration if use_early_stopping else num_rounds
    if dval is not None:
        del dval
    gc.collect()
    return booster, best_iter


# ── Helper: score test rows with all models → submission CSV ──────────────────

def _score_test_v2(
    lgbm_boosters: dict,
    susp_model: CatBoostClassifier,
    rgs_model: CatBoostClassifier,
    main_model: CatBoostClassifier | None,
    recent_booster,
    feature_cols: list,
    hier_all_features: list,
    hier_cat_features: list,
    submission_path: str,
    w_lgbm: float,
    w_hier: float,
    w_main: float = 0.0,
    w_recent: float = 0.0,
    blend_logit: bool = True,
    chunk_size: int = 500_000,
) -> None:
    """
    Score test rows (is_train==1, event_dttm >= TRAIN_END_DATE) with all models,
    blend scores, and write submission CSV.
    """
    test_files  = _parquet_files(FEATURES_DIR)
    train_end   = datetime.fromisoformat(TRAIN_END_DATE)
    needs_mcc   = "mcc_code_int" in hier_cat_features

    print(f"\nScoring test set ({len(test_files)} partitions) …", flush=True)
    print(f"  Blend : w_lgbm={w_lgbm:.3f}  w_hier={w_hier:.3f}  "
          f"w_main={w_main:.3f}  w_recent={w_recent:.3f}  "
          f"(logit={blend_logit})\n", flush=True)

    # Columns needed from parquet
    hier_extra_parquet = [c for c in hier_cat_features if c != "mcc_code_int"]
    hier_num_needed    = [c for c in hier_all_features if c not in hier_cat_features]
    all_needed_set = (
        set(feature_cols)
        | set(hier_extra_parquet)
        | set(hier_num_needed)
        | {"event_id", "event_dttm", "is_train", "model_group"}
    )
    if needs_mcc:
        all_needed_set.add("mcc_code")

    all_event_ids: list[np.ndarray] = []
    all_scores:    list[np.ndarray] = []
    wall_times:    list[float]      = []
    total_rows = 0

    for i, f in enumerate(test_files):
        t0 = time.perf_counter()

        schema     = pl.scan_parquet(f).collect_schema()
        avail_cols = set(schema.names())
        cols_to_read = list(all_needed_set & avail_cols)

        chunk = pl.read_parquet(f, columns=cols_to_read)
        if chunk["event_dttm"].dtype == pl.Utf8:
            chunk = chunk.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )
        test_chunk = chunk.filter(
            (pl.col("is_train") == 1) & (pl.col("event_dttm") >= train_end)
        )
        del chunk
        gc.collect()

        if len(test_chunk) == 0:
            wall_times.append(time.perf_counter() - t0)
            _progress(i + 1, len(test_files), wall_times,
                      suffix=f"test rows {total_rows:,}")
            del test_chunk
            continue

        # Derive mcc_code_int for hier models
        if needs_mcc and "mcc_code" in test_chunk.columns:
            test_chunk = test_chunk.with_columns(
                pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_int")
            )

        # Fill nulls for hier features
        fill_exprs = []
        for c in hier_all_features:
            if c not in test_chunk.columns:
                continue
            if c in hier_cat_features:
                fill_exprs.append(pl.col(c).fill_null(-1))
            else:
                fill_exprs.append(pl.col(c).fill_null(0.0))
        if fill_exprs:
            test_chunk = test_chunk.with_columns(fill_exprs)

        event_ids = test_chunk["event_id"].to_numpy()
        tg_arr    = test_chunk["model_group"].to_numpy().astype(np.int8)
        n_rows    = len(test_chunk)

        avail_feats     = [c for c in feature_cols if c in test_chunk.columns]
        avail_hier      = [c for c in hier_all_features if c in test_chunk.columns]
        hier_cat_avail  = [c for c in hier_cat_features if c in avail_hier]
        hier_cat_idx    = [avail_hier.index(c) for c in hier_cat_avail]

        final_scores = np.zeros(n_rows, dtype=np.float32)

        # Score in sub-chunks to stay within RAM
        for cs in range(0, n_rows, chunk_size):
            ce      = min(cs + chunk_size, n_rows)
            sub_df  = test_chunk.slice(cs, ce - cs)
            sub_tg  = tg_arr[cs:ce]

            # ── LightGBM: per-group (single seed per group) ──────────────────
            X_lgbm  = sub_df.select(avail_feats).to_numpy().astype(np.float32)
            lgbm_sc = np.zeros(ce - cs, dtype=np.float64)
            for group_id, boosters in lgbm_boosters.items():
                g_mask = sub_tg == group_id
                if not g_mask.any():
                    continue
                X_g = X_lgbm[g_mask]
                s_sum = np.zeros(g_mask.sum(), dtype=np.float64)
                for b in boosters:
                    s_sum += b.predict(X_g, num_iteration=b.best_iteration).astype(np.float64)
                lgbm_sc[g_mask] = (s_sum / len(boosters))
                del X_g, s_sum
            del X_lgbm
            gc.collect()

            # ── Hierarchical: sigmoid product ────────────────────────────────
            sub_pd  = sub_df.to_pandas()
            X_hier  = sub_pd[avail_hier]
            pool    = Pool(X_hier, cat_features=hier_cat_idx)

            susp_logit = susp_model.predict(pool, prediction_type="RawFormulaVal").astype(np.float64)
            rgs_logit  = rgs_model.predict(pool, prediction_type="RawFormulaVal").astype(np.float64)
            hier_prob  = (_sigmoid(susp_logit) * _sigmoid(rgs_logit)).astype(np.float64)

            # ── Main CatBoost ────────────────────────────────────────────────
            main_logit_val = np.zeros(ce - cs, dtype=np.float64)
            if main_model is not None and w_main > 0:
                main_logit_val = main_model.predict(pool, prediction_type="RawFormulaVal").astype(np.float64)

            del X_hier, pool, sub_pd
            gc.collect()

            # ── Recent LightGBM (optional) ────────────────────────────────────
            recent_sc = None
            if recent_booster is not None and w_recent > 0.0:
                avail_recent = [c for c in feature_cols if c in sub_df.columns]
                X_rec = sub_df.select(avail_recent).to_numpy().astype(np.float32)
                recent_sc = recent_booster.predict(
                    X_rec, num_iteration=recent_booster.best_iteration
                ).astype(np.float64)
                del X_rec
                gc.collect()

            # ── Blend ─────────────────────────────────────────────────────────
            if blend_logit:
                lgbm_logit = _logit(np.clip(lgbm_sc, 1e-7, 1 - 1e-7))
                hier_logit = _logit(np.clip(hier_prob, 1e-7, 1 - 1e-7))
                blended = w_lgbm * lgbm_logit + w_hier * hier_logit
                if w_main > 0:
                    blended = blended + w_main * main_logit_val
                if recent_sc is not None:
                    recent_logit = _logit(np.clip(recent_sc, 1e-7, 1 - 1e-7))
                    blended = blended + w_recent * recent_logit
                # Convert back to probability for submission
                blended = _sigmoid(blended)
            else:
                blended = w_lgbm * lgbm_sc + w_hier * hier_prob
                if w_main > 0:
                    blended = blended + w_main * _sigmoid(main_logit_val)
                if recent_sc is not None:
                    blended = blended + w_recent * recent_sc
            final_scores[cs:ce] = blended.astype(np.float32)

            del sub_df, lgbm_sc, hier_prob, blended
            if recent_sc is not None:
                del recent_sc
            gc.collect()

        all_event_ids.append(event_ids)
        all_scores.append(final_scores)
        total_rows += n_rows

        del test_chunk
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(test_files), wall_times, suffix=f"test rows {total_rows:,}")

    print(flush=True)

    if total_rows == 0:
        raise RuntimeError("No test rows found. Check TRAIN_END_DATE and parquet files.")

    event_ids_all = np.concatenate(all_event_ids)
    scores_all    = np.concatenate(all_scores)
    del all_event_ids, all_scores
    gc.collect()

    submission = pl.DataFrame({"event_id": event_ids_all, "predict": scores_all})
    n_dupes = len(submission) - submission["event_id"].n_unique()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes:,} duplicate event_ids in submission.", flush=True)

    submission.write_csv(submission_path)
    print(f"  Test rows scored   : {total_rows:,}")
    print(f"  Submission written : {submission_path}")
    for pct, lbl in [(0, "min"), (50, "median"), (95, "p95"), (99, "p99"), (100, "max")]:
        print(f"    {lbl:6s}: {float(np.percentile(scores_all, pct)):.6f}")


# ── Helper: compute blended score in logit or probability space ───────────────

def _blend_scores(
    lgbm_prob, hier_prob, main_logit, recent_prob,
    w_lgbm, w_hier, w_main, w_recent,
    blend_logit=True,
):
    """Blend model outputs.  Returns the blended score (probability-scale)."""
    if blend_logit:
        lgbm_l = _logit(np.clip(lgbm_prob, 1e-7, 1 - 1e-7))
        hier_l = _logit(np.clip(hier_prob, 1e-7, 1 - 1e-7))
        bl = w_lgbm * lgbm_l + w_hier * hier_l
        if w_main > 0 and main_logit is not None:
            bl = bl + w_main * main_logit  # already in logit space
        if w_recent > 0 and recent_prob is not None:
            bl = bl + w_recent * _logit(np.clip(recent_prob, 1e-7, 1 - 1e-7))
        return bl  # keep in logit space for ranking (monotonic transform)
    else:
        bl = w_lgbm * lgbm_prob + w_hier * hier_prob
        if w_main > 0 and main_logit is not None:
            bl = bl + w_main * _sigmoid(main_logit)
        if w_recent > 0 and recent_prob is not None:
            bl = bl + w_recent * recent_prob
        return bl


# ── Main orchestrator ──────────────────────────────────────────────────────────

def train_v2(gpu: bool = False) -> None:
    """
    Hierarchical ensemble: per-group LGBM + global hierarchical CatBoost
    + Main CatBoost + optional recent LGBM.
    Logit-space blending, all-rows blend optimisation.

    Parameters
    ----------
    gpu : bool
        When True, override CatBoost task_type to "GPU" and LightGBM
        device_type to "gpu".  Requires GPU-enabled CatBoost/LightGBM builds.
    """
    total_start = time.perf_counter()
    cutoff      = datetime.fromisoformat(VAL_CUTOFF_DATE)
    train_end   = datetime.fromisoformat(TRAIN_END_DATE)
    recent_bdr  = datetime.fromisoformat(RECENT_BORDER)

    # ── GPU overrides ────────────────────────────────────────────────────────
    device_tag = "GPU" if gpu else "CPU"

    susp_params = {**SUSPICIOUS_CATBOOST_PARAMS}
    rgs_params  = {**RGS_CATBOOST_PARAMS}
    main_params = {**MAIN_CATBOOST_PARAMS}
    lgbm_params_base   = {**LGBM_PARAMS}
    lgbm_params_recent = {**LGBM_PARAMS_RECENT}

    if gpu:
        for d in (susp_params, rgs_params, main_params):
            d["task_type"] = "GPU"
        lgbm_params_base["device_type"]   = "gpu"
        lgbm_params_recent["device_type"] = "gpu"

    _prepare_models_dir(MODELS_DIR)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")
    print(f"Found {len(files)} parquet partitions in {FEATURES_DIR!r}\n", flush=True)
    print(f"Device                : {device_tag}")
    print(f"LGBM ensemble seeds   : {ENSEMBLE_SEEDS}")
    print(f"Yellow weight         : {YELLOW_WEIGHT_MULTIPLIER}")
    print(f"Hier full features    : {HIER_USE_ALL_FEATURES}")
    print(f"Blend in logit space  : {BLEND_IN_LOGIT_SPACE}")
    print(f"Blend on all rows     : {BLEND_ON_ALL_ROWS}")
    print(f"Train recent LGBM     : {TRAIN_RECENT_LGBM}")
    print(f"Green weights         : old={SUSPICIOUS_GREEN_OLD_W}  recent={SUSPICIOUS_GREEN_RECENT_W}\n",
          flush=True)

    # ── Step 1: Build or reload memmap split files ────────────────────────────
    print("=" * 65)
    print("  Step 1 — Build memmap splits")
    print("=" * 65 + "\n")

    cached = _load_cache(STAGING_DIR)
    if cached is not None:
        (X_train, y_train, X_val, y_val, il_val,
         tg_train, tg_val, il_train, feature_cols) = cached
    else:
        labels = load_labels()
        n_train, n_val = count_rows(files, cutoff, train_end)
        (X_train, y_train, X_val, y_val, il_val,
         tg_train, tg_val, il_train, feature_cols) = build_memmaps(
            files, labels, cutoff, train_end, n_train, n_val, STAGING_DIR
        )
        del labels
        gc.collect()

    print(f"\n  Feature columns : {len(feature_cols)}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}\n")

    tg_train_np = np.asarray(tg_train)
    tg_val_np   = np.asarray(tg_val)
    il_val_np   = np.asarray(il_val)
    y_val_np    = np.asarray(y_val)

    # ── Derive hierarchical feature lists at runtime ─────────────────────────
    if HIER_USE_ALL_FEATURES:
        # Use the same full feature set as LGBM + extra categoricals
        hier_cat_features = list(HIERARCHICAL_CAT_FEATURES)  # includes customer_id, mcc_code_int
        hier_num_features = [c for c in feature_cols if c not in set(hier_cat_features)]
        hier_all_features = list(dict.fromkeys(hier_num_features + hier_cat_features))
        green_ratio = HIER_FULL_GREEN_RATIO
        print(f"  Hier features (full) : {len(hier_all_features)} "
              f"({len(hier_num_features)} num + {len(hier_cat_features)} cat)")
        print(f"  Green ratio          : {green_ratio}\n")
    else:
        hier_cat_features = list(HIERARCHICAL_CAT_FEATURES)
        hier_num_features = list(HIERARCHICAL_NUM_FEATURES)
        hier_all_features = list(dict.fromkeys(hier_num_features + hier_cat_features))
        green_ratio = SUSPICIOUS_GREEN_RATIO
        print(f"  Hier features (curated) : {len(hier_all_features)} "
              f"({len(hier_num_features)} num + {len(hier_cat_features)} cat)\n")

    # ── Step 2: Pre-load labeled val from parquet (for blend optimization) ────
    print("=" * 65)
    print("  Step 2 — Pre-load val from parquet")
    print("=" * 65 + "\n")

    labels_df = load_labels()

    unlabeled_ratio = BLEND_UNLABELED_SAMPLE_RATIO if BLEND_ON_ALL_ROWS else 0.0
    val_df = _load_labeled_val(
        files, labels_df, cutoff, train_end, feature_cols,
        hier_cat_features=hier_cat_features,
        hier_all_features=hier_all_features,
        unlabeled_sample_ratio=unlabeled_ratio,
    )

    # Separate labeled rows for model-level evaluation
    labeled_mask_val = val_df["_is_labeled"].values == 1
    labeled_val_df   = val_df[labeled_mask_val].copy()

    _avail_feats  = [c for c in feature_cols if c in val_df.columns]
    X_lbl_full    = val_df[_avail_feats].to_numpy().astype(np.float32)
    y_blend       = val_df["target"].values.astype(np.int8)   # labeled fraud=1, everything else=0
    y_lbl         = labeled_val_df["target"].values.astype(np.int8)
    mg_lbl        = val_df["model_group"].values.astype(np.int8)
    n_blend       = len(val_df)
    n_lbl         = len(labeled_val_df)

    lgbm_blend_sum = np.zeros(n_blend, dtype=np.float64)

    # ── Step 3: Per-group LightGBM ───────────────────────────────────────────
    print("=" * 65)
    print("  Step 3 — Per-group LightGBM")
    print("=" * 65 + "\n")

    best_iters_lgbm: dict = {}

    for group_id, group_name in TX_TYPE_GROUPS.items():
        sep = "─" * 65
        print(f"\n{sep}")
        print(f"  Group {group_id} — {group_name.upper()}")
        print(f"{sep}\n")

        train_mask        = tg_train_np == group_id
        labeled_val_mask  = (tg_val_np == group_id) & (il_val_np == 1)
        n_pos_train       = int(y_train[train_mask].sum()) if train_mask.any() else 0
        n_labeled_val     = int(labeled_val_mask.sum())

        print(f"  Train rows    : {int(train_mask.sum()):,}  ({n_pos_train:,} pos)")
        print(f"  Labeled val   : {n_labeled_val:,}  "
              f"({int(y_val_np[labeled_val_mask].sum()):,} pos)\n")

        if not train_mask.any() or n_pos_train == 0:
            print(f"  SKIP: no training data for group {group_id}.\n")
            continue

        lv_idx          = np.where(labeled_val_mask)[0]
        X_val_lbl_g     = np.array(X_val[lv_idx])
        y_val_lbl_g     = np.array(y_val[lv_idx])
        il_val_ones_g   = np.ones(len(lv_idx), dtype=np.int8)

        group_neg_ratio   = NEG_SAMPLE_RATIO_BY_GROUP.get(group_id, NEG_SAMPLE_RATIO)
        group_lgbm_params = {**lgbm_params_base, **LGBM_PARAMS_BY_GROUP.get(group_id, {})}
        group_es_rounds   = EARLY_STOPPING_ROUNDS_BY_GROUP.get(group_id, None)

        blend_g_mask = (mg_lbl == group_id)
        seed_best_iters: list[int] = []

        for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
            print(f"\n  ── LGBM seed {seed_idx + 1}/{len(ENSEMBLE_SEEDS)} "
                  f"(seed={seed}) ──────────────────────────────────────────")
            booster = train_model(
                X_train, y_train,
                X_val_lbl_g, y_val_lbl_g, il_val_ones_g,
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

            if seed_idx == 0:
                evaluate(
                    booster, X_val, y_val_np, il_val_np, feature_cols,
                    row_mask=(tg_val_np == group_id),
                    label=f"{group_name} seed-0",
                )

            # Score ALL val rows in this group (for blend optimization)
            X_g = X_lbl_full[blend_g_mask]
            s = booster.predict(X_g, num_iteration=booster.best_iteration).astype(np.float64)
            lgbm_blend_sum[blend_g_mask] += s
            del booster, X_g, s
            gc.collect()

        best_iters_lgbm[group_id] = seed_best_iters
        del X_val_lbl_g, y_val_lbl_g, il_val_ones_g, lv_idx
        gc.collect()

    lgbm_blend_avg = (lgbm_blend_sum / len(ENSEMBLE_SEEDS)).astype(np.float32)
    del lgbm_blend_sum
    gc.collect()

    pr_auc_lgbm = average_precision_score(y_lbl, lgbm_blend_avg[labeled_mask_val])
    print(f"\n  LGBM only PR-AUC on labeled val : {pr_auc_lgbm:.6f}\n", flush=True)

    # ── Step 4: Load hierarchical train data from parquet ─────────────────────
    print("=" * 65)
    print("  Step 4 — Load hierarchical train data")
    print("=" * 65 + "\n")

    hier_train_df = load_hier_split(
        files       = files,
        labels_df   = labels_df,
        cutoff_lo   = datetime(2000, 1, 1),
        cutoff_hi   = cutoff,
        hier_num_features = hier_num_features,
        hier_cat_features = hier_cat_features,
        green_ratio = green_ratio,
        recent_border = recent_bdr,
        verbose     = True,
    )
    print(f"  Hier train data shape : {hier_train_df.shape}", flush=True)

    # ── Step 5: Suspicious CatBoost  P(labeled | tx) ─────────────────────────
    print("\n" + "=" * 65)
    print("  Step 5 — Suspicious CatBoost  P(labeled | tx)")
    print("=" * 65 + "\n")

    susp_model_val, susp_best_iter, _ = train_suspicious(
        train_df        = hier_train_df,
        val_df          = val_df,
        all_features    = hier_all_features,
        cat_features    = hier_cat_features,
        params          = susp_params,
        labeled_weight  = SUSPICIOUS_LABELED_WEIGHT,
        green_recent_weight = SUSPICIOUS_GREEN_RECENT_W,
        green_old_weight    = SUSPICIOUS_GREEN_OLD_W,
        recent_border   = recent_bdr,
    )

    # ── Step 6: RGS CatBoost  P(fraud | labeled, tx) ────────────────────────
    print("\n" + "=" * 65)
    print("  Step 6 — RGS CatBoost  P(fraud | labeled, tx)")
    print("=" * 65 + "\n")

    rgs_model_val, rgs_best_iter, _ = train_rgs(
        train_df        = hier_train_df,
        val_labeled_df  = labeled_val_df,
        all_features    = hier_all_features,
        cat_features    = hier_cat_features,
        params          = rgs_params,
        red_weight      = RGS_RED_WEIGHT,
        yellow_weight   = RGS_YELLOW_WEIGHT,
    )

    # ── Step 6b: Main CatBoost (direct fraud) ────────────────────────────────
    print("\n" + "=" * 65)
    print("  Step 6b — Main CatBoost (direct fraud)")
    print("=" * 65 + "\n")

    main_model_val, main_best_iter, _ = train_main_catboost(
        train_df        = hier_train_df,
        val_labeled_df  = labeled_val_df,
        all_features    = hier_all_features,
        cat_features    = hier_cat_features,
        params          = main_params,
        red_weight      = MAIN_RED_WEIGHT,
        yellow_weight   = MAIN_YELLOW_WEIGHT,
        green_recent_weight = MAIN_GREEN_RECENT_W,
        green_old_weight    = MAIN_GREEN_OLD_W,
        recent_border   = recent_bdr,
    )

    # Release hier train data
    del hier_train_df
    gc.collect()

    # ── Step 7: Score val with all CatBoost models ───────────────────────────
    print("=" * 65)
    print("  Step 7 — Score val with hierarchical + main models")
    print("=" * 65 + "\n")

    avail_hier_val = [c for c in hier_all_features if c in val_df.columns]
    hier_cat_avail = [c for c in hier_cat_features if c in avail_hier_val]
    hier_cat_idx   = [avail_hier_val.index(c) for c in hier_cat_avail]
    val_pool = Pool(val_df[avail_hier_val], cat_features=hier_cat_idx)

    # Suspicious
    susp_logit_avg = susp_model_val.predict(
        val_pool, prediction_type="RawFormulaVal"
    ).astype(np.float32)

    # RGS
    rgs_logit_avg = rgs_model_val.predict(
        val_pool, prediction_type="RawFormulaVal"
    ).astype(np.float32)

    hier_blend_prob = (_sigmoid(susp_logit_avg) * _sigmoid(rgs_logit_avg)).astype(np.float32)

    # Main CatBoost
    main_logit_avg = main_model_val.predict(
        val_pool, prediction_type="RawFormulaVal"
    ).astype(np.float32)

    del val_pool
    gc.collect()

    # Report individual model PR-AUC on labeled val
    pr_auc_hier = average_precision_score(y_lbl, hier_blend_prob[labeled_mask_val])
    pr_auc_main = average_precision_score(y_lbl, _sigmoid(main_logit_avg[labeled_mask_val]))
    print(f"  Hierarchical PR-AUC on labeled val : {pr_auc_hier:.6f}")
    print(f"  Main CatBoost PR-AUC on labeled val: {pr_auc_main:.6f}\n", flush=True)

    # ── Step 8: Optional — Train Recent LightGBM ─────────────────────────────
    recent_booster_val  = None
    recent_best_iter    = None
    recent_blend_prob   = None
    pr_auc_recent       = 0.0

    if TRAIN_RECENT_LGBM:
        print("=" * 65)
        print("  Step 8 — Train Recent LightGBM")
        print("=" * 65 + "\n")

        recent_booster_val, recent_best_iter = _train_recent_lgbm(
            files             = files,
            labels_df         = labels_df,
            recent_border     = recent_bdr,
            cutoff            = cutoff,
            feature_cols      = feature_cols,
            neg_sample_ratio  = RECENT_NEG_SAMPLE_RATIO,
            lgbm_params       = lgbm_params_recent,
            early_stopping_rounds = EARLY_STOPPING_ROUNDS,
            val_df            = val_df,
        )
        # Score all val rows
        avail_recent = [c for c in feature_cols if c in val_df.columns]
        X_rec_val = val_df[avail_recent].to_numpy().astype(np.float32)
        recent_blend_prob = recent_booster_val.predict(
            X_rec_val, num_iteration=recent_booster_val.best_iteration
        ).astype(np.float32)
        del X_rec_val
        gc.collect()

        pr_auc_recent = average_precision_score(y_lbl, recent_blend_prob[labeled_mask_val])
        print(f"  Recent LGBM PR-AUC on labeled val : {pr_auc_recent:.6f}\n", flush=True)
    else:
        print("  Step 8 skipped (TRAIN_RECENT_LGBM = False)\n", flush=True)

    # ── Step 9: Blend weight grid search ─────────────────────────────────────
    print("=" * 65)
    print("  Step 9 — Blend weight grid search")
    print("=" * 65 + "\n")

    # Choose which target/scores to optimise on
    if BLEND_ON_ALL_ROWS:
        opt_y = y_blend          # all loaded val rows (labeled + sampled unlabeled)
        opt_lgbm = lgbm_blend_avg
        opt_hier = hier_blend_prob
        opt_main = main_logit_avg
        opt_recent = recent_blend_prob
        opt_label = "all val rows"
    else:
        opt_y = y_lbl
        opt_lgbm = lgbm_blend_avg[labeled_mask_val]
        opt_hier = hier_blend_prob[labeled_mask_val]
        opt_main = main_logit_avg[labeled_mask_val]
        opt_recent = recent_blend_prob[labeled_mask_val] if recent_blend_prob is not None else None
        opt_label = "labeled val only"

    print(f"  Optimising on : {opt_label} ({len(opt_y):,} rows, "
          f"{int(opt_y.sum()):,} positives)\n", flush=True)

    best_prauc_blend = 0.0
    best_w_lgbm = best_w_hier = best_w_main = best_w_recent = 0.0

    has_recent = TRAIN_RECENT_LGBM and opt_recent is not None
    step = 0.05

    if has_recent:
        # 4D grid: (w_lgbm, w_hier, w_main, w_recent), constrained to sum = 1
        for w_l in np.arange(0.0, 1.0 + step / 2, step):
            for w_h in np.arange(0.0, 1.0 - w_l + step / 2, step):
                for w_m in np.arange(0.0, 1.0 - w_l - w_h + step / 2, step):
                    w_r = 1.0 - w_l - w_h - w_m
                    if w_r < -0.001:
                        continue
                    w_r = max(0.0, w_r)
                    blended = _blend_scores(
                        opt_lgbm, opt_hier, opt_main, opt_recent,
                        w_l, w_h, w_m, w_r,
                        blend_logit=BLEND_IN_LOGIT_SPACE,
                    )
                    p = average_precision_score(opt_y, blended)
                    if p > best_prauc_blend:
                        best_prauc_blend = p
                        best_w_lgbm, best_w_hier = float(w_l), float(w_h)
                        best_w_main, best_w_recent = float(w_m), float(w_r)
    else:
        # 3D grid: (w_lgbm, w_hier, w_main), constrained to sum = 1
        for w_l in np.arange(0.0, 1.0 + step / 2, step):
            for w_h in np.arange(0.0, 1.0 - w_l + step / 2, step):
                w_m = 1.0 - w_l - w_h
                if w_m < -0.001:
                    continue
                w_m = max(0.0, w_m)
                blended = _blend_scores(
                    opt_lgbm, opt_hier, opt_main, None,
                    w_l, w_h, w_m, 0.0,
                    blend_logit=BLEND_IN_LOGIT_SPACE,
                )
                p = average_precision_score(opt_y, blended)
                if p > best_prauc_blend:
                    best_prauc_blend = p
                    best_w_lgbm, best_w_hier = float(w_l), float(w_h)
                    best_w_main = float(w_m)

    if best_prauc_blend == 0.0:
        best_w_lgbm, best_w_hier, best_w_main, best_w_recent = 0.1, 0.5, 0.3, 0.1

    print(f"  Grid search result  : w_lgbm={best_w_lgbm:.2f}  "
          f"w_hier={best_w_hier:.2f}  w_main={best_w_main:.2f}  "
          f"w_recent={best_w_recent:.2f}")
    print(f"  Best blend PR-AUC   : {best_prauc_blend:.6f}  ({opt_label})")

    # Also report labeled-only PR-AUC for comparison
    if BLEND_ON_ALL_ROWS:
        lbl_blend = _blend_scores(
            lgbm_blend_avg[labeled_mask_val],
            hier_blend_prob[labeled_mask_val],
            main_logit_avg[labeled_mask_val],
            recent_blend_prob[labeled_mask_val] if recent_blend_prob is not None else None,
            best_w_lgbm, best_w_hier, best_w_main, best_w_recent,
            blend_logit=BLEND_IN_LOGIT_SPACE,
        )
        pr_auc_lbl_blend = average_precision_score(y_lbl, lbl_blend)
        print(f"  Labeled-only PR-AUC : {pr_auc_lbl_blend:.6f}")

    print(f"\n  Model PRs on labeled val:")
    print(f"    LGBM alone          : {pr_auc_lgbm:.6f}")
    print(f"    Hierarchical alone  : {pr_auc_hier:.6f}")
    print(f"    Main CatBoost alone : {pr_auc_main:.6f}")
    if has_recent:
        print(f"    Recent LGBM alone   : {pr_auc_recent:.6f}")
    print(f"    Best blend          : {best_prauc_blend:.6f}\n", flush=True)

    del lgbm_blend_avg, hier_blend_prob, main_logit_avg, recent_blend_prob
    del X_lbl_full, val_df, labeled_val_df
    gc.collect()

    # ── Step 10: Full-data retraining ─────────────────────────────────────────
    print("=" * 65)
    print("  Step 10 — Full-data retraining  (train + labeled val)")
    print("=" * 65 + "\n")

    lgbm_boosters_full: dict = {}

    # ── 10a. Per-group LGBM (full retrain from memmaps) ──────────────────────
    for group_id, group_name in TX_TYPE_GROUPS.items():
        if group_id not in best_iters_lgbm:
            continue

        print(f"\n  ── LGBM full retrain: Group {group_id} — {group_name.upper()}")

        train_mask       = tg_train_np == group_id
        labeled_val_mask = (tg_val_np == group_id) & (il_val_np == 1)
        group_neg_ratio  = NEG_SAMPLE_RATIO_BY_GROUP.get(group_id, NEG_SAMPLE_RATIO)
        group_lgbm_params = {**lgbm_params_base, **LGBM_PARAMS_BY_GROUP.get(group_id, {})}

        lv_idx_all   = np.where(labeled_val_mask)[0]
        X_val_ext_all = np.array(X_val[lv_idx_all])
        y_val_ext_all = np.array(y_val[lv_idx_all])
        val_pos_idx   = np.where(y_val_ext_all == 1)[0]
        val_neg_idx   = np.where(y_val_ext_all == 0)[0]
        if len(val_neg_idx) > 0:
            n_keep   = max(int(len(val_neg_idx) * group_neg_ratio), len(val_pos_idx))
            n_keep   = min(n_keep, len(val_neg_idx))
            _rng     = np.random.default_rng(UNDERSAMPLE_SEED)
            keep_neg = _rng.choice(val_neg_idx, size=n_keep, replace=False)
            val_keep = np.sort(np.concatenate([val_pos_idx, keep_neg]))
        else:
            val_keep = val_pos_idx
        X_val_ext = X_val_ext_all[val_keep]
        y_val_ext = y_val_ext_all[val_keep]
        del X_val_ext_all, y_val_ext_all, val_pos_idx, val_neg_idx, val_keep
        gc.collect()

        print(f"    Extra val rows: {len(y_val_ext):,} "
              f"({int(y_val_ext.sum())} pos, {int((y_val_ext==0).sum())} neg)")

        _dummy_X  = np.empty((0, len(feature_cols)), dtype=np.float32)
        _dummy_y  = np.empty((0,), dtype=np.int8)
        _dummy_il = np.empty((0,), dtype=np.int8)

        seed_boosters_full: list = []
        for seed_idx, seed in enumerate(ENSEMBLE_SEEDS):
            bi       = best_iters_lgbm[group_id][seed_idx]
            n_rounds = int(bi * RETRAIN_FULL_ITER_FACTOR)
            print(f"\n    seed {seed_idx+1}/{len(ENSEMBLE_SEEDS)} "
                  f"(seed={seed}, rounds={n_rounds})")
            booster = train_model(
                X_train, y_train,
                _dummy_X, _dummy_y, _dummy_il,
                feature_cols,
                train_row_mask=train_mask,
                neg_sample_ratio=group_neg_ratio,
                lgbm_params=group_lgbm_params,
                seed_override=seed,
                retrain_extra=(X_val_ext, y_val_ext),
                n_rounds_fixed=n_rounds,
            )
            path = os.path.join(
                MODELS_DIR,
                LGBM_MODEL_PATH_FMT.format(name=group_name, seed_idx=seed_idx),
            )
            booster.save_model(path)
            print(f"    Saved → {path}")
            seed_boosters_full.append(booster)
            del booster
            gc.collect()

        lgbm_boosters_full[group_id] = seed_boosters_full
        del X_val_ext, y_val_ext
        gc.collect()

    # ── 10b. Hierarchical + Main CatBoost (full retrain from parquet) ────────
    print(f"\n  ── Hierarchical full retrain: load full data "
          f"({datetime(2000,1,1).date()} – {train_end.date()})")

    hier_full_df = load_hier_split(
        files             = files,
        labels_df         = labels_df,
        cutoff_lo         = datetime(2000, 1, 1),
        cutoff_hi         = train_end,
        hier_num_features = hier_num_features,
        hier_cat_features = hier_cat_features,
        green_ratio       = green_ratio,
        recent_border     = recent_bdr,
        verbose           = True,
    )
    print(f"  Hier full data shape : {hier_full_df.shape}", flush=True)

    # Defragment before adding derived columns (avoids PerformanceWarning)
    hier_full_df = hier_full_df.copy()

    avail_hier = [c for c in hier_all_features if c in hier_full_df.columns]
    cat_idx    = [avail_hier.index(c) for c in hier_cat_features if c in avail_hier]

    # ── Suspicious model full retrain ────────────────────────────────────────
    print(f"\n  ── Suspicious model full retrain …", flush=True)
    hier_full_df["__susp_target"] = (
        hier_full_df["raw_target"] != -1
    ).astype(np.int8)

    w_susp = np.where(hier_full_df["__susp_target"] == 1,
                      SUSPICIOUS_LABELED_WEIGHT,
                      SUSPICIOUS_GREEN_OLD_W).astype(np.float32)
    if "is_recent" in hier_full_df.columns:
        is_recent_green = (hier_full_df["__susp_target"] == 0) & (hier_full_df["is_recent"] == 1)
        w_susp[is_recent_green.values] = SUSPICIOUS_GREEN_RECENT_W
    hier_full_df["__susp_weight"] = w_susp

    n_rounds_susp = max(300, int(susp_best_iter * RETRAIN_FULL_ITER_FACTOR))
    _susp_retrain_params = {k: v for k, v in susp_params.items() if k not in ("verbose",)}
    _susp_retrain_params.update({
        "iterations": n_rounds_susp,
        "od_type": "Iter",
        "od_wait": n_rounds_susp + 1,
        "allow_writing_files": False,
    })
    _susp_retrain_params.pop("use_best_model", None)

    susp_pool = Pool(
        hier_full_df[avail_hier],
        label=hier_full_df["__susp_target"].values,
        weight=hier_full_df["__susp_weight"].values,
        cat_features=cat_idx,
    )
    susp_model_full = CatBoostClassifier(**_susp_retrain_params)
    susp_model_full.fit(susp_pool, verbose=100)
    del susp_pool

    susp_path = os.path.join(MODELS_DIR, SUSPICIOUS_MODEL_PATH)
    susp_model_full.save_model(susp_path)
    print(f"  Saved → {susp_path}")
    gc.collect()

    # ── RGS model full retrain ───────────────────────────────────────────────
    print(f"\n  ── RGS model full retrain …", flush=True)
    labeled_full = hier_full_df[hier_full_df["raw_target"] != -1].copy()
    w_rgs = np.where(
        labeled_full["raw_target"].values == 1, RGS_RED_WEIGHT, RGS_YELLOW_WEIGHT
    ).astype(np.float32)

    avail_rgs = [c for c in hier_all_features if c in labeled_full.columns]
    cat_idx_r = [avail_rgs.index(c) for c in hier_cat_features if c in avail_rgs]

    n_rounds_rgs = max(300, int(rgs_best_iter * RETRAIN_FULL_ITER_FACTOR))
    _rgs_retrain_params = {k: v for k, v in rgs_params.items() if k not in ("verbose",)}
    _rgs_retrain_params.update({
        "iterations": n_rounds_rgs,
        "od_type": "Iter",
        "od_wait": n_rounds_rgs + 1,
        "allow_writing_files": False,
    })
    _rgs_retrain_params.pop("use_best_model", None)

    rgs_pool = Pool(
        labeled_full[avail_rgs],
        label=labeled_full["raw_target"].values.astype(np.int8),
        weight=w_rgs,
        cat_features=cat_idx_r,
    )
    rgs_model_full = CatBoostClassifier(**_rgs_retrain_params)
    rgs_model_full.fit(rgs_pool, verbose=100)
    del rgs_pool, labeled_full
    gc.collect()

    rgs_path = os.path.join(MODELS_DIR, RGS_MODEL_PATH)
    rgs_model_full.save_model(rgs_path)
    print(f"  Saved → {rgs_path}")
    gc.collect()

    # ── Main CatBoost model full retrain ─────────────────────────────────────
    print(f"\n  ── Main CatBoost model full retrain …", flush=True)
    hier_full_df["__main_target"] = (
        hier_full_df["raw_target"] == 1
    ).astype(np.int8)

    w_main_arr = np.full(len(hier_full_df), MAIN_GREEN_OLD_W, dtype=np.float32)
    w_main_arr[hier_full_df["raw_target"].values == 1] = MAIN_RED_WEIGHT
    w_main_arr[hier_full_df["raw_target"].values == 0] = MAIN_YELLOW_WEIGHT
    if "is_recent" in hier_full_df.columns:
        is_recent_green = (hier_full_df["raw_target"] == -1) & (hier_full_df["is_recent"] == 1)
        w_main_arr[is_recent_green.values] = MAIN_GREEN_RECENT_W
    hier_full_df["__main_weight"] = w_main_arr

    n_rounds_main = max(300, int(main_best_iter * RETRAIN_FULL_ITER_FACTOR))
    _main_retrain_params = {k: v for k, v in main_params.items() if k not in ("verbose",)}
    _main_retrain_params.update({
        "iterations": n_rounds_main,
        "od_type": "Iter",
        "od_wait": n_rounds_main + 1,
        "allow_writing_files": False,
    })
    _main_retrain_params.pop("use_best_model", None)

    main_pool = Pool(
        hier_full_df[avail_hier],
        label=hier_full_df["__main_target"].values,
        weight=hier_full_df["__main_weight"].values,
        cat_features=cat_idx,
    )
    main_model_full = CatBoostClassifier(**_main_retrain_params)
    main_model_full.fit(main_pool, verbose=100)
    del main_pool

    main_path = os.path.join(MODELS_DIR, MAIN_MODEL_PATH)
    main_model_full.save_model(main_path)
    print(f"  Saved → {main_path}")

    del hier_full_df
    gc.collect()

    # ── 10c. Recent LightGBM full retrain ────────────────────────────────────
    recent_booster_full = None
    if TRAIN_RECENT_LGBM and recent_booster_val is not None:
        print("\n  ── Recent LGBM full retrain …", flush=True)
        recent_n_rounds = int(recent_best_iter * RETRAIN_FULL_ITER_FACTOR)
        recent_booster_full, _ = _train_recent_lgbm(
            files             = files,
            labels_df         = labels_df,
            recent_border     = recent_bdr,
            cutoff            = train_end,
            feature_cols      = feature_cols,
            neg_sample_ratio  = RECENT_NEG_SAMPLE_RATIO,
            lgbm_params       = lgbm_params_recent,
            early_stopping_rounds = EARLY_STOPPING_ROUNDS,
            val_df            = None,
            n_rounds_fixed    = recent_n_rounds,
        )
        recent_path = os.path.join(MODELS_DIR, RECENT_MODEL_PATH)
        recent_booster_full.save_model(recent_path)
        print(f"  Recent LGBM saved → {recent_path}", flush=True)

    del susp_model_val, rgs_model_val, main_model_val, recent_booster_val
    gc.collect()

    # ── Step 11: Score test set ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Step 11 — Score test set → submission.csv")
    print("=" * 65 + "\n")

    _score_test_v2(
        lgbm_boosters   = lgbm_boosters_full,
        susp_model      = susp_model_full,
        rgs_model       = rgs_model_full,
        main_model      = main_model_full,
        recent_booster  = recent_booster_full,
        feature_cols    = feature_cols,
        hier_all_features = hier_all_features,
        hier_cat_features = hier_cat_features,
        submission_path = SUBMISSION_PATH,
        w_lgbm          = best_w_lgbm,
        w_hier          = best_w_hier,
        w_main          = best_w_main,
        w_recent        = best_w_recent if TRAIN_RECENT_LGBM else 0.0,
        blend_logit     = BLEND_IN_LOGIT_SPACE,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.perf_counter() - total_start
    print(f"\n{'─' * 65}")
    print(f"  Total wall time     : {_fmt(total_time)}")
    print(f"  Blend PR-AUC (val)  : {best_prauc_blend:.6f}")
    print(f"    LGBM              : {pr_auc_lgbm:.6f}  (w={best_w_lgbm:.2f})")
    print(f"    Hierarchical      : {pr_auc_hier:.6f}  (w={best_w_hier:.2f})")
    print(f"    Main CatBoost     : {pr_auc_main:.6f}  (w={best_w_main:.2f})")
    if TRAIN_RECENT_LGBM and pr_auc_recent > 0:
        print(f"    Recent LGBM       : {pr_auc_recent:.6f}  (w={best_w_recent:.2f})")
    print("─" * 65)
