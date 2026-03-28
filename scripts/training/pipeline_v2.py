"""
pipeline_v2.py — Hierarchical ensemble training pipeline.

New architecture vs pipeline.py:
    1. Per-group multi-seed LightGBM  (memmaps, yellow-weighted negatives)
    2. Hierarchical CatBoost pair     (global, trained from processed parquet):
         - Suspicious  : P(labeled | tx)      — (red | yellow) vs green
         - RGS         : P(fraud | labeled)   — red vs yellow, labeled only
         - Product     : sigmoid(susp) × sigmoid(rgs)  ≈ P(fraud | tx)
    3. Optional: Recent LightGBM     (global, data from RECENT_BORDER onward)
    4. Blend weight optimisation on labeled val (2D / 3D grid search)
    5. Full-data retraining of all models
    6. Test scoring → submission.csv

Entry point:  train_v2()
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
from catboost import Pool
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
)
from scripts.training._utils import _fmt, _parquet_files, _progress
from scripts.training.data import load_labels, count_rows, build_memmaps, _load_cache
from scripts.training.train import train_model
from scripts.training.evaluate import evaluate
from scripts.training.hierarchical import (
    load_hier_split, train_suspicious, train_rgs,
    score_hierarchical, refit_model, _sigmoid,
)
from scripts.training.pipeline import _prepare_models_dir, _score_val_group


# ── Derived config ──────────────────────────────────────────────────────────────
_ALL_HIER_FEATURES_ORDERED = list(dict.fromkeys(
    HIERARCHICAL_NUM_FEATURES + HIERARCHICAL_CAT_FEATURES
))

# "mcc_code_int" is derived at load time from "mcc_code" (String) — not in parquet directly
_NEEDS_MCC_INT      = "mcc_code_int" in HIERARCHICAL_CAT_FEATURES
_HIER_PARQUET_EXTRA = [c for c in HIERARCHICAL_CAT_FEATURES if c != "mcc_code_int"]


# ── Helper: load labeled val rows from parquet ─────────────────────────────────

def _load_labeled_val(
    files: list,
    labels_df: pl.DataFrame,
    cutoff: datetime,
    train_end: datetime,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Load labeled val rows (cutoff ≤ dttm < train_end, is_train==1, in labels_df)
    with LGBM feature columns + hierarchical extra columns.

    Returns a pandas DataFrame with:
        event_id, target, raw_target, model_group,
        <feature_cols>, <hier extra cols not in feature_cols>
    """
    # Build the set of parquet columns to read
    feat_set  = set(feature_cols)
    hier_extra_set = set(_HIER_PARQUET_EXTRA)   # cat features minus mcc_code_int
    # customer_id, model_group are NOT in feature_cols (NON_FEATURE_COLS) but needed
    base_cols  = {"event_id", "event_dttm", "is_train", "model_group"}
    # mcc_code (String) read only if mcc_code_int needed
    if _NEEDS_MCC_INT:
        base_cols.add("mcc_code")

    all_needed = feat_set | hier_extra_set | base_cols

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

        # Inner join with labels → only keep labeled rows; also gets "target" column
        lf = lf.join(
            labels_df.lazy().select(["event_id", "target"]),
            on="event_id",
            how="inner",
        )

        # Derive mcc_code_int
        if _NEEDS_MCC_INT and "mcc_code" in avail_cols:
            lf = lf.with_columns(
                pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_int")
            )

        # Fill nulls
        fill_exprs = []
        for c in _ALL_HIER_FEATURES_ORDERED:
            if c not in avail_cols and c != "mcc_code_int":
                continue
            if c in HIERARCHICAL_CAT_FEATURES:
                fill_exprs.append(pl.col(c).fill_null(-1))
            else:
                fill_exprs.append(pl.col(c).fill_null(0.0))
        if fill_exprs:
            lf = lf.with_columns(fill_exprs)

        chunk = lf.collect()
        # Drop parquet helper cols not needed downstream
        drop_c = [c for c in ("is_train", "mcc_code") if c in chunk.columns]
        if drop_c:
            chunk = chunk.drop(drop_c)

        parts.append(chunk.to_pandas())
        del chunk, lf
        gc.collect()

    if not parts:
        raise RuntimeError("No labeled val rows found — check date boundaries.")

    # Add raw_target before concat to avoid fragmented-DataFrame warning
    for p in parts:
        p["raw_target"] = p["target"].astype(np.int8)

    result = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    n_pos = int((result["target"] == 1).sum())
    print(f"  Labeled val loaded : {len(result):,} rows  "
          f"({n_pos:,} positives / {len(result) - n_pos:,} negatives)\n", flush=True)
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
    Load recent train data (recent_border ≤ dttm < cutoff, is_train==1)
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
    # Use Polars hash-modulo sampling for deterministic, stateless undersampling
    # of negatives. This avoids holding all rows in memory.
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
        X_val_lbl = val_df[avail_val].to_numpy().astype(np.float32)
        y_val_lbl = val_df["target"].values.astype(np.int8)
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
    susp_model,
    rgs_model,
    recent_booster,
    feature_cols: list,
    hier_all_features: list,
    hier_cat_features: list,
    submission_path: str,
    w_lgbm: float,
    w_hier: float,
    w_recent: float = 0.0,
    chunk_size: int = 500_000,
) -> None:
    """
    Score test rows (is_train==1, event_dttm >= TRAIN_END_DATE) with all models,
    blend scores according to (w_lgbm, w_hier, w_recent), and write submission CSV.
    """
    test_files  = _parquet_files(FEATURES_DIR)
    train_end   = datetime.fromisoformat(TRAIN_END_DATE)
    needs_mcc   = _NEEDS_MCC_INT

    print(f"\nScoring test set ({len(test_files)} partitions) …", flush=True)
    print(f"  Blend : w_lgbm={w_lgbm:.3f}  w_hier={w_hier:.3f}  w_recent={w_recent:.3f}\n",
          flush=True)

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

            # ── LightGBM: per-group multi-seed average ────────────────────────
            X_lgbm  = sub_df.select(avail_feats).to_numpy().astype(np.float32)
            lgbm_sc = np.zeros(ce - cs, dtype=np.float32)
            for group_id, boosters in lgbm_boosters.items():
                g_mask = sub_tg == group_id
                if not g_mask.any():
                    continue
                X_g = X_lgbm[g_mask]
                s_sum = np.zeros(g_mask.sum(), dtype=np.float64)
                for b in boosters:
                    s_sum += b.predict(X_g, num_iteration=b.best_iteration).astype(np.float64)
                lgbm_sc[g_mask] = (s_sum / len(boosters)).astype(np.float32)
                del X_g, s_sum
            del X_lgbm
            gc.collect()

            # ── Hierarchical: sigmoid(susp) × sigmoid(rgs) ────────────────────
            sub_pd  = sub_df.to_pandas()
            X_hier  = sub_pd[avail_hier]
            pool    = Pool(X_hier, cat_features=hier_cat_idx)
            susp_raw = susp_model.predict(pool, prediction_type="RawFormulaVal")
            rgs_raw  = rgs_model.predict(pool, prediction_type="RawFormulaVal")
            hier_sc  = (_sigmoid(susp_raw) * _sigmoid(rgs_raw)).astype(np.float32)
            del X_hier, pool, sub_pd, susp_raw, rgs_raw
            gc.collect()

            # ── Recent LightGBM (optional) ────────────────────────────────────
            if recent_booster is not None and w_recent > 0.0:
                avail_recent = [c for c in feature_cols if c in sub_df.columns]
                X_rec = sub_df.select(avail_recent).to_numpy().astype(np.float32)
                recent_sc = recent_booster.predict(
                    X_rec, num_iteration=recent_booster.best_iteration
                ).astype(np.float32)
                del X_rec
                gc.collect()
            else:
                recent_sc = None

            # ── Blend ─────────────────────────────────────────────────────────
            blended = w_lgbm * lgbm_sc + w_hier * hier_sc
            if recent_sc is not None:
                blended = blended + w_recent * recent_sc
            final_scores[cs:ce] = blended.astype(np.float32)

            del sub_df, lgbm_sc, hier_sc, blended
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


# ── Main orchestrator ──────────────────────────────────────────────────────────

def train_v2() -> None:
    """
    Hierarchical ensemble: per-group LGBM + global hierarchical CatBoost
    + optional recent LGBM.  3-way blend weight optimisation on labeled val.
    """
    total_start = time.perf_counter()
    cutoff      = datetime.fromisoformat(VAL_CUTOFF_DATE)
    train_end   = datetime.fromisoformat(TRAIN_END_DATE)
    recent_bdr  = datetime.fromisoformat(RECENT_BORDER)

    _prepare_models_dir(MODELS_DIR)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")
    print(f"Found {len(files)} parquet partitions in {FEATURES_DIR!r}\n", flush=True)
    print(f"LGBM ensemble seeds : {ENSEMBLE_SEEDS}")
    print(f"Yellow weight       : {YELLOW_WEIGHT_MULTIPLIER}")
    print(f"Hierarchical feats  : {len(_ALL_HIER_FEATURES_ORDERED)} "
          f"({len(HIERARCHICAL_NUM_FEATURES)} num + {len(HIERARCHICAL_CAT_FEATURES)} cat)")
    print(f"Train recent LGBM   : {TRAIN_RECENT_LGBM}\n", flush=True)

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

    # ── Step 2: Pre-load labeled val from parquet (for blend optimization) ────
    print("=" * 65)
    print("  Step 2 — Pre-load labeled val from parquet")
    print("=" * 65 + "\n")

    # Need labels_df for parquet loading
    labels_df = load_labels()

    labeled_val_df = _load_labeled_val(files, labels_df, cutoff, train_end, feature_cols)

    # Pre-materialise LGBM feature matrix for fast scoring during training loop
    _avail_feats  = [c for c in feature_cols if c in labeled_val_df.columns]
    X_lbl_full    = labeled_val_df[_avail_feats].to_numpy().astype(np.float32)
    y_lbl         = labeled_val_df["target"].values.astype(np.int8)
    mg_lbl        = labeled_val_df["model_group"].values.astype(np.int8)
    n_lbl         = len(labeled_val_df)

    lgbm_lbl_sum  = np.zeros(n_lbl, dtype=np.float64)

    # ── Step 3: Per-group multi-seed LightGBM ────────────────────────────────
    print("=" * 65)
    print("  Step 3 — Per-group multi-seed LightGBM")
    print("=" * 65 + "\n")

    best_iters_lgbm: dict = {}   # group_id → list[int]

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

        # Extract labeled val subset for early stopping (from memmaps, tiny subset)
        lv_idx          = np.where(labeled_val_mask)[0]
        X_val_lbl_g     = np.array(X_val[lv_idx])
        y_val_lbl_g     = np.array(y_val[lv_idx])
        il_val_ones_g   = np.ones(len(lv_idx), dtype=np.int8)

        group_neg_ratio   = NEG_SAMPLE_RATIO_BY_GROUP.get(group_id, NEG_SAMPLE_RATIO)
        group_lgbm_params = {**LGBM_PARAMS, **LGBM_PARAMS_BY_GROUP.get(group_id, {})}
        group_es_rounds   = EARLY_STOPPING_ROUNDS_BY_GROUP.get(group_id, None)

        lbl_g_mask = (mg_lbl == group_id)   # indices in labeled_val_df for this group
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

            # Feature importance from first seed only
            if seed_idx == 0:
                evaluate(
                    booster, X_val, y_val_np, il_val_np, feature_cols,
                    row_mask=(tg_val_np == group_id),
                    label=f"{group_name} seed-0",
                )

            # Score labeled val rows (27K max) for blend optimization
            X_lbl_g = X_lbl_full[lbl_g_mask]
            s = booster.predict(X_lbl_g, num_iteration=booster.best_iteration).astype(np.float64)
            lgbm_lbl_sum[lbl_g_mask] += s
            del booster, X_lbl_g, s
            gc.collect()

        best_iters_lgbm[group_id] = seed_best_iters
        del X_val_lbl_g, y_val_lbl_g, il_val_ones_g, lv_idx
        gc.collect()

    # Average LGBM scores across seeds
    lgbm_lbl_avg = (lgbm_lbl_sum / len(ENSEMBLE_SEEDS)).astype(np.float32)
    del lgbm_lbl_sum
    gc.collect()

    pr_auc_lgbm = average_precision_score(y_lbl, lgbm_lbl_avg)
    print(f"\n  LGBM only PR-AUC on labeled val : {pr_auc_lgbm:.6f}\n", flush=True)

    # ── Step 4: Load hierarchical train data from parquet ─────────────────────
    print("=" * 65)
    print("  Step 4 — Load hierarchical train data")
    print("=" * 65 + "\n")

    # Train period: VAL_CUTOFF_DATE acts as upper bound (no val leakage)
    # Use a very early lower bound so all is_train==1 rows are included
    hier_train_df = load_hier_split(
        files       = files,
        labels_df   = labels_df,
        cutoff_lo   = datetime(2000, 1, 1),    # all is_train==1 rows start 2024-10-01
        cutoff_hi   = cutoff,                   # VAL_CUTOFF_DATE (exclusive)
        hier_num_features = HIERARCHICAL_NUM_FEATURES,
        hier_cat_features = HIERARCHICAL_CAT_FEATURES,
        green_ratio = SUSPICIOUS_GREEN_RATIO,
        recent_border = recent_bdr,
        verbose     = True,
    )
    print(f"  Hier train data shape : {hier_train_df.shape}", flush=True)

    # ── Step 5: Train Suspicious CatBoost ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Step 5 — Train Suspicious CatBoost  P(labeled | tx)")
    print("=" * 65 + "\n")

    susp_model_val, susp_best_iter, susp_prauc_val = train_suspicious(
        train_df        = hier_train_df,
        val_labeled_df  = labeled_val_df,
        all_features    = _ALL_HIER_FEATURES_ORDERED,
        cat_features    = HIERARCHICAL_CAT_FEATURES,
        params          = SUSPICIOUS_CATBOOST_PARAMS,
        labeled_weight  = SUSPICIOUS_LABELED_WEIGHT,
        green_recent_weight = SUSPICIOUS_GREEN_RECENT_W,
        green_old_weight    = SUSPICIOUS_GREEN_OLD_W,
        recent_border   = recent_bdr,
    )

    # ── Step 6: Train RGS CatBoost ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Step 6 — Train RGS CatBoost  P(fraud | labeled, tx)")
    print("=" * 65 + "\n")

    rgs_model_val, rgs_best_iter, rgs_prauc_val = train_rgs(
        train_df        = hier_train_df,
        val_labeled_df  = labeled_val_df,
        all_features    = _ALL_HIER_FEATURES_ORDERED,
        cat_features    = HIERARCHICAL_CAT_FEATURES,
        params          = RGS_CATBOOST_PARAMS,
        red_weight      = RGS_RED_WEIGHT,
        yellow_weight   = RGS_YELLOW_WEIGHT,
    )

    # Release hier train data — it will be reloaded for full retrain later
    del hier_train_df
    gc.collect()

    # ── Step 7: Compute hierarchical scores on labeled val ───────────────────
    print("=" * 65)
    print("  Step 7 — Score labeled val with hierarchical models")
    print("=" * 65 + "\n")

    hier_lbl_scores = score_hierarchical(
        susp_model_val, rgs_model_val,
        labeled_val_df,
        _ALL_HIER_FEATURES_ORDERED,
        HIERARCHICAL_CAT_FEATURES,
    )
    pr_auc_hier = average_precision_score(y_lbl, hier_lbl_scores)
    print(f"  Hierarchical PR-AUC on labeled val : {pr_auc_hier:.6f}\n", flush=True)

    # ── Step 8: Optional — Train Recent LightGBM ─────────────────────────────
    recent_booster_val  = None
    recent_best_iter    = None
    recent_lbl_scores   = None
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
            lgbm_params       = LGBM_PARAMS_RECENT,
            early_stopping_rounds = EARLY_STOPPING_ROUNDS,
            val_df            = labeled_val_df,
        )
        # Score labeled val
        avail_recent_feats = [c for c in feature_cols if c in labeled_val_df.columns]
        X_rec_val = labeled_val_df[avail_recent_feats].to_numpy().astype(np.float32)
        recent_lbl_scores = recent_booster_val.predict(
            X_rec_val, num_iteration=recent_booster_val.best_iteration
        ).astype(np.float32)
        del X_rec_val
        gc.collect()

        pr_auc_recent = average_precision_score(y_lbl, recent_lbl_scores)
        print(f"  Recent LGBM PR-AUC on labeled val : {pr_auc_recent:.6f}\n", flush=True)
    else:
        print("  Step 8 skipped (TRAIN_RECENT_LGBM = False)\n", flush=True)

    # ── Step 9: Blend weight grid search on labeled val ───────────────────────
    print("=" * 65)
    print("  Step 9 — Blend weight grid search")
    print("=" * 65 + "\n")

    ws = np.arange(0.0, 1.025, 0.05)
    best_prauc_blend = 0.0

    if TRAIN_RECENT_LGBM and recent_lbl_scores is not None:
        # 3D grid: (w_lgbm, w_hier, w_recent), constrained to sum = 1
        best_w_lgbm = best_w_hier = best_w_recent = None
        for w_l in np.arange(0.0, 1.025, 0.1):
            for w_h in np.arange(0.0, 1.025 - w_l, 0.1):
                w_r = 1.0 - w_l - w_h
                if w_r < -0.001:
                    continue
                w_r = max(0.0, w_r)
                blended = (w_l * lgbm_lbl_avg
                           + w_h * hier_lbl_scores
                           + w_r * recent_lbl_scores)
                p = average_precision_score(y_lbl, blended)
                if p > best_prauc_blend:
                    best_prauc_blend = p
                    best_w_lgbm, best_w_hier, best_w_recent = float(w_l), float(w_h), float(w_r)
        if best_w_lgbm is None:
            best_w_lgbm, best_w_hier, best_w_recent = 0.5, 0.3, 0.2
        print(f"  Grid search result  : w_lgbm={best_w_lgbm:.2f}  "
              f"w_hier={best_w_hier:.2f}  w_recent={best_w_recent:.2f}")
    else:
        # 2D grid: (w_lgbm = 1 - w_h, w_hier = w_h)
        best_w_hier = 0.3
        best_w_lgbm = 0.7
        best_w_recent = 0.0
        for w_h in ws:
            w_l = 1.0 - w_h
            blended = w_l * lgbm_lbl_avg + w_h * hier_lbl_scores
            p = average_precision_score(y_lbl, blended)
            if p > best_prauc_blend:
                best_prauc_blend = p
                best_w_lgbm, best_w_hier = float(w_l), float(w_h)
        print(f"  Grid search result  : w_lgbm={best_w_lgbm:.2f}  "
              f"w_hier={best_w_hier:.2f}")

    print(f"  Best blend PR-AUC   : {best_prauc_blend:.6f}")
    print(f"\n  Model PRs on labeled val:")
    print(f"    LGBM alone          : {pr_auc_lgbm:.6f}")
    print(f"    Hierarchical alone  : {pr_auc_hier:.6f}")
    if TRAIN_RECENT_LGBM and recent_lbl_scores is not None:
        print(f"    Recent LGBM alone   : {pr_auc_recent:.6f}")
    print(f"    Best blend          : {best_prauc_blend:.6f}\n", flush=True)

    del lgbm_lbl_avg, hier_lbl_scores, recent_lbl_scores
    del X_lbl_full
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
        group_lgbm_params = {**LGBM_PARAMS, **LGBM_PARAMS_BY_GROUP.get(group_id, {})}

        # Undersample labeled val negatives before appending
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

    # ── 10b. Hierarchical CatBoost (full retrain from parquet) ───────────────
    print(f"\n  ── Hierarchical full retrain: load full data "
          f"({datetime(2000,1,1).date()} – {train_end.date()})")

    hier_full_df = load_hier_split(
        files             = files,
        labels_df         = labels_df,
        cutoff_lo         = datetime(2000, 1, 1),
        cutoff_hi         = train_end,             # include val period too
        hier_num_features = HIERARCHICAL_NUM_FEATURES,
        hier_cat_features = HIERARCHICAL_CAT_FEATURES,
        green_ratio       = SUSPICIOUS_GREEN_RATIO,
        recent_border     = recent_bdr,
        verbose           = True,
    )
    print(f"  Hier full data shape : {hier_full_df.shape}", flush=True)

    # Build weight columns for suspicious model
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

    print("\n  ── Suspicious model full retrain …", flush=True)
    susp_model_full = refit_model(
        model        = susp_model_val,
        train_df     = hier_full_df,
        all_features = _ALL_HIER_FEATURES_ORDERED,
        cat_features = HIERARCHICAL_CAT_FEATURES,
        target_col   = "__susp_target",
        weight_col   = "__susp_weight",
        best_iter    = susp_best_iter,
        retrain_factor = RETRAIN_FULL_ITER_FACTOR,
    )
    susp_save_path = os.path.join(MODELS_DIR, SUSPICIOUS_MODEL_PATH)
    susp_model_full.save_model(susp_save_path)
    print(f"  Suspicious model saved → {susp_save_path}", flush=True)

    # Build weight columns for RGS model (labeled rows only)
    labeled_full = hier_full_df[hier_full_df["raw_target"] != -1].copy()
    w_rgs = np.where(
        labeled_full["raw_target"].values == 1, RGS_RED_WEIGHT, RGS_YELLOW_WEIGHT
    ).astype(np.float32)
    labeled_full["__rgs_weight"] = w_rgs

    print("\n  ── RGS model full retrain …", flush=True)
    rgs_model_full = refit_model(
        model        = rgs_model_val,
        train_df     = labeled_full,
        all_features = _ALL_HIER_FEATURES_ORDERED,
        cat_features = HIERARCHICAL_CAT_FEATURES,
        target_col   = "raw_target",
        weight_col   = "__rgs_weight",
        best_iter    = rgs_best_iter,
        retrain_factor = RETRAIN_FULL_ITER_FACTOR,
    )
    rgs_save_path = os.path.join(MODELS_DIR, RGS_MODEL_PATH)
    rgs_model_full.save_model(rgs_save_path)
    print(f"  RGS model saved → {rgs_save_path}", flush=True)

    del hier_full_df, labeled_full
    gc.collect()

    # ── 10c. Recent LightGBM full retrain ────────────────────────────────────
    recent_booster_full = None
    if TRAIN_RECENT_LGBM and recent_booster_val is not None:
        print("\n  ── Recent LGBM full retrain …", flush=True)
        recent_n_rounds = int(recent_best_iter * RETRAIN_FULL_ITER_FACTOR)
        # Full recent data: recent_border to train_end (includes val period)
        recent_booster_full, _ = _train_recent_lgbm(
            files             = files,
            labels_df         = labels_df,
            recent_border     = recent_bdr,
            cutoff            = train_end,           # extend to train_end for full retrain
            feature_cols      = feature_cols,
            neg_sample_ratio  = RECENT_NEG_SAMPLE_RATIO,
            lgbm_params       = LGBM_PARAMS_RECENT,
            early_stopping_rounds = EARLY_STOPPING_ROUNDS,
            val_df            = None,                # no val for full retrain
            n_rounds_fixed    = recent_n_rounds,     # fixed rounds, no early stopping
        )
        recent_path = os.path.join(MODELS_DIR, RECENT_MODEL_PATH)
        recent_booster_full.save_model(recent_path)
        print(f"  Recent LGBM saved → {recent_path}", flush=True)

    # Clean up val-phase models (replaced by full-retrain versions)
    del susp_model_val, rgs_model_val, recent_booster_val
    gc.collect()

    # ── Step 11: Score test set ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Step 11 — Score test set → submission.csv")
    print("=" * 65 + "\n")

    _score_test_v2(
        lgbm_boosters   = lgbm_boosters_full,
        susp_model      = susp_model_full,
        rgs_model       = rgs_model_full,
        recent_booster  = recent_booster_full,
        feature_cols    = feature_cols,
        hier_all_features = _ALL_HIER_FEATURES_ORDERED,
        hier_cat_features = HIERARCHICAL_CAT_FEATURES,
        submission_path = SUBMISSION_PATH,
        w_lgbm          = best_w_lgbm,
        w_hier          = best_w_hier,
        w_recent        = best_w_recent if TRAIN_RECENT_LGBM else 0.0,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.perf_counter() - total_start
    print(f"\n{'─' * 65}")
    print(f"  Total wall time     : {_fmt(total_time)}")
    print(f"  Blend PR-AUC (val)  : {best_prauc_blend:.6f}")
    print(f"    LGBM              : {pr_auc_lgbm:.6f}  (w={best_w_lgbm:.2f})")
    print(f"    Hierarchical      : {pr_auc_hier:.6f}  (w={best_w_hier:.2f})")
    if TRAIN_RECENT_LGBM and pr_auc_recent > 0:
        print(f"    Recent LGBM       : {pr_auc_recent:.6f}  (w={best_w_recent:.2f})")
    print("─" * 65)


# ── Resume helper ─────────────────────────────────────────────────────────────

def resume_from_step10b(
    susp_best_iter: int = 3000,
    rgs_best_iter: int = 704,
    recent_best_iter: int = 1277,
    w_lgbm: float = 0.10,
    w_hier: float = 0.80,
    w_recent: float = 0.10,
) -> None:
    """
    Resume train_v2() from Step 10b onward.

    Reloads the 20 LGBM models already saved to disk (Step 10a),
    then runs:
      10b — Suspicious + RGS CatBoost full retrain  (from parquet)
      10c — Recent LightGBM full retrain             (from parquet)
      11  — Score test set → submission.csv

    Parameters are the values printed in the original run's log.
    """
    import json
    total_start = time.perf_counter()

    train_end   = datetime.fromisoformat(TRAIN_END_DATE)
    recent_bdr  = datetime.fromisoformat(RECENT_BORDER)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")

    # ── Load feature_cols from memmap metadata ────────────────────────────────
    meta_path = Path(STAGING_DIR) / "memmap_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"memmap_meta.json not found in {STAGING_DIR!r}. "
            "Cannot determine feature_cols — run train_v2() first."
        )
    with open(meta_path) as fh:
        feature_cols = json.load(fh)["feature_cols"]

    print(f"Feature columns   : {len(feature_cols)}")
    print(f"Blend weights     : w_lgbm={w_lgbm:.2f}  w_hier={w_hier:.2f}  w_recent={w_recent:.2f}")
    print(f"Best iters        : susp={susp_best_iter}  rgs={rgs_best_iter}  recent={recent_best_iter}")
    print(f"LGBM models dir   : {MODELS_DIR}\n", flush=True)

    # ── Reload saved LGBM models ──────────────────────────────────────────────
    print("=" * 65)
    print("  Reloading saved LGBM models (Step 10a output)")
    print("=" * 65 + "\n")

    lgbm_boosters_full: dict = {}
    for group_id, group_name in TX_TYPE_GROUPS.items():
        boosters = []
        for seed_idx in range(len(ENSEMBLE_SEEDS)):
            path = os.path.join(
                MODELS_DIR,
                LGBM_MODEL_PATH_FMT.format(name=group_name, seed_idx=seed_idx),
            )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Expected model not found: {path}. "
                    "Cannot resume — Step 10a models are missing."
                )
            booster = lgb.Booster(model_file=path)
            boosters.append(booster)
            print(f"  Loaded {path}")
        lgbm_boosters_full[group_id] = boosters
    print(flush=True)

    # ── Load labels ───────────────────────────────────────────────────────────
    labels_df = load_labels()

    # ── Step 10b: Hierarchical CatBoost full retrain ──────────────────────────
    print("=" * 65)
    print("  Step 10b — Hierarchical CatBoost full retrain")
    print("=" * 65 + "\n")

    print(f"  Loading hierarchical data ({datetime(2000,1,1).date()} – {train_end.date()}) …")
    hier_full_df = load_hier_split(
        files             = files,
        labels_df         = labels_df,
        cutoff_lo         = datetime(2000, 1, 1),
        cutoff_hi         = train_end,
        hier_num_features = HIERARCHICAL_NUM_FEATURES,
        hier_cat_features = HIERARCHICAL_CAT_FEATURES,
        green_ratio       = SUSPICIOUS_GREEN_RATIO,
        recent_border     = recent_bdr,
        verbose           = True,
    )
    print(f"  Hier full data shape : {hier_full_df.shape}\n", flush=True)

    # ── Suspicious model ─────────────────────────────────────────────────────
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

    # Build directly from config params (no need for a val-phase model object)
    susp_params = {k: v for k, v in SUSPICIOUS_CATBOOST_PARAMS.items()
                   if k not in ("verbose",)}
    susp_n_rounds = max(300, int(susp_best_iter * RETRAIN_FULL_ITER_FACTOR))
    susp_params.update({
        "iterations": susp_n_rounds,
        "od_type": "Iter",
        "od_wait": susp_n_rounds + 1,
        "allow_writing_files": False,
    })
    susp_params.pop("use_best_model", None)

    avail_hier = [c for c in _ALL_HIER_FEATURES_ORDERED if c in hier_full_df.columns]
    cat_idx    = [avail_hier.index(c) for c in HIERARCHICAL_CAT_FEATURES if c in avail_hier]

    print(f"  Training Suspicious model ({susp_n_rounds} iterations) …", flush=True)
    t0 = time.perf_counter()
    susp_pool = Pool(
        hier_full_df[avail_hier],
        label=hier_full_df["__susp_target"].values,
        weight=hier_full_df["__susp_weight"].values,
        cat_features=cat_idx,
    )
    from catboost import CatBoostClassifier
    susp_model_full = CatBoostClassifier(**susp_params)
    susp_model_full.fit(susp_pool, verbose=100)
    del susp_pool
    gc.collect()

    susp_save_path = os.path.join(MODELS_DIR, SUSPICIOUS_MODEL_PATH)
    susp_model_full.save_model(susp_save_path)
    print(f"  Suspicious model saved → {susp_save_path}  ({_fmt(time.perf_counter() - t0)})\n",
          flush=True)

    # ── RGS model ─────────────────────────────────────────────────────────────
    labeled_full = hier_full_df[hier_full_df["raw_target"] != -1].copy()
    del hier_full_df
    gc.collect()

    w_rgs = np.where(
        labeled_full["raw_target"].values == 1, RGS_RED_WEIGHT, RGS_YELLOW_WEIGHT
    ).astype(np.float32)

    rgs_params = {k: v for k, v in RGS_CATBOOST_PARAMS.items()
                  if k not in ("verbose",)}
    rgs_n_rounds = max(300, int(rgs_best_iter * RETRAIN_FULL_ITER_FACTOR))
    rgs_params.update({
        "iterations": rgs_n_rounds,
        "od_type": "Iter",
        "od_wait": rgs_n_rounds + 1,
        "allow_writing_files": False,
    })
    rgs_params.pop("use_best_model", None)

    avail_rgs = [c for c in _ALL_HIER_FEATURES_ORDERED if c in labeled_full.columns]
    cat_idx_r = [avail_rgs.index(c) for c in HIERARCHICAL_CAT_FEATURES if c in avail_rgs]

    print(f"  Training RGS model ({rgs_n_rounds} iterations, "
          f"{len(labeled_full):,} labeled rows) …", flush=True)
    t0 = time.perf_counter()
    rgs_pool = Pool(
        labeled_full[avail_rgs],
        label=labeled_full["raw_target"].values.astype(np.int8),
        weight=w_rgs,
        cat_features=cat_idx_r,
    )
    rgs_model_full = CatBoostClassifier(**rgs_params)
    rgs_model_full.fit(rgs_pool, verbose=100)
    del rgs_pool, labeled_full
    gc.collect()

    rgs_save_path = os.path.join(MODELS_DIR, RGS_MODEL_PATH)
    rgs_model_full.save_model(rgs_save_path)
    print(f"  RGS model saved → {rgs_save_path}  ({_fmt(time.perf_counter() - t0)})\n",
          flush=True)

    # ── Step 10c: Recent LightGBM full retrain ────────────────────────────────
    recent_booster_full = None
    if TRAIN_RECENT_LGBM:
        print("=" * 65)
        print("  Step 10c — Recent LightGBM full retrain")
        print("=" * 65 + "\n")

        recent_n_rounds = int(recent_best_iter * RETRAIN_FULL_ITER_FACTOR)
        recent_booster_full, _ = _train_recent_lgbm(
            files             = files,
            labels_df         = labels_df,
            recent_border     = recent_bdr,
            cutoff            = train_end,
            feature_cols      = feature_cols,
            neg_sample_ratio  = RECENT_NEG_SAMPLE_RATIO,
            lgbm_params       = LGBM_PARAMS_RECENT,
            early_stopping_rounds = EARLY_STOPPING_ROUNDS,
            val_df            = None,
            n_rounds_fixed    = recent_n_rounds,
        )
        recent_path = os.path.join(MODELS_DIR, RECENT_MODEL_PATH)
        recent_booster_full.save_model(recent_path)
        print(f"  Recent LGBM saved → {recent_path}\n", flush=True)

    # ── Step 11: Score test set ───────────────────────────────────────────────
    print("=" * 65)
    print("  Step 11 — Score test set → submission.csv")
    print("=" * 65 + "\n")

    _score_test_v2(
        lgbm_boosters   = lgbm_boosters_full,
        susp_model      = susp_model_full,
        rgs_model       = rgs_model_full,
        recent_booster  = recent_booster_full,
        feature_cols    = feature_cols,
        hier_all_features = _ALL_HIER_FEATURES_ORDERED,
        hier_cat_features = HIERARCHICAL_CAT_FEATURES,
        submission_path = SUBMISSION_PATH,
        w_lgbm          = w_lgbm,
        w_hier          = w_hier,
        w_recent        = w_recent if TRAIN_RECENT_LGBM else 0.0,
    )

    total_time = time.perf_counter() - total_start
    print(f"\n{'─' * 65}")
    print(f"  Resume total time   : {_fmt(total_time)}")
    print(f"  Blend weights used  : w_lgbm={w_lgbm:.2f}  w_hier={w_hier:.2f}  w_recent={w_recent:.2f}")
    print("─" * 65)
