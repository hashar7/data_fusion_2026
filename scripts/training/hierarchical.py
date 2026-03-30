"""
Hierarchical CatBoost ensemble for pipeline_v2.

Two complementary models decompose the fraud detection problem:

  Suspicious detector  : P(labeled | tx)
      Target  = 1 for red or yellow rows, 0 for unlabeled green rows.
      Learns which transactions are "interesting" (get reviewed at all).

  Red|Suspicious model : P(fraud | labeled, tx)
      Target  = 1 for red (fraud), 0 for yellow (confirmed safe).
      Trained ONLY on labeled rows — sees a near-50% fraud rate.
      Learns to distinguish fraud from clean among reviewed transactions.

  Final hierarchical score:
      prod_prob = sigmoid(susp_raw) × sigmoid(rgs_raw)

Both models use customer_id as a native CatBoost categorical feature, letting
the model learn per-customer fraud histories without explicit feature engineering.
"""
import gc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

from scripts.training._utils import _fmt, _progress


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_hier_split(
    files: list,
    labels_df: pl.DataFrame,
    cutoff_lo: datetime,
    cutoff_hi: datetime,
    hier_num_features: list,
    hier_cat_features: list,
    green_ratio: float = 0.10,
    rng_seed: int = 42,
    recent_border: datetime | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load labeled rows (all) + sampled green rows from processed parquet files.

    Parameters
    ----------
    files          : sorted list of parquet file paths (output of build_processed_dataset)
    labels_df      : pl.DataFrame with columns [event_id, target] (train_labels.parquet)
    cutoff_lo      : include rows where  cutoff_lo <= event_dttm < cutoff_hi
    cutoff_hi      : upper date bound (exclusive)
    hier_num_features : numerical features to include
    hier_cat_features : categorical features to include (int-encoded; "mcc_code_int"
                        is derived from "mcc_code" String automatically)
    green_ratio    : fraction of unlabeled green rows to keep (hash-based, deterministic)
    rng_seed       : seed used in the Polars hash modulo sampling
    recent_border  : if provided, adds a boolean "is_recent" column (for weight assignment)
    verbose        : print per-file progress

    Returns
    -------
    pd.DataFrame with columns:
        feature columns (hier_num_features + hier_cat_features)
        "event_id", "event_dttm", "model_group", "raw_target", "is_recent" (optional)

    raw_target encoding:  1 = red (fraud),  0 = yellow (confirmed safe),  -1 = green
    """
    labels_event_ids = labels_df["event_id"]
    labels_map = {
        int(r["event_id"]): int(r["target"])
        for r in labels_df.iter_rows(named=True)
    }

    # Columns we need to read from parquet
    # "mcc_code" (String) is read and cast → "mcc_code_int"; exclude from direct read list
    needs_mcc_int = "mcc_code_int" in hier_cat_features
    cat_cols_from_parquet = [c for c in hier_cat_features
                             if c not in ("mcc_code_int",)]
    # Remove customer_id duplication; it is always present in parquet
    read_cols_set = (
        {"event_id", "event_dttm", "is_train", "model_group"}
        | set(hier_num_features)
        | set(cat_cols_from_parquet)
    )
    if needs_mcc_int:
        read_cols_set.add("mcc_code")

    parts: list[pd.DataFrame] = []
    wall_times: list = []
    total_labeled = 0
    total_green   = 0
    green_denom   = max(1, round(1.0 / green_ratio))

    for i, f in enumerate(files):
        t0 = time.perf_counter()

        schema     = pl.scan_parquet(f).collect_schema()
        avail_cols = set(schema.names())
        cols_to_use = list(read_cols_set & avail_cols)

        lf = (
            pl.scan_parquet(f)
            .select(cols_to_use)
        )

        # Parse datetime if stored as string
        if schema.get("event_dttm") == pl.Utf8:
            lf = lf.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )

        # Date range filter
        lf = lf.filter(
            (pl.col("is_train") == 1)
            & (pl.col("event_dttm") >= cutoff_lo)
            & (pl.col("event_dttm") < cutoff_hi)
        )

        # Derive mcc_code_int from mcc_code String
        if needs_mcc_int and "mcc_code" in avail_cols:
            lf = lf.with_columns(
                pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_int")
            )

        # Join labels → raw_target column
        lf = lf.with_columns(pl.lit(None, dtype=pl.Int8).alias("raw_target"))
        lf = (
            lf.join(
                labels_df.lazy().select(["event_id", "target"]),
                on="event_id",
                how="left",
            )
            .with_columns(
                pl.when(pl.col("target").is_not_null())
                  .then(pl.col("target").cast(pl.Int8))
                  .otherwise(pl.lit(-1, dtype=pl.Int8))
                  .alias("raw_target")
            )
            .drop("target")
        )

        # is_labeled flag for sampling
        lf = lf.with_columns(
            (pl.col("raw_target") != -1).cast(pl.Int8).alias("is_labeled")
        )

        # Sample greens deterministically via hash modulo
        lf = lf.filter(
            (pl.col("is_labeled") == 1)
            | ((pl.col("event_id").hash(seed=rng_seed) % green_denom) == 0)
        )

        # Optional: recent flag for weight assignment
        if recent_border is not None:
            lf = lf.with_columns(
                (pl.col("event_dttm") >= recent_border).cast(pl.Int8).alias("is_recent")
            )

        chunk_df = lf.collect()

        n_lab = int((chunk_df["is_labeled"] == 1).sum())
        n_gr  = int((chunk_df["is_labeled"] == 0).sum())
        total_labeled += n_lab
        total_green   += n_gr

        # Fill nulls: cat features → -1, num features → 0
        fill_exprs = []
        for c in hier_cat_features:
            if c in chunk_df.columns:
                fill_exprs.append(pl.col(c).fill_null(-1))
        for c in hier_num_features:
            if c in chunk_df.columns:
                fill_exprs.append(pl.col(c).fill_null(0.0))
        if fill_exprs:
            chunk_df = chunk_df.with_columns(fill_exprs)

        # Drop helper columns not needed downstream
        drop_cols = [c for c in ("is_train", "mcc_code", "is_labeled")
                     if c in chunk_df.columns]
        if drop_cols:
            chunk_df = chunk_df.drop(drop_cols)

        parts.append(chunk_df.to_pandas())

        del chunk_df, lf
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        if verbose:
            _progress(i + 1, len(files), wall_times,
                      suffix=f"labeled={total_labeled:,}  green={total_green:,}")

    if verbose:
        print(f"\n  Hierarchical split loaded: "
              f"{total_labeled:,} labeled + {total_green:,} green  "
              f"({total_labeled + total_green:,} total)", flush=True)

    result = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return result


# ── CatBoost Pool builder ──────────────────────────────────────────────────────

def _build_pool(
    df: pd.DataFrame,
    all_features: list,
    cat_features: list,
    target_col: str,
    weight_col: str | None = None,
) -> Pool:
    """Build a CatBoost Pool from a pandas DataFrame."""
    avail = [c for c in all_features if c in df.columns]
    X = df[avail]
    y = df[target_col].values
    w = df[weight_col].values if weight_col and weight_col in df.columns else None
    cat_idx = [avail.index(c) for c in cat_features if c in avail]
    return Pool(X, label=y, weight=w, cat_features=cat_idx)


# ── Model training ─────────────────────────────────────────────────────────────

def train_suspicious(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    all_features: list,
    cat_features: list,
    params: dict,
    labeled_weight: float = 6.0,
    green_recent_weight: float = 1.5,
    green_old_weight: float = 1.0,
    recent_border: datetime | None = None,
) -> tuple:
    """
    Train the suspicious detector: (red | yellow) vs green.

    Target = 1 for labeled rows (raw_target != -1), 0 for unlabeled green rows.
    val_df must include BOTH labeled and unlabeled rows so the AUC eval metric
    has both positive and negative examples.

    Returns (model, best_iteration, val_auc)
    """
    print(f"\nTraining Suspicious Detector (P(labeled | tx)) …", flush=True)

    # Build train target and weights
    train_df = train_df.copy()
    train_df["__susp_target"] = (train_df["raw_target"] != -1).astype(np.int8)

    # Weight assignment: labeled rows get high weight; green rows get lower
    # weight (recent green slightly higher to reflect possible undetected fraud)
    w = np.where(train_df["__susp_target"] == 1, labeled_weight, green_old_weight).astype(np.float32)
    if recent_border is not None and "is_recent" in train_df.columns:
        is_recent_green = (train_df["__susp_target"] == 0) & (train_df["is_recent"] == 1)
        w[is_recent_green.values] = green_recent_weight
    train_df["__weight"] = w

    avail_feats = [c for c in all_features if c in train_df.columns]
    cat_idx = [avail_feats.index(c) for c in cat_features if c in avail_feats]

    train_pool = Pool(
        train_df[avail_feats],
        label=train_df["__susp_target"].values,
        weight=train_df["__weight"].values,
        cat_features=cat_idx,
    )

    # Val pool: labeled → target=1, unlabeled → target=0
    val_copy = val_df.copy()
    if "_is_labeled" in val_copy.columns:
        val_copy["__susp_target"] = val_copy["_is_labeled"].astype(np.int8)
    elif "raw_target" in val_copy.columns:
        val_copy["__susp_target"] = (val_copy["raw_target"] != -1).astype(np.int8)
    else:
        val_copy["__susp_target"] = 1

    n_val_pos = int(val_copy["__susp_target"].sum())
    n_val_neg = len(val_copy) - n_val_pos
    print(f"  Val rows: {len(val_copy):,}  ({n_val_pos:,} labeled, {n_val_neg:,} unlabeled)",
          flush=True)

    avail_val = [c for c in avail_feats if c in val_copy.columns]
    cat_idx_val = [avail_val.index(c) for c in cat_features if c in avail_val]
    val_pool = Pool(
        val_copy[avail_val],
        label=val_copy["__susp_target"].values,
        cat_features=cat_idx_val,
    )

    # Fit
    _params = {k: v for k, v in params.items() if k not in ("verbose",)}
    verbose = params.get("verbose", 100)
    es = _params.pop("od_wait", 200)
    od_type = _params.pop("od_type", "Iter")
    _params.update({"od_type": od_type, "od_wait": es})

    t0 = time.perf_counter()
    model = CatBoostClassifier(**_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=verbose)

    best_iter = model.get_best_iteration() or _params.get("iterations", 1000)

    # AUC on the full val set (labeled=1 vs unlabeled=0)
    val_raw = model.predict(val_pool, prediction_type="RawFormulaVal")
    val_auc = average_precision_score(val_copy["__susp_target"].values, val_raw)
    print(f"\n  Suspicious | training time : {_fmt(time.perf_counter() - t0)}")
    print(f"  Suspicious | best_iter     : {best_iter}")
    print(f"  Suspicious | val PR-AUC (labeled vs unlabeled): {val_auc:.6f}\n",
          flush=True)

    del train_df, val_copy, train_pool, val_pool
    gc.collect()
    return model, best_iter, val_auc


def train_rgs(
    train_df: pd.DataFrame,
    val_labeled_df: pd.DataFrame,
    all_features: list,
    cat_features: list,
    params: dict,
    red_weight: float = 2.5,
    yellow_weight: float = 1.0,
) -> tuple:
    """
    Train the Red|Suspicious model: red vs yellow on labeled rows only.

    This model sees a near-50% fraud rate (labeled rows only) and directly
    solves the evaluation task (PR-AUC on labeled rows).

    Returns (model, best_iteration, val_pr_auc)
    """
    print(f"\nTraining Red|Suspicious Model (P(fraud | labeled, tx)) …", flush=True)

    # Keep only labeled rows (raw_target != -1)
    labeled_train = train_df[train_df["raw_target"] != -1].copy()
    labeled_val   = val_labeled_df.copy()

    print(f"  RGS train rows: {len(labeled_train):,}  "
          f"({int((labeled_train['raw_target'] == 1).sum()):,} red, "
          f"{int((labeled_train['raw_target'] == 0).sum()):,} yellow)", flush=True)
    print(f"  RGS val rows  : {len(labeled_val):,}  "
          f"({int((labeled_val['raw_target'] == 1).sum()):,} red, "
          f"{int((labeled_val['raw_target'] == 0).sum()):,} yellow)", flush=True)

    avail_feats = [c for c in all_features if c in labeled_train.columns]
    cat_idx = [avail_feats.index(c) for c in cat_features if c in avail_feats]

    w_train = np.where(
        labeled_train["raw_target"].values == 1, red_weight, yellow_weight
    ).astype(np.float32)

    train_pool = Pool(
        labeled_train[avail_feats],
        label=labeled_train["raw_target"].values.astype(np.int8),
        weight=w_train,
        cat_features=cat_idx,
    )

    avail_val = [c for c in avail_feats if c in labeled_val.columns]
    cat_idx_val = [avail_val.index(c) for c in cat_features if c in avail_val]
    val_pool = Pool(
        labeled_val[avail_val],
        label=labeled_val["raw_target"].values.astype(np.int8),
        cat_features=cat_idx_val,
    )

    _params = {k: v for k, v in params.items() if k not in ("verbose",)}
    verbose = params.get("verbose", 100)
    es = _params.pop("od_wait", 300)
    od_type = _params.pop("od_type", "Iter")
    _params.update({"od_type": od_type, "od_wait": es})

    t0 = time.perf_counter()
    model = CatBoostClassifier(**_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=verbose)

    best_iter = model.get_best_iteration() or _params.get("iterations", 1000)
    val_raw   = model.predict(val_pool, prediction_type="RawFormulaVal")
    val_prauc = average_precision_score(
        labeled_val["raw_target"].values.astype(np.int8), val_raw
    )
    print(f"\n  RGS | training time : {_fmt(time.perf_counter() - t0)}")
    print(f"  RGS | best_iter     : {best_iter}")
    print(f"  RGS | val PR-AUC    : {val_prauc:.6f}\n", flush=True)

    del labeled_train, labeled_val, train_pool, val_pool
    gc.collect()
    return model, best_iter, val_prauc


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_hierarchical(
    susp_model: CatBoostClassifier,
    rgs_model: CatBoostClassifier,
    df: pd.DataFrame,
    all_features: list,
    cat_features: list,
) -> np.ndarray:
    """
    Compute the hierarchical product score.

    product_prob = sigmoid(susp_raw) × sigmoid(rgs_raw)

    Both models share the same feature set.  Returns float32 array of length len(df).
    """
    avail = [c for c in all_features if c in df.columns]
    cat_idx = [avail.index(c) for c in cat_features if c in avail]
    pool = Pool(df[avail], cat_features=cat_idx)

    susp_raw = susp_model.predict(pool, prediction_type="RawFormulaVal")
    rgs_raw  = rgs_model.predict(pool, prediction_type="RawFormulaVal")

    prod_prob = (_sigmoid(susp_raw) * _sigmoid(rgs_raw)).astype(np.float32)
    return prod_prob


def refit_model(
    model: CatBoostClassifier,
    train_df: pd.DataFrame,
    all_features: list,
    cat_features: list,
    target_col: str,
    weight_col: str | None,
    best_iter: int,
    retrain_factor: float = 1.05,
    verbose: int = 100,
) -> CatBoostClassifier:
    """
    Retrain a CatBoost model on extended data (train + labeled val) for a fixed
    number of iterations = best_iter × retrain_factor.  No early stopping.
    """
    n_rounds = max(300, int(best_iter * retrain_factor))
    print(f"  Retraining with {n_rounds} iterations (best_iter={best_iter} × {retrain_factor})")

    avail = [c for c in all_features if c in train_df.columns]
    cat_idx = [avail.index(c) for c in cat_features if c in avail]

    # get_params() returns only user-set params (safe for __init__);
    # get_all_params() leaks internal keys like bayesian_matrix_reg.
    params = model.get_params()
    params.update({
        "iterations": n_rounds,
        "od_type": "Iter",
        "od_wait": n_rounds + 1,   # effectively disable early stopping
        "verbose": verbose,
        "allow_writing_files": False,
    })
    # Remove early stopping related keys that conflict
    for k in ("use_best_model",):
        params.pop(k, None)

    w = train_df[weight_col].values if weight_col and weight_col in train_df.columns else None
    pool = Pool(
        train_df[avail],
        label=train_df[target_col].values,
        weight=w,
        cat_features=cat_idx,
    )
    new_model = CatBoostClassifier(**params)
    new_model.fit(pool, verbose=verbose)
    return new_model


# ── Main CatBoost (direct fraud prediction) ──────────────────────────────────

def train_main_catboost(
    train_df: pd.DataFrame,
    val_labeled_df: pd.DataFrame,
    all_features: list,
    cat_features: list,
    params: dict,
    red_weight: float = 10.0,
    yellow_weight: float = 2.5,
    green_recent_weight: float = 1.5,
    green_old_weight: float = 1.0,
    recent_border: datetime | None = None,
) -> tuple:
    """
    Train a direct fraud CatBoost: red (target=1) vs yellow+green (target=0).

    Unlike the hierarchical pair, this is a single model that directly predicts
    P(fraud | tx).  CatBoost's native categorical handling (especially for
    customer_id) provides orthogonal signal to the LGBM ensemble.

    Returns (model, best_iteration, val_pr_auc)
    """
    print(f"\nTraining Main CatBoost (direct fraud) …", flush=True)

    train_df = train_df.copy()
    # target: red=1, yellow+green=0
    train_df["__main_target"] = (train_df["raw_target"] == 1).astype(np.int8)

    # Weights: red > yellow > green (recent slightly higher than old green)
    w = np.full(len(train_df), green_old_weight, dtype=np.float32)
    w[train_df["raw_target"].values == 1] = red_weight
    w[train_df["raw_target"].values == 0] = yellow_weight    # yellow (labeled non-fraud)
    if recent_border is not None and "is_recent" in train_df.columns:
        is_recent_green = (train_df["raw_target"] == -1) & (train_df["is_recent"] == 1)
        w[is_recent_green.values] = green_recent_weight
    train_df["__main_weight"] = w

    n_red    = int((train_df["__main_target"] == 1).sum())
    n_yellow = int((train_df["raw_target"] == 0).sum())
    n_green  = int((train_df["raw_target"] == -1).sum())
    print(f"  Main train : {len(train_df):,} rows  "
          f"({n_red:,} red, {n_yellow:,} yellow, {n_green:,} green)", flush=True)

    avail_feats = [c for c in all_features if c in train_df.columns]
    cat_idx = [avail_feats.index(c) for c in cat_features if c in avail_feats]

    train_pool = Pool(
        train_df[avail_feats],
        label=train_df["__main_target"].values,
        weight=train_df["__main_weight"].values,
        cat_features=cat_idx,
    )

    # Val: labeled rows only, raw_target as binary target
    val_df = val_labeled_df.copy()
    avail_val = [c for c in avail_feats if c in val_df.columns]
    cat_idx_val = [avail_val.index(c) for c in cat_features if c in avail_val]
    val_pool = Pool(
        val_df[avail_val],
        label=val_df["raw_target"].values.astype(np.int8),
        cat_features=cat_idx_val,
    )

    _params = {k: v for k, v in params.items() if k not in ("verbose",)}
    verbose = params.get("verbose", 100)
    es = _params.pop("od_wait", 300)
    od_type = _params.pop("od_type", "Iter")
    _params.update({"od_type": od_type, "od_wait": es})

    t0 = time.perf_counter()
    model = CatBoostClassifier(**_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=verbose)

    best_iter = model.get_best_iteration() or _params.get("iterations", 1000)
    val_raw   = model.predict(val_pool, prediction_type="RawFormulaVal")
    val_prauc = average_precision_score(
        val_df["raw_target"].values.astype(np.int8), val_raw
    )
    print(f"\n  Main CatBoost | training time : {_fmt(time.perf_counter() - t0)}")
    print(f"  Main CatBoost | best_iter     : {best_iter}")
    print(f"  Main CatBoost | val PR-AUC    : {val_prauc:.6f}\n", flush=True)

    del train_df, val_df, train_pool, val_pool
    gc.collect()
    return model, best_iter, val_prauc
