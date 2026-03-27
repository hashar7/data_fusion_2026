import gc
import time
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from scripts.training.config import (
    RF_BASE_PARAMS,
    RF_PARAM_GRID,
    RF_CV_N_SPLITS,
    RF_CV_SEED,
)
from scripts.training._utils import _progress


def sanitize_rf_features(
    X: np.ndarray,
    feature_cols: list | None = None,
    stage: str = "X",
) -> np.ndarray:
    """
    Prepare feature matrix for sklearn RandomForest:
    - cast to float32
    - replace +inf / -inf with NaN
    - replace NaN with 0.0

    Returns a new float32 array safe for sklearn tree models.
    """
    X = np.asarray(X, dtype=np.float32)

    inf_mask = ~np.isfinite(X)
    n_bad = int(inf_mask.sum())

    if n_bad > 0:
        print(f"  Sanitizing {stage}: found {n_bad:,} non-finite values", flush=True)

        if feature_cols is not None:
            bad_cols = np.flatnonzero(np.any(inf_mask, axis=0))
            if len(bad_cols) > 0:
                preview = [feature_cols[i] for i in bad_cols[:20]]
                print(
                    f"    Columns with non-finite values (first {len(preview)}): {preview}",
                    flush=True,
                )
                if len(bad_cols) > 20:
                    print(f"    ... and {len(bad_cols) - 20} more columns", flush=True)

        X = X.copy()
        X[~np.isfinite(X)] = np.nan

    nan_count = int(np.isnan(X).sum())
    if nan_count > 0:
        print(f"  Filling {nan_count:,} NaN values in {stage} with 0.0", flush=True)
        X = np.nan_to_num(X, nan=-1.0, posinf=-2.0, neginf=-3.0)

    return X


def build_labeled_train_indices(
    files: list[str],
    labels: pl.DataFrame,
    cutoff: datetime,
) -> np.ndarray:
    """
    Reconstruct global row indices of labeled train rows inside X_train memmap.

    This must scan processed parquet partitions in the same order as build_memmaps()
    so the produced indices align exactly with X_train.npy / y_train.npy.

    Returns
    -------
    np.ndarray[int64]
        Global memmap row indices for labeled train rows (F or G only).
    """
    print("Building labeled-train row mapping from processed parquet partitions …", flush=True)

    labels_small = labels.select(["customer_id", "event_id", "target"])
    train_cursor = 0
    idx_parts: list[np.ndarray] = []
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

        labeled_local_mask = train_chunk["target"].is_not_null().to_numpy()
        n_train_chunk = len(train_chunk)

        if labeled_local_mask.any():
            idx_parts.append(train_cursor + np.flatnonzero(labeled_local_mask))

        train_cursor += n_train_chunk

        del chunk, train_chunk, labeled_local_mask
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(
            i + 1,
            len(files),
            wall_times,
            suffix=f"mapped labeled train rows {sum(len(x) for x in idx_parts):,}",
        )

    print()

    if not idx_parts:
        return np.empty(0, dtype=np.int64)

    labeled_train_idx = np.concatenate(idx_parts).astype(np.int64, copy=False)
    print(f"  Labeled train rows mapped: {len(labeled_train_idx):,}\n", flush=True)
    return labeled_train_idx


def tune_rf_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict | None = None,
    n_splits: int = RF_CV_N_SPLITS,
    cv_seed: int = RF_CV_SEED,
) -> tuple[dict, list[dict]]:
    """
    6-fold CV hyperparameter search for RandomForest on labeled train rows only.

    Returns
    -------
    best_params : dict
        Full best parameter dict ready for RandomForestClassifier(**best_params)
    cv_results : list[dict]
        Sorted CV results, best first.
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID

    grid = list(ParameterGrid(param_grid))
    if not grid:
        raise RuntimeError("RF_PARAM_GRID is empty.")

    print(f"RandomForest CV tuning …", flush=True)
    print(f"  Parameter sets : {len(grid)}")
    print(f"  CV folds       : {n_splits}")
    print(f"  Train rows     : {len(y_train):,}")
    print(f"  Positives      : {int(y_train.sum()):,}")
    print(f"  Negatives      : {int((y_train == 0).sum()):,}\n")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=cv_seed,
    )

    results: list[dict] = []

    for param_i, candidate in enumerate(grid, start=1):
        fold_scores: list[float] = []
        print(f"  Param set {param_i}/{len(grid)}: {candidate}", flush=True)

        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
            params = {
                **RF_BASE_PARAMS,
                **candidate,
                "random_state": cv_seed + fold_i,
            }

            model = RandomForestClassifier(**params)
            model.fit(X_train[tr_idx], y_train[tr_idx])

            val_pred = model.predict_proba(X_train[va_idx])[:, 1].astype(np.float32)
            fold_ap = average_precision_score(y_train[va_idx], val_pred)
            fold_scores.append(float(fold_ap))

            print(
                f"    fold {fold_i}/{n_splits}  PR-AUC: {fold_ap:.6f}",
                flush=True,
            )

            del model, val_pred
            gc.collect()

        mean_ap = float(np.mean(fold_scores))
        std_ap = float(np.std(fold_scores))

        print(
            f"    mean PR-AUC: {mean_ap:.6f}  |  std: {std_ap:.6f}\n",
            flush=True,
        )

        results.append(
            {
                "candidate_params": candidate,
                "full_params": {
                    **RF_BASE_PARAMS,
                    **candidate,
                    "random_state": cv_seed,
                },
                "fold_scores": fold_scores,
                "mean_pr_auc": mean_ap,
                "std_pr_auc": std_ap,
            }
        )

    results.sort(key=lambda x: x["mean_pr_auc"], reverse=True)

    print("Top CV results:", flush=True)
    for rank, row in enumerate(results[:10], start=1):
        print(
            f"  {rank:>2}. PR-AUC={row['mean_pr_auc']:.6f} ± {row['std_pr_auc']:.6f}  "
            f"{row['candidate_params']}",
            flush=True,
        )
    print()

    best_params = results[0]["full_params"]
    print(f"Best RF params: {best_params}\n", flush=True)

    return best_params, results


def train_rf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: dict,
) -> RandomForestClassifier:
    """
    Fit one RandomForestClassifier with supplied params.
    """
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    return model