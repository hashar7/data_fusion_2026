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
    RF_CV_SEED,
)
from scripts.training._utils import _progress


def sanitize_rf_features(
    X: np.ndarray,
    feature_cols: list | None = None,
    stage: str = "X",
) -> np.ndarray:
    """
    Prepare feature matrix for sklearn RandomForest.
    """
    X = np.asarray(X, dtype=np.float32)
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


def build_labeled_fg_train_mapping(
    files: list[str],
    labels: pl.DataFrame,
    cutoff: datetime,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct global row indices of labeled train rows inside X_train memmap,
    together with their original labels from train_labels.parquet.

    Returns
    -------
    train_idx : np.ndarray[int64]
        Global memmap row indices for labeled train rows (F or G only).
    orig_target : np.ndarray[int8]
        Original labels from train_labels.parquet:
            1 = F
            0 = G
    """
    print("Building labeled F/G train mapping from processed parquet partitions …", flush=True)

    labels_small = labels.select(["customer_id", "event_id", "target"])
    train_cursor = 0
    idx_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
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

        labeled_mask = train_chunk["target"].is_not_null().to_numpy()
        n_train_chunk = len(train_chunk)

        if labeled_mask.any():
            idx_parts.append(train_cursor + np.flatnonzero(labeled_mask))
            y_parts.append(
                train_chunk["target"].filter(pl.Series(labeled_mask)).to_numpy().astype(np.int8)
            )

        train_cursor += n_train_chunk

        del chunk, train_chunk, labeled_mask
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
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)

    train_idx = np.concatenate(idx_parts).astype(np.int64, copy=False)
    orig_target = np.concatenate(y_parts).astype(np.int8, copy=False)

    print(f"  Labeled train rows mapped: {len(train_idx):,}\n", flush=True)
    return train_idx, orig_target


def train_rf_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: dict | None = None,
) -> RandomForestClassifier:
    """
    Train one RandomForestClassifier with fixed params.
    """
    params = RF_BASE_PARAMS.copy() if rf_params is None else rf_params.copy()
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

