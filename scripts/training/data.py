"""
Data loading, row counting, and memmap construction.
"""
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from scripts.training.config import LABELS_PATH
from scripts.training._utils import _progress, _get_feature_cols

_CACHE_META_FILE = "memmap_meta.json"


def load_labels() -> pl.DataFrame:
    print("Loading labels …", flush=True)
    labels = pl.read_parquet(LABELS_PATH).with_columns(
        pl.col("target").cast(pl.Int8)
    )
    n_pos = int(labels["target"].sum())
    print(f"  {len(labels):,} labelled rows  |  {n_pos:,} positives  "
          f"({n_pos / len(labels):.3%})\n")
    return labels


def count_rows(files: list, cutoff: datetime, train_end: datetime) -> tuple:
    print("Pre-scan: counting train/val rows for memmap allocation …", flush=True)
    n_train = n_val = 0
    wall_times: list = []

    for i, f in enumerate(files):
        t0     = time.perf_counter()
        schema = pl.scan_parquet(f).collect_schema()
        lf = pl.scan_parquet(f).select(["event_dttm", "is_train"])
        if schema["event_dttm"] == pl.Utf8:
            lf = lf.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )
        df = lf.filter(pl.col("is_train") == 1).collect()
        dttm = df["event_dttm"]
        n_train += int((dttm < cutoff).sum())
        n_val   += int(((dttm >= cutoff) & (dttm < train_end)).sum())
        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(files), wall_times,
                  suffix=f"train {n_train:,} | val {n_val:,}")

    print(f"\n  train rows: {n_train:,}  |  val rows: {n_val:,}\n", flush=True)
    return n_train, n_val


def _load_cache(staging_dir: str):
    """
    Try to load previously built memmap files from staging_dir.
    Returns (X_train, y_train, X_val, y_val, il_val, tg_train, tg_val, feature_cols)
    or None on cache miss.
    """
    out           = Path(staging_dir)
    meta_path     = out / _CACHE_META_FILE
    X_train_path  = out / "X_train.npy"
    y_train_path  = out / "y_train.npy"
    X_val_path    = out / "X_val.npy"
    y_val_path    = out / "y_val.npy"
    il_val_path   = out / "is_labeled_val.npy"
    tg_train_path = out / "tx_type_group_train.npy"
    tg_val_path   = out / "tx_type_group_val.npy"

    missing = [p.name for p in [meta_path, X_train_path, y_train_path,
                                 X_val_path, y_val_path, il_val_path,
                                 tg_train_path, tg_val_path]
               if not p.exists()]
    if missing:
        print(f"  Cache miss — missing: {missing}", flush=True)
        return None

    with open(meta_path) as fh:
        meta = json.load(fh)

    n_train       = meta["n_train"]
    n_val         = meta["n_val"]
    n_features    = meta["n_features"]
    feature_cols  = meta["feature_cols"]
    n_labeled_val = meta.get("n_labeled_val", n_val)

    print(f"  Cache hit — reusing memmap files from {staging_dir!r}", flush=True)
    print(f"    X_train       : {n_train:,} × {n_features}")
    print(f"    X_val         : {n_val:,} × {n_features}")
    print(f"    Labeled in val: {n_labeled_val:,} / {n_val:,}")
    print(f"    Features      : {n_features}  "
          f"({feature_cols[0]} … {feature_cols[-1]})\n", flush=True)

    X_train_mm  = np.memmap(str(X_train_path),  dtype="float32", mode="r", shape=(n_train, n_features))
    y_train_mm  = np.memmap(str(y_train_path),  dtype="int8",    mode="r", shape=(n_train,))
    X_val_mm    = np.memmap(str(X_val_path),    dtype="float32", mode="r", shape=(n_val, n_features))
    y_val_mm    = np.memmap(str(y_val_path),    dtype="int8",    mode="r", shape=(n_val,))
    il_val_mm   = np.memmap(str(il_val_path),   dtype="int8",    mode="r", shape=(n_val,))
    tg_train_mm = np.memmap(str(tg_train_path), dtype="int8",    mode="r", shape=(n_train,))
    tg_val_mm   = np.memmap(str(tg_val_path),   dtype="int8",    mode="r", shape=(n_val,))

    return X_train_mm, y_train_mm, X_val_mm, y_val_mm, il_val_mm, tg_train_mm, tg_val_mm, feature_cols


def _save_cache_meta(staging_dir: str, n_train: int, n_val: int,
                     feature_cols: list, n_labeled_val: int = 0) -> None:
    meta = {
        "n_train": n_train, "n_val": n_val,
        "n_features": len(feature_cols), "feature_cols": feature_cols,
        "n_labeled_val": n_labeled_val,
    }
    path = Path(staging_dir) / _CACHE_META_FILE
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Metadata saved → {path}", flush=True)


def build_memmaps(
    files: list,
    labels: pl.DataFrame,
    cutoff: datetime,
    train_end: datetime,
    n_train: int,
    n_val: int,
    staging_dir: str,
) -> tuple:
    """
    Stream parquet chunks into memmap files for the train and val splits.
    Returns (X_train, y_train, X_val, y_val, il_val, tg_train, tg_val, feature_cols).
    """
    out = Path(staging_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Detecting feature count from first chunk …", flush=True)
    first_chunk = pl.read_parquet(files[0])
    if first_chunk["event_dttm"].dtype == pl.Utf8:
        first_chunk = first_chunk.with_columns(
            pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
        )
    first_chunk = first_chunk.join(
        labels.select(["customer_id", "event_id", "target"]),
        on=["customer_id", "event_id"], how="left"
    ).with_columns(pl.col("target").fill_null(0).cast(pl.Int8))
    feature_cols = _get_feature_cols(first_chunk)
    n_features   = len(feature_cols)
    del first_chunk
    gc.collect()
    print(f"  {n_features} feature columns detected.\n", flush=True)

    X_train_path  = str(out / "X_train.npy")
    y_train_path  = str(out / "y_train.npy")
    X_val_path    = str(out / "X_val.npy")
    y_val_path    = str(out / "y_val.npy")
    il_val_path   = str(out / "is_labeled_val.npy")
    tg_train_path = str(out / "tx_type_group_train.npy")
    tg_val_path   = str(out / "tx_type_group_val.npy")

    print("Allocating memmap files …", flush=True)
    X_train_mm  = np.memmap(X_train_path,  dtype="float32", mode="w+", shape=(n_train, n_features))
    y_train_mm  = np.memmap(y_train_path,  dtype="int8",    mode="w+", shape=(n_train,))
    X_val_mm    = np.memmap(X_val_path,    dtype="float32", mode="w+", shape=(n_val, n_features))
    y_val_mm    = np.memmap(y_val_path,    dtype="int8",    mode="w+", shape=(n_val,))
    il_val_mm   = np.memmap(il_val_path,   dtype="int8",    mode="w+", shape=(n_val,))
    tg_train_mm = np.memmap(tg_train_path, dtype="int8",    mode="w+", shape=(n_train,))
    tg_val_mm   = np.memmap(tg_val_path,   dtype="int8",    mode="w+", shape=(n_val,))

    print(f"  X_train.npy : {n_train:,} × {n_features}  "
          f"≈ {n_train * n_features * 4 / 1e9:.1f} GB on disk")
    print(f"  X_val.npy   : {n_val:,} × {n_features}  "
          f"≈ {n_val * n_features * 4 / 1e9:.1f} GB on disk\n")

    labels_event_ids = labels["event_id"]
    print("Writing chunks into memmap files …", flush=True)
    train_cursor = val_cursor = 0
    wall_times:   list = []
    total_pos_train = total_pos_val = 0

    for i, f in enumerate(files):
        t0 = time.perf_counter()

        chunk = pl.read_parquet(f)
        if chunk["event_dttm"].dtype == pl.Utf8:
            chunk = chunk.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )
        chunk = (
            chunk
            .join(labels.select(["customer_id", "event_id", "target"]),
                  on=["customer_id", "event_id"], how="left")
            .with_columns(pl.col("target").fill_null(0).cast(pl.Int8))
        )

        labeled     = chunk.filter(pl.col("is_train") == 1)
        train_chunk = labeled.filter(pl.col("event_dttm") < cutoff)
        val_chunk   = labeled.filter(
            (pl.col("event_dttm") >= cutoff) & (pl.col("event_dttm") < train_end)
        )
        del chunk, labeled
        gc.collect()

        if len(train_chunk) > 0:
            X_np  = train_chunk.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
            y_np  = train_chunk["target"].to_numpy().astype(np.int8)
            tg_np = train_chunk["tx_type_group"].to_numpy().astype(np.int8)
            n     = len(train_chunk)
            X_train_mm[train_cursor : train_cursor + n]  = X_np
            y_train_mm[train_cursor : train_cursor + n]  = y_np
            tg_train_mm[train_cursor : train_cursor + n] = tg_np
            train_cursor    += n
            total_pos_train += int(y_np.sum())
            del X_np, y_np, tg_np
        del train_chunk
        gc.collect()

        if len(val_chunk) > 0:
            X_np  = val_chunk.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
            y_np  = val_chunk["target"].to_numpy().astype(np.int8)
            il_np = val_chunk["event_id"].is_in(labels_event_ids).cast(pl.Int8).to_numpy().astype(np.int8)
            tg_np = val_chunk["tx_type_group"].to_numpy().astype(np.int8)
            n     = len(val_chunk)
            X_val_mm[val_cursor : val_cursor + n]  = X_np
            y_val_mm[val_cursor : val_cursor + n]  = y_np
            il_val_mm[val_cursor : val_cursor + n] = il_np
            tg_val_mm[val_cursor : val_cursor + n] = tg_np
            val_cursor    += n
            total_pos_val += int(y_np.sum())
            del X_np, y_np, il_np, tg_np
        del val_chunk
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(files), wall_times,
                  suffix=f"written train {train_cursor:,} | val {val_cursor:,}")

    X_train_mm.flush();  y_train_mm.flush();  tg_train_mm.flush()
    X_val_mm.flush();    y_val_mm.flush();    il_val_mm.flush();  tg_val_mm.flush()
    n_labeled_val = int(il_val_mm[:val_cursor].sum())

    print(f"\n\n  Train : {train_cursor:,} rows  "
          f"| {total_pos_train:,} positives "
          f"({total_pos_train / max(train_cursor, 1):.4%})")
    print(f"  Val   : {val_cursor:,} rows  "
          f"| {total_pos_val:,} positives "
          f"({total_pos_val / max(val_cursor, 1):.4%})\n")

    if train_cursor == 0:
        raise RuntimeError("Train split is empty.")
    if val_cursor == 0:
        raise RuntimeError("Val split is empty. Move VAL_CUTOFF_DATE earlier.")
    if total_pos_val == 0:
        raise RuntimeError("Val split has zero positives — PR-AUC undefined.")

    _save_cache_meta(staging_dir, train_cursor, val_cursor,
                     feature_cols, n_labeled_val)

    X_train_mm  = np.memmap(X_train_path,  dtype="float32", mode="r", shape=(train_cursor, n_features))
    y_train_mm  = np.memmap(y_train_path,  dtype="int8",    mode="r", shape=(train_cursor,))
    X_val_mm    = np.memmap(X_val_path,    dtype="float32", mode="r", shape=(val_cursor, n_features))
    y_val_mm    = np.memmap(y_val_path,    dtype="int8",    mode="r", shape=(val_cursor,))
    il_val_mm   = np.memmap(il_val_path,   dtype="int8",    mode="r", shape=(val_cursor,))
    tg_train_mm = np.memmap(tg_train_path, dtype="int8",    mode="r", shape=(train_cursor,))
    tg_val_mm   = np.memmap(tg_val_path,   dtype="int8",    mode="r", shape=(val_cursor,))

    return X_train_mm, y_train_mm, X_val_mm, y_val_mm, il_val_mm, tg_train_mm, tg_val_mm, feature_cols
