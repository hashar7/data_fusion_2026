"""
train_baseline.py
─────────────────
Memory-efficient baseline LightGBM fraud-detection model.

All four data periods (pretrain, train, pretest, test) live in one directory
and are distinguished by date + is_train column:

  Period    Dates                    is_train  Role
  ────────  ───────────────────────  ────────  ──────────────────────────────
  pretrain  2023-10-01 – 2024-09-30  0         history for feature engineering
  train     2024-10-01 – 2025-05-31  1         labeled; used for train + val
  pretest   2025-06-01 – 2025-08-09  0         history for test feature eng.
  test      2025-06-01 – 2025-08-09  1         scored; submission output

Row routing
───────────
  event_dttm < VAL_CUTOFF_DATE  AND  is_train == 1  →  train memmap
  VAL_CUTOFF_DATE ≤ event_dttm ≤ TRAIN_END_DATE  AND  is_train == 1  →  val memmap
  event_dttm < VAL_CUTOFF_DATE  AND  is_train == 0  →  discarded (pretrain history)
  event_dttm > TRAIN_END_DATE   (pretest / test)    →  scored in score_test()

Why pretrain/pretest rows are loaded but not written to memmaps
───────────────────────────────────────────────────────────────
Rolling and cumulative features were computed per-customer during feature
engineering using the full history.  By the time we reach this script, those
features are already materialised in the parquet files — we simply need to
filter to the right rows.  Pretrain and pretest rows contain no labels and
must not enter the train/val splits; they appear in the parquets only because
the feature pipeline needed them for context.

Memory strategy
───────────────
The full matrix (~60 GB) cannot fit in RAM.  Rows are streamed chunk by chunk
into numpy memory-mapped files on disk.  Only one chunk is in RAM at a time.
LightGBM reads the memmaps lazily, keeping peak RAM to ~2-4 GB.

Label semantics
───────────────
  target == 1   unconfirmed (fraud)         ← positive class
  target == 0   client-confirmed (legit)    ← negative class
  open-loop     no label row               ← treated as 0 in train AND val
"""

import gc
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LABELS_PATH     = "../../data/train_labels.parquet"
FEATURES_DIR    = "../data_processed"   # ALL periods in one directory
STAGING_DIR     = "../data_splits"
MODEL_OUT_PATH  = "baseline_lgbm.txt"
SUBMISSION_PATH = "submission.csv"

# Train period: 2024-10-01 → 2025-05-31  (is_train == 1)
# Val   window: VAL_CUTOFF_DATE → TRAIN_END_DATE  (is_train == 1)
# Test  period: 2025-06-01 → 2025-08-09  (is_train == 1)
#
# TRAIN_END_DATE is the exclusive upper boundary of the train+val window.
# Any row with event_dttm > TRAIN_END_DATE belongs to pretest or test and
# is never written to the train/val memmaps.
VAL_CUTOFF_DATE = "2025-04-01"
TRAIN_END_DATE  = "2025-06-01"   # first date of the test period (exclusive)

# ── Negative undersampling ────────────────────────────────────────────────────
# Fraction of TRAINING negatives (label=0) to keep.
# All positives are always kept.
#
# Why undersample instead of relying solely on is_unbalance=True:
#   is_unbalance reweights gradients but still processes all 66M rows every
#   tree iteration — making each round slow and gradient updates noisy.
#   Undersampling physically reduces the dataset so each round is faster,
#   allowing more iterations in the same wall-clock time.
#
# Why disable is_unbalance when undersampling:
#   After undersampling to ratio R, the effective class ratio is already
#   (1-R)*n_neg : n_pos.  Applying is_unbalance on top double-corrects and
#   over-weights the positive class, causing the model to be over-aggressive.
#
# Recommended values and trade-offs:
#   None  → no undersampling; use full 66M rows (original behavior, slowest)
#   0.05  → keep 5% of negatives  → ~1:20  pos:neg ratio, ~5M train rows
#   0.02  → keep 2% of negatives  → ~1:8   pos:neg ratio, ~2M train rows (fast)
#   0.01  → keep 1% of negatives  → ~1:4   ratio, may underfit on neg patterns
#
# Starting point: 0.05 (1:20 ratio).  If val PR-AUC is higher than with None
# but train PR-AUC is very high, try increasing back toward 0.10.
# After first run with 0.05 (PR-AUC 0.725, converged at round 1046 in 4m),
# bumping to 0.10 exposes the model to more negative diversity — likely to
# improve recall on rare fraud patterns without much speed penalty.
NEG_SAMPLE_RATIO = 0.05   # set to None to disable undersampling
UNDERSAMPLE_SEED = 42

NON_FEATURE_COLS = {
    "customer_id", "event_id", "event_dttm", "target", "is_train",
    "mcc_code", "accept_language", "browser_language",
    "battery", "device_system_version", "screen_size",
    "developer_tools", "compromised",
}

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",
    "verbosity":         -1,
    "device_type":       "cpu",
    "num_threads":       max(1, (os.cpu_count() or 4) - 1),
    # is_unbalance is set dynamically in train_model depending on whether
    # NEG_SAMPLE_RATIO is active (see train_model docstring).
    # ── Tree structure ────────────────────────────────────────────────────────
    # Increased num_leaves for more expressive trees on 65M rows.
    # min_child_samples raised proportionally to prevent overfitting.
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 200,
    # ── Learning rate & iterations ────────────────────────────────────────────
    # lr=0.05 with n_estimators=3000 gives roughly the same total capacity as
    # lr=0.003 with 1000 rounds but trains ~3x faster and converges properly.
    "learning_rate":     0.05,
    "n_estimators":      3000,      # model did not converge at 1000 — extend
    # ── Memory / speed ────────────────────────────────────────────────────────
    "max_bin":           255,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "seed":              42,
}

EARLY_STOPPING_ROUNDS = 100   # give the model more patience before stopping
LOG_EVAL_PERIOD       = 50


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 35) -> str:
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _progress(done: int, total: int, wall_times: list, suffix: str = "") -> None:
    pct    = 100.0 * done / max(total, 1)
    recent = wall_times[-5:]
    eta    = _fmt((sum(recent) / len(recent)) * (total - done)) if recent else "?"
    print(
        f"\r  {_bar(done, total)} {pct:5.1f}%  "
        f"{done}/{total}  ETA {eta}  {suffix}          ",
        end="", flush=True,
    )


def _parquet_files(directory: str) -> list:
    return sorted(Path(directory).glob("*.parquet"))


def _get_feature_cols(df: pl.DataFrame) -> list:
    numeric = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and df[c].dtype in numeric]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load labels (87k rows, always fits in RAM)
# ─────────────────────────────────────────────────────────────────────────────

def load_labels() -> pl.DataFrame:
    print("Loading labels …", flush=True)
    labels = pl.read_parquet(LABELS_PATH).with_columns(
        pl.col("target").cast(pl.Int8)
    )
    n_pos = int(labels["target"].sum())
    print(f"  {len(labels):,} labelled rows  |  {n_pos:,} positives  "
          f"({n_pos / len(labels):.3%})\n")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Count rows needed for memmap pre-allocation
# ─────────────────────────────────────────────────────────────────────────────

def count_rows(files: list, cutoff: datetime, train_end: datetime) -> tuple:
    """
    Lazy-scan all files to count train/val rows without loading features.
    Only event_dttm + is_train are read, so peak RAM is negligible.

    Counted rows (is_train == 1 only):
      train : event_dttm <  cutoff
      val   : cutoff <= event_dttm < train_end

    Pretest/test rows (event_dttm >= train_end) are not counted here —
    they are handled by score_test() at inference time.

    Returns (n_train, n_val).
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build memmap files (skipped on subsequent runs if cache exists)
# ─────────────────────────────────────────────────────────────────────────────

# Name of the JSON sidecar that stores shapes + feature column names.
# np.memmap requires the exact shape at open time, so we persist it here.
# Delete this file (along with the .npy files) to trigger a full rebuild.
_CACHE_META_FILE = "memmap_meta.json"
# Five memmap files are written: X_train, y_train, X_val, y_val, is_labeled_val.
# is_labeled_val marks which val rows have real ground truth (1) vs open-loop (0).
# PR-AUC is computed only on rows where is_labeled_val == 1.


def _load_cache(staging_dir: str):
    """
    Try to load previously built memmap files from staging_dir.

    Returns (X_train_mm, y_train_mm, X_val_mm, y_val_mm, feature_cols)
    if all four .npy files AND memmap_meta.json exist, otherwise None.

    The sidecar stores: n_train, n_val, n_features, feature_cols.
    This is the minimum needed to reopen the memmaps without re-scanning data.

    Note: the sidecar is written only AFTER a successful flush, so a partial
    write caused by a mid-run crash will always leave the cache invalid
    (sidecar absent) and force a clean rebuild on the next run.
    """
    import json

    out          = Path(staging_dir)
    meta_path    = out / _CACHE_META_FILE
    X_train_path = out / "X_train.npy"
    y_train_path = out / "y_train.npy"
    X_val_path   = out / "X_val.npy"
    y_val_path   = out / "y_val.npy"

    il_val_path  = out / "is_labeled_val.npy"
    missing = [p.name for p in [meta_path, X_train_path, y_train_path,
                                 X_val_path, y_val_path, il_val_path]
               if not p.exists()]
    if missing:
        print(f"  Cache miss — missing: {missing}", flush=True)
        return None

    with open(meta_path) as fh:
        meta = json.load(fh)

    n_train      = meta["n_train"]
    n_val        = meta["n_val"]
    n_features   = meta["n_features"]
    feature_cols = meta["feature_cols"]
    n_labeled_val = meta.get("n_labeled_val", n_val)  # back-compat

    print(f"  Cache hit — reusing memmap files from {staging_dir!r}", flush=True)
    print(f"    X_train       : {n_train:,} × {n_features}")
    print(f"    X_val         : {n_val:,} × {n_features}")
    print(f"    Labeled in val: {n_labeled_val:,} / {n_val:,}")
    print(f"    Features      : {n_features}  "
          f"({feature_cols[0]} … {feature_cols[-1]})\n", flush=True)

    X_train_mm    = np.memmap(str(X_train_path), dtype="float32", mode="r",
                              shape=(n_train, n_features))
    y_train_mm    = np.memmap(str(y_train_path), dtype="int8",    mode="r",
                              shape=(n_train,))
    X_val_mm      = np.memmap(str(X_val_path),   dtype="float32", mode="r",
                              shape=(n_val,   n_features))
    y_val_mm      = np.memmap(str(y_val_path),   dtype="int8",    mode="r",
                              shape=(n_val,))
    il_val_mm     = np.memmap(str(il_val_path),  dtype="int8",    mode="r",
                              shape=(n_val,))

    return X_train_mm, y_train_mm, X_val_mm, y_val_mm, il_val_mm, feature_cols


def _save_cache_meta(staging_dir: str, n_train: int, n_val: int,
                     feature_cols: list, n_labeled_val: int = 0) -> None:
    """Write JSON sidecar so future runs can reopen memmaps without rebuilding."""
    import json
    meta = {"n_train": n_train, "n_val": n_val,
            "n_features": len(feature_cols), "feature_cols": feature_cols,
            "n_labeled_val": n_labeled_val}
    path = Path(staging_dir) / _CACHE_META_FILE
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Metadata saved → {path}", flush=True)


def build_memmaps(
    files: list,
    labels,
    cutoff: datetime,
    train_end: datetime,
    n_train: int,
    n_val: int,
    staging_dir: str,
) -> tuple:
    """
    Stream parquet chunks into memmap files for the train and val splits.

    Row routing (applied per chunk):
      is_train == 1  AND  event_dttm <  cutoff                → train memmap
      is_train == 1  AND  cutoff <= event_dttm < train_end    → val   memmap
      is_train == 0  (pretrain / pretest)                     → skipped
      event_dttm >= train_end (test period)                   → skipped

    Only one parquet chunk is in RAM at a time.
    Returns (X_train_mm, y_train_mm, X_val_mm, y_val_mm, il_val_mm, feature_cols).
    """
    # ── Build from scratch ────────────────────────────────────────────────────
    out = Path(staging_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Infer feature columns from the first chunk (need n_features for shape)
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

    # ── Allocate memmap files ─────────────────────────────────────────────────
    X_train_path = str(out / "X_train.npy")
    y_train_path = str(out / "y_train.npy")
    X_val_path   = str(out / "X_val.npy")
    y_val_path   = str(out / "y_val.npy")

    il_val_path  = str(out / "is_labeled_val.npy")

    print("Allocating memmap files …", flush=True)
    X_train_mm = np.memmap(X_train_path, dtype="float32", mode="w+",
                           shape=(n_train, n_features))
    y_train_mm = np.memmap(y_train_path, dtype="int8",    mode="w+",
                           shape=(n_train,))
    X_val_mm   = np.memmap(X_val_path,   dtype="float32", mode="w+",
                           shape=(n_val,   n_features))
    y_val_mm   = np.memmap(y_val_path,   dtype="int8",    mode="w+",
                           shape=(n_val,))
    # is_labeled_val: 1 if this val row has a real label, 0 if open-loop.
    # Only rows where this is 1 are used for PR-AUC computation.
    il_val_mm  = np.memmap(il_val_path,  dtype="int8",    mode="w+",
                           shape=(n_val,))

    print(f"  X_train.npy : {n_train:,} × {n_features}  "
          f"≈ {n_train * n_features * 4 / 1e9:.1f} GB on disk")
    print(f"  X_val.npy   : {n_val:,} × {n_features}  "
          f"≈ {n_val * n_features * 4 / 1e9:.1f} GB on disk\n")

    # ── Stream chunks into memmaps ────────────────────────────────────────────
    # Precompute the set of labeled event_ids for fast is_in() lookup.
    # This is the same labels DataFrame passed in, but we only need event_id.
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

        # Route rows: only is_train==1 rows enter the memmaps.
        # Pretrain/pretest (is_train==0) and test-period rows are skipped.
        labeled = chunk.filter(pl.col("is_train") == 1)
        train_chunk = labeled.filter(pl.col("event_dttm") <  cutoff)
        val_chunk   = labeled.filter(
            (pl.col("event_dttm") >= cutoff) & (pl.col("event_dttm") < train_end)
        )
        del chunk, labeled
        gc.collect()

        if len(train_chunk) > 0:
            X_np = train_chunk.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
            y_np = train_chunk["target"].to_numpy().astype(np.int8)
            n    = len(train_chunk)
            X_train_mm[train_cursor : train_cursor + n] = X_np
            y_train_mm[train_cursor : train_cursor + n] = y_np
            train_cursor    += n
            total_pos_train += int(y_np.sum())
            del X_np, y_np
        del train_chunk
        gc.collect()

        if len(val_chunk) > 0:
            X_np  = val_chunk.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)
            y_np  = val_chunk["target"].to_numpy().astype(np.int8)
            # is_labeled: 1 where the row was found in the labels file.
            # The label join added a temporary column via the left-join; we
            # detect a real label by checking if the original join produced a
            # non-null value. Since we already filled nulls to 0, we must
            # re-join just the event_ids to recover the mask — but it is
            # simpler to pass the information through from the join before
            # filling nulls.  We use a workaround: a row is labeled if its
            # event_id appears in the labels index we pass in from outside.
            # For the build path, labels_index is set just before the loop.
            il_np = val_chunk["event_id"].is_in(labels_event_ids).cast(pl.Int8).to_numpy().astype(np.int8)
            n     = len(val_chunk)
            X_val_mm[val_cursor : val_cursor + n] = X_np
            y_val_mm[val_cursor : val_cursor + n] = y_np
            il_val_mm[val_cursor : val_cursor + n] = il_np
            val_cursor    += n
            total_pos_val += int(y_np.sum())
            del X_np, y_np, il_np
        del val_chunk
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(files), wall_times,
                  suffix=f"written train {train_cursor:,} | val {val_cursor:,}")

    X_train_mm.flush()
    y_train_mm.flush()
    X_val_mm.flush()
    y_val_mm.flush()
    il_val_mm.flush()
    n_labeled_val = int(il_val_mm[:val_cursor].sum())

    print(f"\n\n  Train : {train_cursor:,} rows  "
          f"| {total_pos_train:,} positives "
          f"({total_pos_train / max(train_cursor, 1):.4%})")
    print(f"  Val   : {val_cursor:,} rows  "
          f"| {total_pos_val:,} positives "
          f"({total_pos_val / max(val_cursor, 1):.4%})\n")

    if train_cursor == 0:
        raise RuntimeError(
            "Train split is empty. Check that event_dttm values are before "
            f"VAL_CUTOFF_DATE={VAL_CUTOFF_DATE}."
        )
    if val_cursor == 0:
        raise RuntimeError("Val split is empty. Move VAL_CUTOFF_DATE earlier.")
    if total_pos_val == 0:
        raise RuntimeError(
            "Val split has zero positives — PR-AUC undefined. "
            "Move VAL_CUTOFF_DATE earlier."
        )

    # Write sidecar AFTER successful flush — a crash before this point leaves
    # no sidecar, so the next run correctly detects a cache miss and rebuilds.
    _save_cache_meta(staging_dir, train_cursor, val_cursor,
                     feature_cols, n_labeled_val)

    # Reopen as read-only to prevent accidental writes during training
    X_train_mm = np.memmap(X_train_path, dtype="float32", mode="r",
                           shape=(train_cursor, n_features))
    y_train_mm = np.memmap(y_train_path, dtype="int8",    mode="r",
                           shape=(train_cursor,))
    X_val_mm   = np.memmap(X_val_path,   dtype="float32", mode="r",
                           shape=(val_cursor,   n_features))
    y_val_mm   = np.memmap(y_val_path,   dtype="int8",    mode="r",
                           shape=(val_cursor,))
    il_val_mm  = np.memmap(il_val_path,  dtype="int8",    mode="r",
                           shape=(val_cursor,))

    return X_train_mm, y_train_mm, X_val_mm, y_val_mm, il_val_mm, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    il_val: np.ndarray,
    feature_cols: list,
) -> lgb.Booster:
    """
    Build LightGBM Datasets from memmap arrays and train.

    X_train / y_train : full train split (labeled + open-loop with label=0)
    X_val / y_val     : full val split   (labeled + open-loop with label=0)
    il_val            : int8 mask, 1 = this val row has a real label

    The validation Dataset passed to LightGBM for early stopping uses ONLY
    the labeled val rows (il_val == 1).  This ensures the average_precision
    metric LightGBM tracks during training is computed on real ground truth,
    not diluted by 27M open-loop rows with synthetic label=0.  Without this
    fix the early-stopping signal is misleadingly pessimistic because the
    model is penalised for correctly assigning high scores to open-loop rows
    that happen to be labelled 0 — when in reality we have no idea whether
    those transactions are fraud or not.
    """
    # ── Negative undersampling (training set only) ───────────────────────────
    # When NEG_SAMPLE_RATIO is set, we keep all positives and a random
    # fraction of negatives.  The result is passed to lgb.Dataset as an
    # index array — no data is copied, LightGBM reads only the selected rows
    # from the memmap.  Val is never undersampled.
    rng = np.random.default_rng(UNDERSAMPLE_SEED)

    y_train_arr = np.array(y_train)   # cheap int8 vector, load fully
    pos_idx = np.where(y_train_arr == 1)[0]
    neg_idx = np.where(y_train_arr == 0)[0]

    if NEG_SAMPLE_RATIO is not None:
        n_neg_keep = max(int(len(neg_idx) * NEG_SAMPLE_RATIO), len(pos_idx))
        neg_idx_sampled = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        neg_idx_sampled.sort()   # sorted order → sequential memmap reads (faster)
        train_idx = np.concatenate([pos_idx, neg_idx_sampled])
        train_idx.sort()

        n_pos  = len(pos_idx)
        n_neg  = len(neg_idx_sampled)
        ratio  = n_neg / max(n_pos, 1)
        print(f"  Undersampling: kept {n_pos:,} pos + {n_neg:,} neg "
              f"(1:{ratio:.0f} ratio, {len(train_idx):,} / {len(y_train_arr):,} total rows)",
              flush=True)

        # When undersampling, the class ratio is already corrected manually.
        # Setting is_unbalance=True on top would double-penalise the negative
        # class and make the model over-aggressive — so we turn it off.
        params = {**LGBM_PARAMS, "is_unbalance": False}

        # Pull the selected rows into RAM (subset of memmap, not full matrix)
        X_train_used = np.array(X_train[train_idx])
        y_train_used = y_train_arr[train_idx]
    else:
        print(f"  No undersampling — using full {len(y_train_arr):,} train rows",
              flush=True)
        # Without undersampling, let LightGBM handle imbalance via reweighting
        params = {**LGBM_PARAMS, "is_unbalance": True}
        X_train_used = X_train   # memmap, free_raw_data=False below
        y_train_used = y_train_arr

    del y_train_arr, pos_idx, neg_idx
    gc.collect()

    print("Training LightGBM …", flush=True)
    for k, v in params.items():
        print(f"  {k:25s}: {v}")
    print(flush=True)

    # ── Val: labeled rows only for the early-stopping signal ─────────────────
    # PR-AUC early stopping is computed only on rows with real ground truth.
    # Including open-loop rows (synthetic label=0) in the val Dataset would
    # dilute the metric and make the stopping signal misleadingly pessimistic.
    labeled_mask  = np.array(il_val) == 1
    X_val_labeled = np.array(X_val[labeled_mask])
    y_val_labeled = np.array(y_val[labeled_mask])
    print(f"  Labeled val rows for early stopping: {labeled_mask.sum():,} / {len(il_val):,}\n",
          flush=True)
    del labeled_mask
    gc.collect()

    free_train = NEG_SAMPLE_RATIO is not None  # only free if we made a copy
    dtrain = lgb.Dataset(
        X_train_used, label=y_train_used,
        feature_name=feature_cols,
        free_raw_data=free_train,
    )
    dval = lgb.Dataset(
        X_val_labeled, label=y_val_labeled,
        feature_name=feature_cols,
        reference=dtrain,
        free_raw_data=True,
    )
    del X_val_labeled, y_val_labeled
    if free_train:
        del X_train_used, y_train_used
    gc.collect()

    t0 = time.perf_counter()
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=LGBM_PARAMS["n_estimators"],
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=LOG_EVAL_PERIOD),
        ],
    )

    print(f"\n  Training time   : {_fmt(time.perf_counter() - t0)}")
    print(f"  Best iteration  : {booster.best_iteration}")
    print(f"  Best val PR-AUC : "
          f"{booster.best_score['val']['average_precision']:.6f}\n")
    return booster


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    booster: lgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    il_val: np.ndarray,
    feature_cols: list,
    chunk_size: int = 500_000,
) -> float:
    """
    Score the val memmap in chunks so that only `chunk_size` rows of float32
    are in RAM at once for prediction.  Accumulate scores and labels
    (both cheap: float32 + int8 vectors) for final metric computation.

    PR-AUC is computed only on rows where il_val == 1 (real ground truth).
    Open-loop rows are scored by the model (they inform score distribution)
    but excluded from the metric calculation.
    """
    print("Evaluating on validation set …", flush=True)
    n_val       = len(y_val)
    n_chunks    = (n_val + chunk_size - 1) // chunk_size
    all_scores: list = []
    wall_times: list = []

    for i in range(n_chunks):
        t0    = time.perf_counter()
        start = i * chunk_size
        end   = min(start + chunk_size, n_val)

        # Slicing a memmap triggers a disk read for only those rows
        X_chunk = np.array(X_val[start:end])   # copy to RAM for predict
        scores  = booster.predict(
            X_chunk, num_iteration=booster.best_iteration
        ).astype(np.float32)
        all_scores.append(scores)
        del X_chunk
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, n_chunks, wall_times)

    print(flush=True)

    scores_all = np.concatenate(all_scores)
    labels_all = np.array(y_val)   # int8 vector, cheap to load fully
    il_all     = np.array(il_val)  # int8 is_labeled mask
    del all_scores
    gc.collect()

    # ── Filter to labeled rows for metric computation ─────────────────────────
    labeled_mask   = il_all == 1
    n_labeled      = labeled_mask.sum()
    n_total        = len(labels_all)
    scores_labeled = scores_all[labeled_mask]
    labels_labeled = labels_all[labeled_mask]
    print(f"  Scoring {n_labeled:,} labeled rows out of {n_total:,} val rows "
          f"({n_labeled/n_total:.2%} labeled)", flush=True)

    # ── Competition metric (labeled rows only) ────────────────────────────────
    pr_auc = average_precision_score(labels_labeled, scores_labeled)
    print(f"\n  PR-AUC (average_precision_score, labeled only) : {pr_auc:.6f}")
    # Also report on all rows for reference (matches LightGBM's internal metric)
    pr_auc_all = average_precision_score(labels_all, scores_all)
    print(f"  PR-AUC (all val rows incl. open-loop)           : {pr_auc_all:.6f}")

    # ── Max-F1 threshold (on labeled rows) ───────────────────────────────────
    prec_arr, rec_arr, thresholds = precision_recall_curve(labels_labeled, scores_labeled)
    f1_arr = np.where(
        (prec_arr + rec_arr) == 0, 0.0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9),
    )
    best_idx = int(np.argmax(f1_arr))
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    print(f"\n  Max-F1 operating point (threshold = {best_thr:.4f})")
    print(f"    Precision : {prec_arr[best_idx]:.4f}")
    print(f"    Recall    : {rec_arr[best_idx]:.4f}")
    print(f"    F1        : {f1_arr[best_idx]:.4f}")

    y_pred_labeled = (scores_labeled >= best_thr).astype(np.int8)
    print("\n  Classification report at max-F1 threshold (labeled rows only):")
    print(classification_report(
        labels_labeled, y_pred_labeled,
        target_names=["confirmed (0)", "unconfirmed (1)"],
        digits=4,
    ))

    print("  Score distribution (labeled val rows):")
    for pct, lbl in [(0, "min"), (25, "p25"), (50, "median"),
                     (75, "p75"), (90, "p90"), (95, "p95"),
                     (99, "p99"), (100, "max")]:
        print(f"    {lbl:6s}: {np.percentile(scores_labeled, pct):.6f}")

    importance = booster.feature_importance(importance_type="gain")
    feat_imp = (
        pl.DataFrame({"feature": feature_cols, "gain": importance.tolist()})
        .sort("gain", descending=True)
        .head(30)
    )
    print("\n  Top-30 features by gain:")
    print(feat_imp)

    return pr_auc



# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Score test set and write submission
# ─────────────────────────────────────────────────────────────────────────────

def score_test(
    booster: lgb.Booster,
    feature_cols: list,
    submission_path: str,
    chunk_size: int = 500_000,
) -> None:
    """
    Score the test set and write a submission CSV.

    All data periods live in FEATURES_DIR.  Test rows are identified by:
        is_train == 1  AND  event_dttm >= TRAIN_END_DATE

    Pretest rows (is_train == 0, same date range) are skipped — they were
    only needed as historical context during feature engineering.

    Output: CSV with columns event_id, predict (raw probability in [0,1]).
    Memory: one parquet file in RAM at a time; only event_ids + scores are
    accumulated across files.
    """
    test_files = _parquet_files(FEATURES_DIR)
    if not test_files:
        raise FileNotFoundError(
            f"No parquet files found in {FEATURES_DIR!r}"
        )
    train_end = datetime.fromisoformat(TRAIN_END_DATE)
    print(f"\nScoring test set from {FEATURES_DIR!r} "
          f"({len(test_files)} partitions) …", flush=True)

    all_event_ids: list = []
    all_scores:    list = []
    wall_times:    list = []
    total_test_rows = 0

    for i, f in enumerate(test_files):
        t0 = time.perf_counter()

        chunk = pl.read_parquet(f)

        # Parse datetime if needed
        if chunk["event_dttm"].dtype == pl.Utf8:
            chunk = chunk.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )
        # Test rows: is_train==1 AND event_dttm >= TRAIN_END_DATE.
        # Pretest rows (is_train==0, same dates) are history-only — skip them.
        # Train/val rows (event_dttm < TRAIN_END_DATE) are also skipped here.
        test_chunk = chunk.filter(
            (pl.col("is_train") == 1) & (pl.col("event_dttm") >= train_end)
        )
        del chunk
        gc.collect()

        if len(test_chunk) == 0:
            wall_times.append(time.perf_counter() - t0)
            _progress(i + 1, len(test_files), wall_times,
                      suffix=f"test rows {total_test_rows:,}")
            del test_chunk
            continue

        event_ids = test_chunk["event_id"].to_numpy()

        # Score in sub-chunks to avoid loading a huge matrix at once
        n = len(test_chunk)
        chunk_scores: list = []
        for start in range(0, n, chunk_size):
            end   = min(start + chunk_size, n)
            X_sub = (
                test_chunk[start:end]
                .select(feature_cols)
                .to_numpy(allow_copy=True)
                .astype(np.float32)
            )
            s = booster.predict(
                X_sub, num_iteration=booster.best_iteration
            ).astype(np.float32)
            chunk_scores.append(s)
            del X_sub
            gc.collect()

        scores = np.concatenate(chunk_scores)
        all_event_ids.append(event_ids)
        all_scores.append(scores)
        total_test_rows += n

        del test_chunk, chunk_scores, scores, event_ids
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(test_files), wall_times,
                  suffix=f"test rows {total_test_rows:,}")

    print(flush=True)

    if total_test_rows == 0:
        raise RuntimeError(
            f"No test rows (is_train==1 AND event_dttm >= {TRAIN_END_DATE}) "
            f"found in {FEATURES_DIR!r}. "
            "Check TRAIN_END_DATE and that parquet files contain "
            "is_train and event_dttm columns."
        )

    event_ids_all = np.concatenate(all_event_ids)
    scores_all    = np.concatenate(all_scores)
    del all_event_ids, all_scores
    gc.collect()

    # ── Write submission CSV ──────────────────────────────────────────────────
    submission = pl.DataFrame({
        "event_id": event_ids_all,
        "predict":  scores_all,
    })

    # Sanity checks before writing
    n_dupes = len(submission) - submission["event_id"].n_unique()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes:,} duplicate event_ids in submission — "
              "check if test and pretest partitions overlap.", flush=True)

    submission.write_csv(submission_path)

    print(f"  Test rows scored    : {total_test_rows:,}")
    print(f"  Submission written  : {submission_path}")
    print(f"  Score distribution  :")
    for pct, lbl in [(0,"min"),(25,"p25"),(50,"median"),
                     (75,"p75"),(90,"p90"),(95,"p95"),(99,"p99"),(100,"max")]:
        print(f"    {lbl:6s}: {float(np.percentile(scores_all, pct)):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline() -> None:
    total_start = time.perf_counter()
    cutoff    = datetime.fromisoformat(VAL_CUTOFF_DATE)
    train_end = datetime.fromisoformat(TRAIN_END_DATE)

    files = _parquet_files(FEATURES_DIR)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")
    print(f"Found {len(files)} parquet partitions in {FEATURES_DIR!r}\n",
          flush=True)

    # 1. Build (or reload) memmap files
    # If memmap_meta.json + the four .npy files already exist in STAGING_DIR,
    # count_rows and the full streaming pass are both skipped — only a fast
    # metadata read and four np.memmap(mode="r") calls are made.
    # Delete any of those five files manually to force a full rebuild.
    cached = _load_cache(STAGING_DIR)
    if cached is not None:
        X_train, y_train, X_val, y_val, il_val, feature_cols = cached
        labels = None
    else:
        labels = load_labels()
        n_train, n_val = count_rows(files, cutoff, train_end)
        X_train, y_train, X_val, y_val, il_val, feature_cols = build_memmaps(
            files, labels, cutoff, train_end, n_train, n_val, STAGING_DIR
        )
        del labels
        gc.collect()

    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}\n")

    # 4. Train
    booster = train_model(X_train, y_train, X_val, y_val, il_val, feature_cols)

    # 5. Evaluate
    pr_auc = evaluate(booster, X_val, y_val, il_val, feature_cols)

    # 6. Save model
    booster.save_model(MODEL_OUT_PATH)
    print(f"\n  Model saved → {MODEL_OUT_PATH}")

    # 7. Score test set and write submission
    score_test(booster, feature_cols, SUBMISSION_PATH)

    print(f"\n{'─' * 60}")
    print(f"Total wall time : {_fmt(time.perf_counter() - total_start)}")
    print(f"Final PR-AUC    : {pr_auc:.6f}")
    print("─" * 60)
