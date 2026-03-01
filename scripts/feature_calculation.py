import polars as pl
from polars import col, lit, when
from scripts.feature_engineering import generate_fraud_features_v4

# =============================================================================
# Main pipeline: build the processed feature dataset from pretrain + train data
# =============================================================================

def build_processed_dataset(
    lf: pl.LazyFrame,
    output_dir: str = "../data_processed/",
    n_partitions: int = 50,
    global_stats: dict[str, pl.LazyFrame] | None = None,
) -> None:
    """
    Process pretrain+train data into per-partition parquet files with all
    engineered features.

    Strategy
    ────────
    We split customers into `n_partitions` roughly equal-sized buckets.
    For each bucket we:
        1. Pull ALL rows for those customers (pretrain + train) so that
           rolling/cumulative history features are computed correctly using
           the full available past.
        2. Run generate_fraud_features_v4() on that slice.
        3. Filter down to is_train == 1 rows only.
        4. Downcast float64 → float32 and int64 → int32 to halve memory usage.
        5. Write the result to a parquet file.

    This means each partition is processed independently and peak RAM usage
    is roughly  (total_rows / n_partitions) * bytes_per_row_after_features.
    Increase n_partitions if you run out of memory; 50 is a reasonable default
    for 32 GB RAM and ~200M rows.

    Parameters
    ──────────
    lf            : LazyFrame with pretrain + train data concatenated.
                    Must contain an `is_train` column (1 = train, 0 = pretrain).
    output_dir    : Directory where parquet partitions will be written.
    n_partitions  : Number of customer buckets / output files (10–100).
    global_stats  : Optional pre-computed global frequency tables from
                    compute_global_stats(). Strongly recommended to avoid
                    leakage; if None, per-customer cumulative counts are used.
    """
    import os
    import time
    import math

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Collect the full list of unique customer IDs ───────────────────────
    # This is a small operation (~100k values) and has to be done upfront so we
    # can partition deterministically.
    print("Collecting customer IDs ...", flush=True)
    customer_ids: list[int] = (
        lf.select("customer_id")
        .unique()
        .sort("customer_id")
        .collect()["customer_id"]
        .to_list()
    )
    total_customers = len(customer_ids)
    chunk_size = math.ceil(total_customers / n_partitions)
    actual_partitions = math.ceil(total_customers / chunk_size)
    print(
        f"  {total_customers:,} customers -> "
        f"{actual_partitions} partitions x ~{chunk_size:,} customers each",
        flush=True,
    )

    # ── 2. Iterate over customer buckets ─────────────────────────────────────
    wall_times: list[float] = []   # rolling history for ETA estimation
    total_train_rows_written = 0

    for part_idx in range(actual_partitions):
        t_start = time.perf_counter()

        # ── 2a. Slice customer IDs for this partition ────────────────────────
        batch_ids = customer_ids[part_idx * chunk_size : (part_idx + 1) * chunk_size]

        # ── 2b. Filter LazyFrame to this customer batch ──────────────────────
        #   We keep ALL rows (pretrain + train) so that historical features
        #   (rolling windows, cumulative counts) are fully populated for every
        #   train-period transaction in this batch.
        batch_lf = lf.filter(col("customer_id").is_in(batch_ids))

        # ── 2c. Compute features ─────────────────────────────────────────────
        featured_lf = generate_fraud_features_v4(batch_lf, global_stats=global_stats)

        # ── 2d. Keep only train rows ─────────────────────────────────────────
        train_lf = featured_lf.filter(col("is_train") == 1)

        # ── 2e. Downcast to save disk space and speed up downstream reads ────
        train_lf = train_lf.with_columns([
            col(c).cast(pl.Float32)
            for c in train_lf.collect_schema().names()
            if train_lf.collect_schema()[c] == pl.Float64
        ]).with_columns([
            col(c).cast(pl.Int32)
            for c in train_lf.collect_schema().names()
            if train_lf.collect_schema()[c] == pl.Int64
            and c not in ("customer_id", "event_id", "session_id")
            # keep IDs as Int64 to avoid overflow
        ])

        # ── 2f. Collect and write ────────────────────────────────────────────
        out_path = os.path.join(output_dir, f"part_{part_idx:04d}.parquet")
        result: pl.DataFrame = train_lf.collect()
        n_rows = len(result)
        total_train_rows_written += n_rows
        result.write_parquet(out_path, compression="snappy")
        del result  # release memory immediately

        # ── 2g. Progress reporting ───────────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        wall_times.append(elapsed)

        parts_done = part_idx + 1
        parts_left = actual_partitions - parts_done

        # ETA: use mean of last 5 partitions for a stable rolling estimate
        recent = wall_times[-5:]
        avg_time = sum(recent) / len(recent)
        eta_seconds = avg_time * parts_left
        eta_str = _format_seconds(eta_seconds)

        pct = 100.0 * parts_done / actual_partitions
        bar = _progress_bar(parts_done, actual_partitions, width=30)
        print(
            f"\r{bar} {pct:5.1f}%  "
            f"part {parts_done}/{actual_partitions}  "
            f"({n_rows:,} train rows)  "
            f"elapsed {_format_seconds(elapsed)}  "
            f"ETA {eta_str}          ",
            end="",
            flush=True,
        )

    print(
        f"\n\nDone. {total_train_rows_written:,} total train rows written "
        f"across {actual_partitions} files in '{output_dir}'.",
        flush=True,
    )


# ── Progress bar helpers ──────────────────────────────────────────────────────

def _progress_bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"