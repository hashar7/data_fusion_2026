"""
Orchestrator: iterates over customer partitions, processes each one, writes parquet.

To add a second pipeline variant (e.g. keep pretest rows, different output dir),
add a new function here and import it from feature_calculation.py or the notebook.
"""
import os
import time

import polars as pl

from scripts.calculation.config import OUTPUT_DIR, N_PARTITIONS, COMPRESSION
from scripts.calculation.partition import get_customer_partitions, process_partition


def build_processed_dataset(
    lf: pl.LazyFrame,
    output_dir: str = OUTPUT_DIR,
    n_partitions: int = N_PARTITIONS,
    global_stats: dict[str, pl.LazyFrame] | None = None,
    train_only: bool = True,
    labels_lf: pl.LazyFrame | None = None,
) -> None:
    """
    Process pretrain+train data into per-partition parquet files with all
    engineered features.

    Strategy
    ────────
    Customers are split into `n_partitions` equal-sized buckets.  For each
    bucket we keep ALL rows (all time periods) so that rolling/cumulative
    history features are fully populated, then filter and write.
    Only one chunk is in RAM at a time — peak RAM ≈ total_rows/n_partitions
    × bytes_per_row_after_features.

    Parameters
    ──────────
    lf            : LazyFrame with all periods concatenated.
                    Must contain an `is_train` column (1 = train, 0 = pretrain).
    output_dir    : Directory where parquet partitions will be written.
    n_partitions  : Number of customer buckets / output files (10–100).
    global_stats  : Optional pre-computed global frequency tables from
                    compute_global_stats(). Strongly recommended.
    train_only    : If True (default), only is_train==1 rows are written.
                    Pass False to also retain pretest/test rows in output.
    labels_lf     : Optional LazyFrame with (event_id, target) for label-feedback
                    features (Section K).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Partition customers ────────────────────────────────────────────────
    print("Collecting customer IDs ...", flush=True)
    partitions = get_customer_partitions(lf, n_partitions)
    n          = len(partitions)
    total_customers = sum(len(p) for p in partitions)
    print(
        f"  {total_customers:,} customers → "
        f"{n} partitions × ~{len(partitions[0]):,} customers each",
        flush=True,
    )

    # ── 2. Process and write each partition ───────────────────────────────────
    wall_times: list[float] = []
    total_rows_written = 0

    for i, batch_ids in enumerate(partitions):
        t_start = time.perf_counter()

        result = process_partition(
            lf, batch_ids, global_stats, train_only=train_only, labels_lf=labels_lf,
        )
        n_rows = len(result)
        total_rows_written += n_rows

        out_path = os.path.join(output_dir, f"part_{i:04d}.parquet")
        result.write_parquet(out_path, compression=COMPRESSION)
        del result

        elapsed = time.perf_counter() - t_start
        wall_times.append(elapsed)
        _print_progress(i + 1, n, n_rows, elapsed, wall_times)

    print(
        f"\n\nDone. {total_rows_written:,} total rows written "
        f"across {n} files in '{output_dir}'.",
        flush=True,
    )


# ── Progress helpers ──────────────────────────────────────────────────────────

def _print_progress(
    done: int, total: int, n_rows: int, elapsed: float, wall_times: list
) -> None:
    recent    = wall_times[-5:]
    eta_secs  = (sum(recent) / len(recent)) * (total - done)
    pct       = 100.0 * done / total
    bar       = _bar(done, total)
    print(
        f"\r{bar} {pct:5.1f}%  "
        f"part {done}/{total}  "
        f"({n_rows:,} rows)  "
        f"elapsed {_fmt(elapsed)}  "
        f"ETA {_fmt(eta_secs)}          ",
        end="", flush=True,
    )


def _bar(done: int, total: int, width: int = 30) -> str:
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
