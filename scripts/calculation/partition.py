"""
Per-partition logic: customer bucketing, feature computation, downcasting.

To add a new processing step to every partition (e.g. an extra filter,
a derived column, a validation check), add it to process_partition().

To change downcasting rules (e.g. keep a column as Float64), edit _downcast().
"""
import math

import polars as pl
from polars import col

from scripts.feature_engineering import generate_fraud_features_v4
from scripts.calculation.config import ID_COLS


def get_customer_partitions(lf: pl.LazyFrame, n_partitions: int) -> list[list]:
    """
    Collect all unique customer_ids and split them into n_partitions buckets.

    Returns a list of lists, each containing the customer_ids for one partition.
    Deterministic (sorted before splitting) so reruns produce identical files.
    """
    customer_ids: list = (
        lf.select("customer_id")
        .unique()
        .sort("customer_id")
        .collect()["customer_id"]
        .to_list()
    )
    chunk_size = math.ceil(len(customer_ids) / n_partitions)
    return [
        customer_ids[i * chunk_size : (i + 1) * chunk_size]
        for i in range(math.ceil(len(customer_ids) / chunk_size))
    ]


def process_partition(
    lf: pl.LazyFrame,
    batch_ids: list,
    global_stats: dict | None = None,
    train_only: bool = True,
    labels_lf: pl.LazyFrame | None = None,
) -> pl.DataFrame:
    """
    Process one customer batch end-to-end and return the result as a DataFrame.

    Steps
    ─────
    1. Filter to this batch's customers (keeps ALL time periods so that
       rolling/cumulative history features are fully populated).
    2. Compute all engineered features via generate_fraud_features_v4.
    3. Optionally filter to is_train == 1 rows only (default: True).
    4. Downcast float64 → float32 and large int64 → int32 to reduce disk size.

    Parameters
    ──────────
    lf           : Full LazyFrame (all customers, all time periods).
    batch_ids    : Customer IDs belonging to this partition.
    global_stats : Pre-computed global frequency tables (strongly recommended).
                   If None, per-customer cumulative counts are used as a proxy.
    train_only   : If True (default), only rows with is_train == 1 are returned.
                   Set to False to keep pretest/test rows as well.
    labels_lf    : Optional LazyFrame with (event_id, target) for label-feedback
                   features.  When provided, per-customer cumulative label
                   statistics are computed (Section K).
    """
    batch_lf    = lf.filter(col("customer_id").is_in(batch_ids))
    featured_lf = generate_fraud_features_v4(
        batch_lf, global_stats=global_stats, labels_lf=labels_lf,
    )

    if train_only:
        featured_lf = featured_lf.filter(col("is_train") == 1)

    return _downcast(featured_lf).collect()


def _downcast(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Downcast float64 → float32 and non-ID int64 → int32.

    Halves memory and disk usage with no loss of precision for the feature
    values used here. ID columns (customer_id, event_id, session_id) are
    kept as Int64 to avoid overflow.

    To exclude additional columns from downcasting, add them to ID_COLS in config.py.
    """
    schema = lf.collect_schema()
    float_cols = [c for c, t in schema.items() if t == pl.Float64]
    int_cols   = [c for c, t in schema.items()
                  if t == pl.Int64 and c not in ID_COLS]

    if float_cols:
        lf = lf.with_columns([col(c).cast(pl.Float32) for c in float_cols])
    if int_cols:
        lf = lf.with_columns([col(c).cast(pl.Int32)   for c in int_cols])
    return lf
