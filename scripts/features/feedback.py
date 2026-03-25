"""Section K: Dynamic feedback features from label history.

These features capture per-customer cumulative label statistics: how many
prior transactions were flagged as fraud (red) or confirmed (yellow),
rates, and time-since-last-label signals.

Prerequisites
─────────────
The caller must have already joined labels into the LazyFrame, producing
three Int8 indicator columns:
    _fb_is_red      1 if target==1 (fraud),   else 0
    _fb_is_yellow   1 if target==0 (confirmed non-fraud), else 0
    _fb_is_labeled  1 if labeled (red OR yellow), else 0

All three are 0 for unlabeled ("green") and pretrain/pretest/test rows.

Also requires ``tx_count_lifetime`` from Section B (behavioral.py) and
``event_dttm`` as Datetime from Section A (transaction.py).

Leakage prevention
──────────────────
Every feature uses *strictly prior* information:
  • counts use  cum_sum() − current_value  (excludes the current row)
  • timestamps use  shift(1) + forward_fill  (looks only at prior rows)
"""
import polars as pl
from polars import col, when, lit


# Intermediate columns created inside this module — dropped at the end.
_INTERMEDIATES = [
    "_fb_red_cum", "_fb_yellow_cum", "_fb_labeled_cum",
    "_fb_desc_red_cum", "_fb_desc_yellow_cum", "_fb_desc_labeled_cum",
    "_fb_type_red_cum", "_fb_type_labeled_cum",
    "_fb_subchan_red_cum", "_fb_subchan_labeled_cum",
    "_fb_red_ts", "_fb_yellow_ts",
    "_fb_prev_red_ts_raw", "_fb_prev_yellow_ts_raw",
    "_fb_prev_red_ts", "_fb_prev_yellow_ts",
]


def add_feedback_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add dynamic feedback features derived from label history.

    Returns the LazyFrame with ~22 new ``fb_*`` columns and all
    ``_fb_*`` intermediates dropped.
    """
    # ── 1. Per-customer / per-category cumulative label counts ──────────
    lf = lf.with_columns([
        # Per customer
        col("_fb_is_red").cum_sum().over("customer_id")
            .cast(pl.Int32).alias("_fb_red_cum"),
        col("_fb_is_yellow").cum_sum().over("customer_id")
            .cast(pl.Int32).alias("_fb_yellow_cum"),
        col("_fb_is_labeled").cum_sum().over("customer_id")
            .cast(pl.Int32).alias("_fb_labeled_cum"),

        # Per (customer, event_desc)
        col("_fb_is_red").cum_sum().over(["customer_id", "event_desc"])
            .cast(pl.Int16).alias("_fb_desc_red_cum"),
        col("_fb_is_yellow").cum_sum().over(["customer_id", "event_desc"])
            .cast(pl.Int16).alias("_fb_desc_yellow_cum"),
        col("_fb_is_labeled").cum_sum().over(["customer_id", "event_desc"])
            .cast(pl.Int16).alias("_fb_desc_labeled_cum"),

        # Per (customer, event_type_nm)
        col("_fb_is_red").cum_sum().over(["customer_id", "event_type_nm"])
            .cast(pl.Int16).alias("_fb_type_red_cum"),
        col("_fb_is_labeled").cum_sum().over(["customer_id", "event_type_nm"])
            .cast(pl.Int16).alias("_fb_type_labeled_cum"),

        # Per (customer, channel_indicator_sub_type)
        col("_fb_is_red").cum_sum().over(["customer_id", "channel_indicator_sub_type"])
            .cast(pl.Int16).alias("_fb_subchan_red_cum"),
        col("_fb_is_labeled").cum_sum().over(["customer_id", "channel_indicator_sub_type"])
            .cast(pl.Int16).alias("_fb_subchan_labeled_cum"),

        # Timestamps for "time since last label" features
        when(col("_fb_is_red") == 1)
            .then(col("event_dttm")).otherwise(None)
            .alias("_fb_red_ts"),
        when(col("_fb_is_yellow") == 1)
            .then(col("event_dttm")).otherwise(None)
            .alias("_fb_yellow_ts"),
    ])

    # Shift timestamps by 1 so the current row's label is never visible,
    # then forward-fill to propagate the most recent prior label timestamp.
    lf = lf.with_columns([
        col("_fb_red_ts").shift(1).over("customer_id")
            .alias("_fb_prev_red_ts_raw"),
        col("_fb_yellow_ts").shift(1).over("customer_id")
            .alias("_fb_prev_yellow_ts_raw"),
    ])
    lf = lf.with_columns([
        col("_fb_prev_red_ts_raw").forward_fill().over("customer_id")
            .alias("_fb_prev_red_ts"),
        col("_fb_prev_yellow_ts_raw").forward_fill().over("customer_id")
            .alias("_fb_prev_yellow_ts"),
    ])

    # ── 2. Prior counts (subtract current row's contribution) ───────────
    lf = lf.with_columns([
        # Per-customer
        (col("_fb_red_cum") - col("_fb_is_red"))
            .cast(pl.Int32).alias("fb_cust_prev_red_cnt"),
        (col("_fb_yellow_cum") - col("_fb_is_yellow"))
            .cast(pl.Int32).alias("fb_cust_prev_yellow_cnt"),
        (col("_fb_labeled_cum") - col("_fb_is_labeled"))
            .cast(pl.Int32).alias("fb_cust_prev_labeled_cnt"),

        # Per-(customer, event_desc)
        (col("_fb_desc_red_cum") - col("_fb_is_red"))
            .cast(pl.Int16).alias("fb_desc_prev_red_cnt"),
        (col("_fb_desc_yellow_cum") - col("_fb_is_yellow"))
            .cast(pl.Int16).alias("fb_desc_prev_yellow_cnt"),
        (col("_fb_desc_labeled_cum") - col("_fb_is_labeled"))
            .cast(pl.Int16).alias("fb_desc_prev_labeled_cnt"),

        # Per-(customer, event_type_nm)
        (col("_fb_type_red_cum") - col("_fb_is_red"))
            .cast(pl.Int16).alias("fb_type_prev_red_cnt"),
        (col("_fb_type_labeled_cum") - col("_fb_is_labeled"))
            .cast(pl.Int16).alias("fb_type_prev_labeled_cnt"),

        # Per-(customer, channel_indicator_sub_type)
        (col("_fb_subchan_red_cum") - col("_fb_is_red"))
            .cast(pl.Int16).alias("fb_subchan_prev_red_cnt"),
        (col("_fb_subchan_labeled_cum") - col("_fb_is_labeled"))
            .cast(pl.Int16).alias("fb_subchan_prev_labeled_cnt"),

        # Time since last label (seconds, or -1 if no prior label)
        when(col("_fb_prev_red_ts").is_not_null())
            .then((col("event_dttm") - col("_fb_prev_red_ts")).dt.total_seconds())
            .otherwise(-1)
            .cast(pl.Int32).alias("fb_sec_since_prev_red"),
        when(col("_fb_prev_yellow_ts").is_not_null())
            .then((col("event_dttm") - col("_fb_prev_yellow_ts")).dt.total_seconds())
            .otherwise(-1)
            .cast(pl.Int32).alias("fb_sec_since_prev_yellow"),
    ])

    # ── 3. Derived rates and flags ──────────────────────────────────────
    lf = lf.with_columns([
        # Per-customer smoothed rates  (additive smoothing avoids 0/0)
        ((col("fb_cust_prev_red_cnt") + 0.1)
         / (col("fb_cust_prev_labeled_cnt") + 1.0))
            .cast(pl.Float32).alias("fb_cust_prev_red_rate"),
        ((col("fb_cust_prev_yellow_cnt") + 0.1)
         / (col("fb_cust_prev_labeled_cnt") + 1.0))
            .cast(pl.Float32).alias("fb_cust_prev_yellow_rate"),

        # Suspicious rate: fraction of all prior events that were labeled
        ((col("fb_cust_prev_labeled_cnt").cast(pl.Float32) + 0.1)
         / (col("tx_count_lifetime").cast(pl.Float32) + 1.0))
            .cast(pl.Float32).alias("fb_cust_prev_susp_rate"),

        # Binary flags
        (col("fb_cust_prev_red_cnt") > 0)
            .cast(pl.Int8).alias("fb_cust_prev_any_red"),
        (col("fb_cust_prev_yellow_cnt") > 0)
            .cast(pl.Int8).alias("fb_cust_prev_any_yellow"),

        # Per-(customer, event_desc) smoothed red rate
        ((col("fb_desc_prev_red_cnt") + 0.1)
         / (col("fb_desc_prev_labeled_cnt") + 1.0))
            .cast(pl.Float32).alias("fb_desc_prev_red_rate"),

        # Per-(customer, event_type_nm) smoothed red rate
        ((col("fb_type_prev_red_cnt") + 0.1)
         / (col("fb_type_prev_labeled_cnt") + 1.0))
            .cast(pl.Float32).alias("fb_type_prev_red_rate"),

        # Per-(customer, subchannel) smoothed red rate
        ((col("fb_subchan_prev_red_cnt") + 0.1)
         / (col("fb_subchan_prev_labeled_cnt") + 1.0))
            .cast(pl.Float32).alias("fb_subchan_prev_red_rate"),
    ])

    # ── Cleanup: drop all intermediate columns ──────────────────────────
    lf = lf.drop(_INTERMEDIATES)
    return lf
