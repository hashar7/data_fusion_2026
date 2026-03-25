import polars as pl
from polars import col, lit

from scripts.features.transaction  import add_transaction_features
from scripts.features.behavioral   import add_behavioral_features
from scripts.features.rolling      import add_rolling_features
from scripts.features.device       import add_device_features
from scripts.features.temporal     import add_temporal_features
from scripts.features.zscore       import add_zscore_features
from scripts.features.category_stats import add_category_stats_features
from scripts.features.category_risk  import add_category_risk_features
from scripts.features.feedback     import add_feedback_features
from scripts.features.global_stats import compute_global_stats  # re-exported for callers

_COLS_TO_DROP = [
    "temp_row_idx", "amount_clean", "channel_mean", "channel_std",
    "mcc_mean", "mcc_std",
    # Internal feedback columns (created by _join_feedback_labels)
    "_fb_target", "_fb_is_red", "_fb_is_yellow", "_fb_is_labeled",
]


def _join_feedback_labels(
    lf: pl.LazyFrame,
    labels_lf: pl.LazyFrame | None,
) -> pl.LazyFrame:
    """Join labels and create feedback indicator columns.

    Produces three Int8 columns consumed by add_feedback_features():
        _fb_is_red      1 if fraud (target==1)
        _fb_is_yellow   1 if confirmed non-fraud (target==0, labeled)
        _fb_is_labeled  1 if labeled at all (red OR yellow)

    When labels_lf is None, all indicators are zero — feedback features
    will degrade gracefully to their default values.
    """
    if labels_lf is not None:
        lf = lf.join(
            labels_lf.select([
                col("event_id"),
                col("target").alias("_fb_target"),
            ]),
            on="event_id",
            how="left",
        )
        lf = lf.with_columns([
            (col("_fb_target") == 1).fill_null(False).cast(pl.Int8).alias("_fb_is_red"),
            (col("_fb_target") == 0).fill_null(False).cast(pl.Int8).alias("_fb_is_yellow"),
        ])
    else:
        lf = lf.with_columns([
            lit(None).cast(pl.Int8).alias("_fb_target"),
            lit(0).cast(pl.Int8).alias("_fb_is_red"),
            lit(0).cast(pl.Int8).alias("_fb_is_yellow"),
        ])

    lf = lf.with_columns(
        (col("_fb_is_red") + col("_fb_is_yellow")).cast(pl.Int8).alias("_fb_is_labeled")
    )
    return lf


def generate_fraud_features(
    lf: pl.LazyFrame,
    global_stats: dict[str, pl.LazyFrame] | None = None,
    labels_lf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    lf = lf.sort(["customer_id", "event_dttm"]).with_row_index("temp_row_idx")
    lf = _join_feedback_labels(lf, labels_lf)
    lf = add_transaction_features(lf)
    lf = add_behavioral_features(lf)
    lf = add_feedback_features(lf)          # Section K — after B (needs tx_count_lifetime)
    lf = add_rolling_features(lf)
    lf = add_device_features(lf)
    lf = add_temporal_features(lf, global_stats)
    lf = add_zscore_features(lf, global_stats)
    lf = add_category_stats_features(lf, global_stats)
    lf = add_category_risk_features(lf)
    return lf.drop(_COLS_TO_DROP)


# Backward-compatible alias — feature_calculation.py and the notebook use this name
generate_fraud_features_v4 = generate_fraud_features
