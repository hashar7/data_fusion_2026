import polars as pl
from polars import col


def prior_mean_expr(group_cols: list) -> pl.Expr:
    """Expanding mean of amount_clean for group_cols, excluding the current row."""
    n = (col("event_id").cum_count().over(group_cols) - 1).clip(lower_bound=1)
    s = col("amount_clean").cum_sum().over(group_cols) - col("amount_clean")
    return (s / n).fill_null(0)


def prior_std_expr(group_cols: list) -> pl.Expr:
    """Expanding sample std of amount_clean for group_cols, excluding the current row."""
    n   = col("event_id").cum_count().over(group_cols) - 1
    s   = col("amount_clean").cum_sum().over(group_cols) - col("amount_clean")
    sq  = (col("amount_clean") ** 2).cum_sum().over(group_cols) - col("amount_clean") ** 2
    var = (sq - s ** 2 / n.clip(lower_bound=1)) / (n - 1).clip(lower_bound=1)
    return var.clip(lower_bound=0).sqrt()


def get_rolling_stats(base_lf: pl.LazyFrame, period: str, suffix: str) -> pl.LazyFrame:
    """Single rolling window LazyFrame keyed by temp_row_idx."""
    return (
        base_lf.rolling(
            index_column="event_dttm",
            period=period,
            by="customer_id",
            closed="left",
        )
        .agg([
            pl.len().alias(f"tx_count_{suffix}"),
            col("amount_clean").mean().alias(f"amount_mean_{suffix}"),
            col("amount_clean").std().alias(f"amount_std_{suffix}"),
            col("amount_clean").median().alias(f"amount_median_{suffix}"),
            col("amount_clean").max().alias(f"amount_max_{suffix}"),
            col("amount_clean").min().alias(f"amount_min_{suffix}"),
            col("amount_clean").sum().alias(f"cumulative_spend_{suffix}"),
            col("channel_indicator_type").n_unique().alias(f"channel_diversity_{suffix}"),
            col("operating_system_type").n_unique().alias(f"device_diversity_{suffix}"),
            col("mcc_code").n_unique().alias(f"merchant_diversity_{suffix}"),
            col("amount_card").sum().alias(f"card_spend_{suffix}"),
            col("amount_p2p").sum().alias(f"p2p_spend_{suffix}"),
            col("event_desc").n_unique().alias(f"event_desc_diversity_{suffix}"),
            col("event_type_nm").n_unique().alias(f"event_type_diversity_{suffix}"),
        ])
        .with_row_index("temp_row_idx")
        .select([col("temp_row_idx"), pl.exclude(["customer_id", "event_dttm", "temp_row_idx"])])
    )
