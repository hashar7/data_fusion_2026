import polars as pl
from polars import col, when
from scripts.features._helpers import get_rolling_stats


def add_rolling_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Section D: Rolling Statistics (Lazy-compatible)."""
    r1d  = get_rolling_stats(lf, "1d",  "1d")
    r3d  = get_rolling_stats(lf, "3d",  "3d")
    r7d  = get_rolling_stats(lf, "7d",  "7d")
    r30d = get_rolling_stats(lf, "30d", "30d")
    r90d = get_rolling_stats(lf, "90d", "90d")

    lf = lf.join(r1d,  on="temp_row_idx", how="left")
    lf = lf.join(r3d,  on="temp_row_idx", how="left")
    lf = lf.join(r7d,  on="temp_row_idx", how="left")
    lf = lf.join(r30d, on="temp_row_idx", how="left")
    lf = lf.join(r90d, on="temp_row_idx", how="left")

    EPS = 1e-9
    lf = lf.with_columns([
        (
            (col("amount_clean") - col("amount_min_7d")) /
            (col("amount_max_7d") - col("amount_min_7d") + EPS)
        ).alias("amount_rank_percentile_7d"),
        (
            (col("amount_clean") - col("amount_min_30d")) /
            (col("amount_max_30d") - col("amount_min_30d") + EPS)
        ).alias("amount_rank_percentile_30d"),
        (
            (col("amount_clean") - col("amount_min_90d")) /
            (col("amount_max_90d") - col("amount_min_90d") + EPS)
        ).alias("amount_rank_percentile_90d"),
    ])

    lf = lf.with_columns([
        ((col("amount_clean") - col("amount_mean_30d")) / col("amount_std_30d").fill_null(1)).alias("amount_zscore_30d"),
        (col("amount_clean") / col("amount_mean_30d").fill_null(1)).alias("amount_ratio_to_mean_30d"),
        (col("tx_count_30d") / 30.0).alias("avg_tx_per_day_30d"),
        (col("cumulative_spend_30d") / 30.0).alias("spend_velocity_1d"),
        (col("tx_count_7d") / col("tx_count_90d").fill_null(1)).alias("tx_count_ratio_7d_vs_90d"),
        (col("amount_mean_7d") / col("amount_mean_30d").fill_null(1)).alias("amount_mean_ratio_7d_vs_90d"),
        (col("amount_clean") - col("amount_clean").shift(1).over("customer_id")).fill_null(0).alias("amount_diff_from_prev"),
        (col("amount_clean") / col("amount_clean").shift(1).over("customer_id").fill_null(1)).alias("amount_ratio_prev"),
        when(col("time_since_last_tx_minutes") < 5).then(1).otherwise(0).cast(pl.Int8).alias("burst_flag"),
    ])

    return lf
