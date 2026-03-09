import polars as pl
from polars import col, when
from scripts.features._helpers import get_rolling_stats


def add_rolling_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Section D: Rolling Statistics (Lazy-compatible)."""
    # Sub-day windows — useful for burst / velocity fraud signals
    r15m = get_rolling_stats(lf, "15m", "15m")
    r1h  = get_rolling_stats(lf, "1h",  "1h")
    r6h  = get_rolling_stats(lf, "6h",  "6h")
    r12h = get_rolling_stats(lf, "12h", "12h")
    # Day+ windows — useful for longer-horizon behavioural baselines
    r1d  = get_rolling_stats(lf, "1d",  "1d")
    r3d  = get_rolling_stats(lf, "3d",  "3d")
    r7d  = get_rolling_stats(lf, "7d",  "7d")
    r30d = get_rolling_stats(lf, "30d", "30d")
    r90d = get_rolling_stats(lf, "90d", "90d")

    lf = lf.join(r15m, on="temp_row_idx", how="left")
    lf = lf.join(r1h,  on="temp_row_idx", how="left")
    lf = lf.join(r6h,  on="temp_row_idx", how="left")
    lf = lf.join(r12h, on="temp_row_idx", how="left")
    lf = lf.join(r1d,  on="temp_row_idx", how="left")
    lf = lf.join(r3d,  on="temp_row_idx", how="left")
    lf = lf.join(r7d,  on="temp_row_idx", how="left")
    lf = lf.join(r30d, on="temp_row_idx", how="left")
    lf = lf.join(r90d, on="temp_row_idx", how="left")

    EPS = 1e-9

    # ── Sub-day derived features ──────────────────────────────────────────────
    # Burst flags: multiple transactions in a very short window is a strong
    # fraud signal (card-testing, rapid credential abuse, etc.).
    lf = lf.with_columns([
        (col("tx_count_15m") > 2).cast(pl.Int8).alias("burst_flag_15m"),
        (col("tx_count_1h")  > 5).cast(pl.Int8).alias("burst_flag_1h"),

        # Spend ratios: what fraction of the daily spend happened in the last
        # 15 min / 1 h / 6 h / 12 h?  High ratios signal a sudden burst.
        (col("cumulative_spend_15m") / (col("cumulative_spend_1d").fill_null(0) + EPS)).alias("spend_ratio_15m_vs_1d"),
        (col("cumulative_spend_1h")  / (col("cumulative_spend_1d").fill_null(0) + EPS)).alias("spend_ratio_1h_vs_1d"),
        (col("cumulative_spend_6h")  / (col("cumulative_spend_1d").fill_null(0) + EPS)).alias("spend_ratio_6h_vs_1d"),
        (col("cumulative_spend_12h") / (col("cumulative_spend_1d").fill_null(0) + EPS)).alias("spend_ratio_12h_vs_1d"),

        # TX-count ratios: concentration of activity in a short sub-window.
        (col("tx_count_15m") / (col("tx_count_1h").fill_null(0) + EPS)).alias("tx_count_ratio_15m_vs_1h"),
        (col("tx_count_1h")  / (col("tx_count_1d").fill_null(0) + EPS)).alias("tx_count_ratio_1h_vs_1d"),

        # Amount vs. short-window mean: is the current transaction unusually
        # large relative to the last hour's activity?
        (col("amount_clean") / (col("amount_mean_1h").fill_null(1) + EPS)).alias("amount_ratio_to_mean_1h"),
        (col("amount_clean") / (col("amount_mean_6h").fill_null(1) + EPS)).alias("amount_ratio_to_mean_6h"),
    ])

    # ── Day+ derived features ─────────────────────────────────────────────────
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
        # Spend ratios: what fraction of long-term spend happened in a short recent window?
        # The model already uses spend_1d and spend_90d independently (#1 and #3 features);
        # the ratio makes the normalised deviation explicit and easier to split on.
        (col("cumulative_spend_1d") / (col("cumulative_spend_30d").fill_null(0) + 1e-9)).alias("spend_ratio_1d_vs_30d"),
        (col("cumulative_spend_1d") / (col("cumulative_spend_90d").fill_null(0) + 1e-9)).alias("spend_ratio_1d_vs_90d"),
        (col("cumulative_spend_7d") / (col("cumulative_spend_90d").fill_null(0) + 1e-9)).alias("spend_ratio_7d_vs_90d"),
        # Per-window spend velocity (complement to the existing spend_velocity_1d=spend_30d/30)
        (col("cumulative_spend_7d")  / 7.0 ).alias("spend_velocity_7d"),
        (col("cumulative_spend_30d") / 30.0).alias("spend_velocity_30d"),
    ])

    # Percentile threshold flags: binary markers for "exceptionally high" transactions.
    # amount_rank_percentile_90d is already rank-18; these give the model clean binary
    # splits at the extreme tail without requiring it to learn the threshold itself.
    lf = lf.with_columns([
        (col("amount_rank_percentile_30d") > 0.95).cast(pl.Int8).alias("amount_top5pct_30d"),
        (col("amount_rank_percentile_90d") > 0.99).cast(pl.Int8).alias("amount_top1pct_90d"),
        # New personal spending record: amount exceeds any transaction in prior 90 days
        (col("amount_clean") > col("amount_max_90d").fill_null(0)).cast(pl.Int8).alias("amount_above_personal_max_flag"),
    ])

    # Per-channel cumulative stats (leakage-free: prior rows only, same -1 shift).
    # global_channel_freq is rank-13; these add the personal dimension: how much does
    # THIS customer typically use this channel, and is this transaction unusual for them?
    lf = lf.with_columns([
        (col("amount_clean").cum_sum().over(["customer_id", "channel_indicator_type"]) - col("amount_clean")).fill_null(0).alias("spend_in_channel_lifetime"),
        (col("event_id").cum_count().over(["customer_id", "channel_indicator_type"]) - 1).alias("tx_count_in_channel_lifetime"),
    ])
    lf = lf.with_columns([
        (col("tx_count_in_channel_lifetime") / col("tx_count_lifetime").clip(lower_bound=1)).fill_null(0).alias("channel_usage_share"),
    ])

    return lf
