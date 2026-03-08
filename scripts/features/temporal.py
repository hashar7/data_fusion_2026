import polars as pl
from polars import col, when


def add_temporal_features(
    lf: pl.LazyFrame,
    global_stats: dict[str, pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    """Section F: Temporal & Global (Leakage-Free)."""
    global_col_map = [
        # (raw_col,                    alias,                    global_stats_key)
        ("mcc_code",                   "global_mcc_freq",        "mcc_global"),
        ("channel_indicator_type",     "global_channel_freq",    "channel_global"),
        ("operating_system_type",      "global_device_os_freq",  "os_global"),
        ("timezone",                   "global_timezone_freq",   "timezone_global"),
        ("accept_language",            "global_language_freq",   "language_global"),
        ("pos_cd",                     "global_pos_cd_freq",     "pos_global"),
        ("event_type_nm",              "global_event_type_freq", "evtype_global"),
        ("event_desc",                 "global_event_desc_freq", "evdesc_global"),
    ]

    if global_stats:
        for raw_col, alias, key in global_col_map:
            if key in global_stats:
                lf = lf.join(
                    global_stats[key].lazy() if isinstance(global_stats[key], pl.DataFrame)
                    else global_stats[key],
                    on=raw_col,
                    how="left",
                ).rename({alias: alias})
            else:
                lf = lf.with_columns([
                    (col(raw_col).cum_count().over(["customer_id", raw_col]) - 1).alias(alias)
                ])
    else:
        lf = lf.with_columns([
            (col(raw_col).cum_count().over(["customer_id", raw_col]) - 1).alias(alias)
            for raw_col, alias, _ in global_col_map
        ])

    lf = lf.with_columns([
        (
            col("hour_of_day") -
            col("hour_of_day").cum_sum().over("customer_id").shift(1).fill_null(0) /
            col("tx_count_lifetime").clip(lower_bound=1)
        ).abs().alias("circadian_deviation_score"),
        (col("channel_indicator_type") != col("channel_indicator_type").shift(1).over("customer_id")).fill_null(False).cast(pl.Int8).alias("channel_shift_score"),
        when(col("tx_count_1d") > col("avg_tx_per_day_30d") * 2).then(1).otherwise(0).cast(pl.Int8).alias("velocity_change_flag"),
        col("time_since_last_tx_minutes").shift(1).rolling_std(window_size=90, min_periods=2).over("customer_id").fill_null(0).alias("time_gap_variance_30d"),
        col("operating_system_is_new").alias("new_device_flag"),
        col("mcc_is_new_for_user").alias("new_mcc_flag"),
        (col("channel_indicator_type").cum_count().over(["customer_id", "channel_indicator_type"]) - 1 == 0).cast(pl.Int8).alias("new_channel_flag"),
        (col("mcc_freq_user_cum") / col("tx_count_lifetime")).fill_null(0).alias("merchant_entropy_user"),
    ])

    lf = lf.with_columns([
        when(col("accept_language") != col("browser_language")).then(1).otherwise(0).cast(pl.Int8).alias("browser_language_mismatch"),
    ])

    lf = lf.with_columns([
        when((col("is_night") == 1) & (col("new_device_flag") == 1)).then(1).otherwise(0).cast(pl.Int8).alias("new_device_and_night_flag"),
        when((col("web_rdp_connection_flag") == 1) & (col("amount_clean") > 1000)).then(1).otherwise(0).cast(pl.Int8).alias("rdp_and_large_amount_flag"),
        (
            (col("amount_clean").cum_sum().over(["customer_id", "channel_indicator_type"]) - col("amount_clean")) /
            (col("event_id").cum_count().over(["customer_id", "channel_indicator_type"]) - 1).clip(lower_bound=1)
        ).fill_null(0).alias("amount_zscore_given_channel"),
        (
            (col("amount_clean").cum_sum().over(["customer_id", "mcc_code"]) - col("amount_clean")) /
            (col("event_id").cum_count().over(["customer_id", "mcc_code"]) - 1).clip(lower_bound=1)
        ).fill_null(0).alias("amount_zscore_given_mcc"),
        (
            (col("amount_clean").cum_sum().over(["customer_id", "operating_system_type"]) - col("amount_clean")) /
            (col("event_id").cum_count().over(["customer_id", "operating_system_type"]) - 1).clip(lower_bound=1)
        ).fill_null(0).alias("amount_zscore_given_device"),
        col("hour_of_day").shift(1).rolling_std(window_size=90, min_periods=2).over("customer_id").fill_null(0).alias("tx_time_zscore_given_user"),
    ])

    return lf
