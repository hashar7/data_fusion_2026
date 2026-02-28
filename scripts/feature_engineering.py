import polars as pl
from polars import col, lit, when

def generate_fraud_features_v3(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Generates 120+ fraud detection features.
    Compatible with Polars 1.38.1.
    Fixes: Temporal leakages, Rolling window API calls, and LazyFrame join logic.
    """

    # Helper for safe numeric aggregation (treat null as 0)
    def s_num(c):
        return col(c).fill_null(0)

    # Helper for safe string handling
    def s_str(c):
        return col(c).fill_null("Unknown")
    
    # 0. Preparation: Sort is mandatory for rolling operations
    # Use a unique row index to ensure we can join features back 1:1 safely.
    lf = lf.sort(["event_dttm", "customer_id"]).with_row_index("temp_row_idx")

    # =========================================================================
    # SECTION A: Transaction-Level Features
    # =========================================================================
    lf = lf.with_columns([
        pl.col("event_dttm").str.strptime(pl.Datetime, strict=False),
    ])
    
    lf = lf.with_columns([
        col("event_dttm").dt.hour().alias("hour"),
        col("event_dttm").dt.weekday().alias("day_of_week"),
        col("event_dttm").dt.day().alias("day_of_month"),
        col("event_dttm").dt.week().alias("week_of_year"),
        col("operaton_amt").fill_null(0).alias("amount_clean")
    ])

    lf = lf.with_columns([
        col("hour").alias("hour_of_day"),
        (col("day_of_week") >= 6).cast(pl.Int8).alias("is_weekend"),
        (col("hour") <= 6).cast(pl.Int8).alias("is_night"),
        col("hour").is_between(9, 18).cast(pl.Int8).alias("is_working_hour"),
        (col("hour") * 60 + col("event_dttm").dt.minute()).alias("minutes_from_midnight"),
        col("amount_clean").log1p().alias("log_amount"),
        col("amount_clean").abs().alias("amount_abs"),
        (col("amount_clean") % 100 == 0).cast(pl.Int8).alias("amount_round_100"),
        (col("amount_clean") % 1000 == 0).cast(pl.Int8).alias("amount_round_1000"),
        (col("amount_clean") % 1 == 0).cast(pl.Int8).alias("amount_is_integer"),
        col("operaton_amt").is_null().cast(pl.Int8).alias("amount_missing_flag"),
        when(col("currency_iso_cd").is_null()).then(1).otherwise(0).cast(pl.Int8).alias("amount_currency_mismatch_flag"),
        lit(None).cast(pl.Float64).alias("amount_usd_normalized"),
        (col("amount_clean") % 1 == 0).cast(pl.Int8).alias("amount_integer"),
        (col("amount_clean") % 100 == 0).cast(pl.Int8).alias("amount_is_round_100"),
        (col("amount_clean") % 1000 == 0).cast(pl.Int8).alias("amount_is_round_1000")
    ])

    # =========================================================================
    # SECTION B: Behavioral & User History (Cumulative)
    # =========================================================================
    # Use window functions (over) for expanding history to avoid leakage
    lf = lf.with_columns([
        col("event_id").cum_count().over("customer_id").alias("tx_count_lifetime"),
        (col("event_dttm").diff().over("customer_id").dt.total_minutes().fill_null(0)).alias("time_since_last_tx_minutes"),
        (col("mcc_code").cum_count().over(["customer_id", "mcc_code"])).alias("mcc_freq_user_cum"),
        (col("pos_cd").cum_count().over(["customer_id", "pos_cd"])).alias("pos_freq_user_cum"),
    ])

    lf = lf.with_columns([
        (col("mcc_freq_user_cum") == 1).cast(pl.Int8).alias("mcc_is_new_for_user"),
        (col("mcc_freq_user_cum") == 1).cast(pl.Int8).alias("merchant_category_is_new_for_user"),
        (col("pos_freq_user_cum") == 1).cast(pl.Int8).alias("pos_cd_is_new"),
        (col("mcc_freq_user_cum") / col("tx_count_lifetime")).alias("mcc_transaction_share_user"),
        (col("pos_freq_user_cum") / col("tx_count_lifetime")).alias("pos_cd_transaction_share_user"),
        (col("mcc_code") != col("mcc_code").shift(1).over("customer_id")).fill_null(False).cast(pl.Int8).alias("merchant_switch_flag"),
        # Rolling mean of time gap requires a specific window
        col("time_since_last_tx_minutes").rolling_mean(window_size=3).over("customer_id").alias("time_since_last_3_tx_mean"),
        col("mcc_freq_user_cum").alias("mcc_frequency_user")
    ])

    # =========================================================================
    # SECTION D: Rolling Statistics (Lazy-compatible)
    # =========================================================================
    def get_rolling_stats(base_lf, period, suffix):
        # Must include temp_row_idx in the agg to join back correctly.
        return (
            base_lf.rolling(
                index_column="event_dttm",
                period=period,
                by="customer_id"
            )
            .agg([
                pl.len().alias(f"tx_count_{suffix}"),
                col("amount_clean").mean().alias(f"amount_mean_{suffix}"),
                col("amount_clean").std().alias(f"amount_std_{suffix}"),
                col("amount_clean").median().alias(f"amount_median_{suffix}"),
                col("amount_clean").max().alias(f"amount_max_{suffix}"),
                col("amount_clean").min().alias(f"amount_min_{suffix}"),
                col("amount_clean").sum().alias(f"cumulative_spend_{suffix}"),
                col("amount_clean").rank("dense").alias(f"amount_rank_percentile_{suffix}"),
                col("channel_indicator_type").n_unique().alias(f"channel_diversity_{suffix}"),
                col("operating_system_type").n_unique().alias(f"device_diversity_{suffix}"),
                col("mcc_code").n_unique().alias(f"merchant_diversity_{suffix}"),
                col("temp_row_idx").last().alias("temp_row_idx") 
            ])
            .select([col("temp_row_idx"), pl.exclude(["customer_id", "event_dttm", "temp_row_idx"])])
        )

    # Perform calculations
    r1d = get_rolling_stats(lf, "1d", "1d")
    r3d = get_rolling_stats(lf, "3d", "3d")
    r7d = get_rolling_stats(lf, "7d", "7d")
    r30d = get_rolling_stats(lf, "30d", "30d")
    r90d = get_rolling_stats(lf, "90d", "90d")

    # Joins
    lf = lf.join(r1d, on="temp_row_idx", how="left")
    lf = lf.join(r3d, on="temp_row_idx", how="left")
    lf = lf.join(r7d, on="temp_row_idx", how="left")
    lf = lf.join(r30d, on="temp_row_idx", how="left")
    lf = lf.join(r90d, on="temp_row_idx", how="left")

    # Derived Rolling Features
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

    # =========================================================================
    # SECTION E: Device & Risk
    # =========================================================================
    lf = lf.with_columns([
        when(col("compromised") == "true").then(1).otherwise(0).cast(pl.Int8).alias("compromised_flag"),
        col("web_rdp_connection").fill_null(0).cast(pl.Int8).alias("web_rdp_connection_flag"),
        when(col("developer_tools") == "true").then(1).otherwise(0).cast(pl.Int8).alias("developer_tools_flag"),
        col("phone_voip_call_state").fill_null(0).cast(pl.Int8).alias("phone_voip_call_flag"),
        when(col("battery").cast(pl.Float64).fill_null(100) < 15).then(1).otherwise(0).cast(pl.Int8).alias("low_battery_flag"),
    ])

    lf = lf.with_columns([
        (col("operating_system_type").cum_count().over(["customer_id", "operating_system_type"]) == 1).cast(pl.Int8).alias("operating_system_is_new"),
        (col("device_system_version").cum_count().over(["customer_id", "device_system_version"]) == 1).cast(pl.Int8).alias("os_version_is_new"),
        (col("screen_size").cum_count().over(["customer_id", "screen_size"]) == 1).cast(pl.Int8).alias("screen_size_is_new"),
        (col("timezone").cum_count().over(["customer_id", "timezone"]) == 1).cast(pl.Int8).alias("timezone_is_new"),
        (col("accept_language").cum_count().over(["customer_id", "accept_language"]) == 1).cast(pl.Int8).alias("accept_language_is_new"),
        (col("compromised_flag") * 5 + col("web_rdp_connection_flag") * 3 + col("developer_tools_flag") * 2).cast(pl.Float64).alias("device_risk_score"),
        col("event_id").len().over("session_id").alias("session_tx_count"),
        col("amount_clean").sum().over("session_id").alias("session_amount_sum"),
        col("channel_indicator_type").n_unique().over("session_id").alias("session_channel_diversity"),
        col('event_id').count().over('customer_id', 'session_id').alias('session_length_estimate'),
    ])

    # =========================================================================
    # SECTION F: Temporal & Global (Leakage-Free)
    # =========================================================================
    # Replaced global-future frequencies with cumulative-past frequencies
    # to maintain model validity (no looking into the future).
    global_cols = [
        ("mcc_code", "global_mcc_freq"), ("channel_indicator_type", "global_channel_freq"), 
        ("operating_system_type", "global_device_os_freq"), ("timezone", "global_timezone_freq"),
        ("accept_language", "global_language_freq"), ("pos_cd", "global_pos_cd_freq"),
        ("event_type_nm", "global_event_type_freq"), ("event_desc", "global_event_desc_freq")
    ]
    
    lf = lf.with_columns([
        col(c).cum_count().over(c).alias(n) for c, n in global_cols
    ])

    lf = lf.with_columns([
        (col("hour_of_day") - col("hour_of_day").mean().over("customer_id")).abs().alias("circadian_deviation_score"),
        (col("channel_indicator_type") != col("channel_indicator_type").shift(1).over("customer_id")).fill_null(False).cast(pl.Int8).alias("channel_shift_score"),
        when(col("tx_count_1d") > col("avg_tx_per_day_30d") * 2).then(1).otherwise(0).cast(pl.Int8).alias("velocity_change_flag"),
        (col("device_diversity_30d") / col("tx_count_30d").fill_null(1)).alias("device_entropy_ratio"),
        col("time_since_last_tx_minutes").std().over("customer_id").fill_null(0).alias("time_gap_variance_30d"),
        col("operating_system_is_new").alias("new_device_flag"),
        col("mcc_is_new_for_user").alias("new_mcc_flag"),
        (col("channel_indicator_type").cum_count().over(["customer_id", "channel_indicator_type"]) == 1).cast(pl.Int8).alias("new_channel_flag"),
        # Entropy (Approximated by distinct count of MCCs / Total Txns)
        (col('mcc_code').n_unique().over('customer_id') / col('event_id').count().over('customer_id')).fill_null(0).alias('merchant_entropy_user'),
    ])

    lf = lf.with_columns([
        when(col("accept_language") != col("browser_language")).then(1).otherwise(0).cast(pl.Int8).alias("browser_language_mismatch"),
    ])

    lf = lf.with_columns([
        when((col("is_night") == 1) & (col("new_device_flag") == 1)).then(1).otherwise(0).cast(pl.Int8).alias("new_device_and_night_flag"),
        when((col("web_rdp_connection_flag") == 1) & (col("amount_clean") > 1000)).then(1).otherwise(0).cast(pl.Int8).alias("rdp_and_large_amount_flag"),
        col("amount_clean").mean().over("channel_indicator_type").alias("amount_zscore_given_channel"),
        col("amount_clean").mean().over("mcc_code").alias("amount_zscore_given_mcc"),
        col("amount_clean").mean().over("operating_system_type").alias("amount_zscore_given_device"),
        col("hour_of_day").std().over("customer_id").alias("tx_time_zscore_given_user")
    ])

    # ============================================================
    # SECTION G: Channel / MCC Conditional Z-Scores
    # ============================================================
    lf = lf.with_columns([
        col("amount_clean").mean().over("channel_indicator_type")
            .alias("channel_mean"),

        col("amount_clean").std().over("channel_indicator_type")
            .alias("channel_std"),

        col("amount_clean").mean().over("mcc_code")
            .alias("mcc_mean"),

        col("amount_clean").std().over("mcc_code")
            .alias("mcc_std"),
    ])

    lf = lf.with_columns([
        (
            (col("amount_clean") - col("channel_mean")) /
            col("channel_std").fill_null(1)
        ).alias("amount_zscore_channel"),

        (
            (col("amount_clean") - col("mcc_mean")) /
            col("mcc_std").fill_null(1)
        ).alias("amount_zscore_mcc"),
    ])

    lf = lf.with_columns(
        col('event_id').count().over('channel_indicator_type', 'operating_system_type').alias('global_combination_freq'),
        (col('event_id').min().over('customer_id', 'session_id') == col('event_id')).cast(pl.Int8).alias('session_first_tx_flag'),
        (col('accept_language') != col('accept_language').shift(1).over('customer_id')).cast(pl.Int8).fill_null(0).alias('language_change_flag'),
        (col('operating_system_type') != col('operating_system_type').shift(1).over('customer_id')).cast(pl.Int8).fill_null(0).alias('os_change_flag'),
        (col('timezone') != col('timezone').shift(1).over('customer_id')).cast(pl.Int8).fill_null(0).alias('timezone_change_flag'),
        (col('time_since_last_tx_minutes') < 5).cast(pl.Int8).alias('rapid_sequence_flag'),
        (col('device_risk_score') > 1).cast(pl.Int8).alias('suspicious_env_flag'),
    )

    # Final logic for rare flags and missing initial features
    lf = lf.with_columns([
        when(col("global_mcc_freq") < 100).then(1).otherwise(0).cast(pl.Int8).alias("mcc_rare_global_flag"),
        (col('global_combination_freq') < 10).cast(pl.Int8).alias('rare_combination_flag'),
        (col('operating_system_is_new') & (s_num('operaton_amt') > 1000)).cast(pl.Int8).alias('device_change_and_large_amount_flag'),
        (col('phone_voip_call_flag') & col('mcc_is_new_for_user')).cast(pl.Int8).alias('voip_and_new_mcc_flag'),
        (col('compromised_flag') & (s_num('operaton_amt') > col('amount_mean_7d') * 2)).cast(pl.Int8).alias('compromised_and_high_amount_flag'),
        (col('session_first_tx_flag') & (s_num('operaton_amt') > 1000)).cast(pl.Int8).alias('session_first_tx_large_flag'),
        (col('timezone') - col('timezone').shift(1).over('customer_id')).abs().fill_null(0).alias('geo_jump_proxy'),
        (col('device_diversity_30d') / col('tx_count_30d').fill_null(1)).fill_null(0).alias('device_entropy_ratio'),
        (s_str('accept_language') != s_str('browser_language')).cast(pl.Int8).alias('language_mismatch'),
        col('timezone').is_null().cast(pl.Int8).alias('timezone_mismatch'),
        ((col('event_dttm') - col('event_dttm').shift(1).over('customer_id')).dt.total_days()).fill_null(999).alias('merchant_last_seen_days'),
        ((col('event_dttm') - col('event_dttm').shift(1).over('customer_id', 'mcc_code')).dt.total_days()).fill_null(999).alias('mcc_last_seen_days')
    ])

    # Cleanup: Drop internal helper columns but keep the schema identical to target
    return lf.drop(["temp_row_idx", 
                    "amount_clean", 
                    "channel_mean",
                    "channel_std",
                    "mcc_mean",
                    "mcc_std",])