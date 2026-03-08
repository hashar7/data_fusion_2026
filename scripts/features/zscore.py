import polars as pl
from polars import col
from scripts.features._helpers import prior_mean_expr, prior_std_expr


def add_zscore_features(
    lf: pl.LazyFrame,
    global_stats: dict[str, pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    """Section G: Channel / MCC Conditional Z-Scores."""

    def _to_lazy(df_or_lf):
        return df_or_lf.lazy() if isinstance(df_or_lf, pl.DataFrame) else df_or_lf

    if global_stats and "channel_stats_global" in global_stats and "mcc_stats_global" in global_stats:
        lf = lf.join(_to_lazy(global_stats["channel_stats_global"]), on="channel_indicator_type", how="left")
        lf = lf.join(_to_lazy(global_stats["mcc_stats_global"]),     on="mcc_code",               how="left")
    else:
        lf = lf.with_columns([
            prior_mean_expr(["customer_id", "channel_indicator_type"]).alias("channel_mean"),
            prior_std_expr( ["customer_id", "channel_indicator_type"]).alias("channel_std"),
            prior_mean_expr(["customer_id", "mcc_code"]).alias("mcc_mean"),
            prior_std_expr( ["customer_id", "mcc_code"]).alias("mcc_std"),
        ])

    lf = lf.with_columns([
        ((col("amount_clean") - col("channel_mean")) / col("channel_std").fill_null(1)).alias("amount_zscore_channel"),
        ((col("amount_clean") - col("mcc_mean"))     / col("mcc_std").fill_null(1)).alias("amount_zscore_mcc"),
    ])

    if global_stats and "combination_global" in global_stats:
        lf = lf.join(_to_lazy(global_stats["combination_global"]), on=["channel_indicator_type", "operating_system_type"], how="left")
        lf = lf.with_columns(col("global_combination_freq").fill_null(0))
    else:
        lf = lf.with_columns(
            (col("event_id").cum_count().over(["customer_id", "channel_indicator_type", "operating_system_type"]) - 1)
            .alias("global_combination_freq")
        )

    def s_num(c):
        return col(c).fill_null(0)

    def s_str(c):
        return col(c).fill_null("Unknown")

    lf = lf.with_columns([
        (col("event_id").cum_count().over(["customer_id", "session_id"]) == 1).cast(pl.Int8).alias("session_first_tx_flag"),
        (col("accept_language") != col("accept_language").shift(1).over("customer_id")).cast(pl.Int8).fill_null(0).alias("language_change_flag"),
        (col("operating_system_type") != col("operating_system_type").shift(1).over("customer_id")).cast(pl.Int8).fill_null(0).alias("os_change_flag"),
        (col("timezone") != col("timezone").shift(1).over("customer_id")).cast(pl.Int8).fill_null(0).alias("timezone_change_flag"),
        (col("time_since_last_tx_minutes") < 5).cast(pl.Int8).alias("rapid_sequence_flag"),
        (col("device_risk_score") > 1).cast(pl.Int8).alias("suspicious_env_flag"),
    ])

    lf = lf.with_columns([
        (col("global_mcc_freq") < 100).cast(pl.Int8).alias("mcc_rare_global_flag"),
        (col("global_combination_freq") < 10).cast(pl.Int8).alias("rare_combination_flag"),
        (col("operating_system_is_new") & (s_num("operaton_amt") > 1000)).cast(pl.Int8).alias("device_change_and_large_amount_flag"),
        (col("phone_voip_call_flag") & col("mcc_is_new_for_user")).cast(pl.Int8).alias("voip_and_new_mcc_flag"),
        (col("compromised_flag") & (s_num("operaton_amt") > col("amount_mean_7d") * 2)).cast(pl.Int8).alias("compromised_and_high_amount_flag"),
        (col("session_first_tx_flag") & (s_num("operaton_amt") > 1000)).cast(pl.Int8).alias("session_first_tx_large_flag"),
        (col("timezone") - col("timezone").shift(1).over("customer_id")).abs().fill_null(0).alias("geo_jump_proxy"),
        (col("device_diversity_30d") / col("tx_count_30d").fill_null(1)).fill_null(0).alias("device_entropy_ratio"),
        (s_str("accept_language") != s_str("browser_language")).cast(pl.Int8).alias("language_mismatch"),
        col("timezone").is_null().cast(pl.Int8).alias("timezone_mismatch"),
        ((col("event_dttm") - col("event_dttm").shift(1).over("customer_id")).dt.total_days()).fill_null(999).alias("merchant_last_seen_days"),
        ((col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "mcc_code"])).dt.total_days()).fill_null(999).alias("mcc_last_seen_days"),
    ])

    return lf
