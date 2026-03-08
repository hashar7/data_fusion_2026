import polars as pl
from polars import col


def add_behavioral_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Section B: Behavioral & User History (Cumulative)."""
    lf = lf.with_columns([
        (col("event_id").cum_count().over("customer_id") - 1).alias("tx_count_lifetime"),
        (col("event_dttm").diff().over("customer_id").dt.total_minutes().fill_null(0)).alias("time_since_last_tx_minutes"),
        (col("mcc_code").cum_count().over(["customer_id", "mcc_code"]) - 1).alias("mcc_freq_user_cum"),
        (col("pos_cd").cum_count().over(["customer_id", "pos_cd"]) - 1).alias("pos_freq_user_cum"),
    ])

    lf = lf.with_columns([
        (col("mcc_freq_user_cum") == 0).cast(pl.Int8).alias("mcc_is_new_for_user"),
        (col("pos_freq_user_cum") == 0).cast(pl.Int8).alias("pos_cd_is_new"),
        (col("mcc_freq_user_cum") / col("tx_count_lifetime")).fill_null(0).alias("mcc_transaction_share_user"),
        (col("pos_freq_user_cum") / col("tx_count_lifetime")).fill_null(0).alias("pos_cd_transaction_share_user"),
        (col("mcc_code") != col("mcc_code").shift(1).over("customer_id")).fill_null(False).cast(pl.Int8).alias("merchant_switch_flag"),
        col("time_since_last_tx_minutes").rolling_mean(window_size=3).over("customer_id").alias("time_since_last_3_tx_mean"),
        col("mcc_freq_user_cum").alias("mcc_frequency_user"),
    ])

    return lf
