import polars as pl
from polars import col


def add_behavioral_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Section B: Behavioral & User History (Cumulative)."""
    lf = lf.with_columns([
        (col("event_id").cum_count().over("customer_id") - 1).alias("tx_count_lifetime"),
        (col("event_dttm").diff().over("customer_id").dt.total_minutes().fill_null(0)).alias("time_since_last_tx_minutes"),
        (col("mcc_code").cum_count().over(["customer_id", "mcc_code"]) - 1).alias("mcc_freq_user_cum"),
        (col("pos_cd").cum_count().over(["customer_id", "pos_cd"]) - 1).alias("pos_freq_user_cum"),
        # Per-user frequency of transaction descriptor and type (mirrors MCC pattern).
        # global_event_desc_freq is the #2 feature overall; per-user novelty of the
        # same field ("has this customer ever used this event_desc before?") is the
        # natural complement and is currently missing entirely.
        (col("event_desc").cum_count().over(["customer_id", "event_desc"]) - 1).alias("event_desc_user_freq"),
        (col("event_type_nm").cum_count().over(["customer_id", "event_type_nm"]) - 1).alias("event_type_user_freq"),
    ])

    lf = lf.with_columns([
        (col("mcc_freq_user_cum") == 0).cast(pl.Int8).alias("mcc_is_new_for_user"),
        (col("pos_freq_user_cum") == 0).cast(pl.Int8).alias("pos_cd_is_new"),
        (col("mcc_freq_user_cum") / col("tx_count_lifetime")).fill_null(0).alias("mcc_transaction_share_user"),
        (col("pos_freq_user_cum") / col("tx_count_lifetime")).fill_null(0).alias("pos_cd_transaction_share_user"),
        (col("mcc_code") != col("mcc_code").shift(1).over("customer_id")).fill_null(False).cast(pl.Int8).alias("merchant_switch_flag"),
        col("time_since_last_tx_minutes").rolling_mean(window_size=3).over("customer_id").alias("time_since_last_3_tx_mean"),
        col("mcc_freq_user_cum").alias("mcc_frequency_user"),
        # Novelty flags and share for event_desc / event_type
        (col("event_desc_user_freq") == 0).cast(pl.Int8).alias("event_desc_is_new_for_user"),
        (col("event_type_user_freq") == 0).cast(pl.Int8).alias("event_type_is_new_for_user"),
        (col("event_desc_user_freq") / col("tx_count_lifetime")).fill_null(0).alias("event_desc_share_user"),
    ])

    # ── Lag features: attributes of the previous N transactions ──────────────
    # Capture the sequence context: what kind of transaction immediately preceded
    # the current one?  A sudden change in type/channel/desc after N consistent
    # transactions is a strong fraud signal that tree models can exploit.
    #
    # fill_null(-1) marks "no prior transaction" as a distinct sentinel category
    # (cleaner than 0, which is a valid encoded value for these integer columns).
    #
    # Note: prev_1_op_timediff is equivalent to time_since_last_tx_minutes
    # (already computed above).  It is added here for naming-convention
    # consistency within the prev_N_* family.
    _LAG_NS = [1, 2, 3, 4, 5]
    lf = lf.with_columns(
        [col("event_type_nm").shift(n).over("customer_id").fill_null(-1).alias(f"prev_{n}_op_type")        for n in _LAG_NS] +
        [col("event_desc").shift(n).over("customer_id").fill_null(-1).alias(f"prev_{n}_op_desc")           for n in _LAG_NS] +
        [col("channel_indicator_type").shift(n).over("customer_id").fill_null(-1).alias(f"prev_{n}_op_channel")     for n in _LAG_NS] +
        [col("channel_indicator_sub_type").shift(n).over("customer_id").fill_null(-1).alias(f"prev_{n}_op_subchannel") for n in _LAG_NS] +
        [col("time_since_last_tx_minutes").alias("prev_1_op_timediff")]
    )

    return lf
