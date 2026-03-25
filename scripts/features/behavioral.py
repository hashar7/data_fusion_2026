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

    # ── Running-max features (leakage-free) ───────────────────────────────────
    # cum_max().shift(1).over(group) gives the maximum of ALL prior transactions
    # in the group, excluding the current row.
    # Competitor uses rolling_max(window_size=9999999) WITHOUT a shift — that
    # includes the current row in its own max, which is a leakage issue.
    # Having the raw prior-max value (not just a binary flag) lets the model
    # express "this amount is N× larger than anything the customer has done before."
    lf = lf.with_columns([
        col("amount_clean").cum_max().shift(1).over("customer_id")
        .fill_null(0).alias("operaton_amt_max_prev"),

        col("amount_clean").cum_max().shift(1).over(["customer_id", "mcc_code"])
        .fill_null(0).alias("operaton_amt_mcc_max_prev"),

        col("amount_clean").cum_max().shift(1).over(["customer_id", "channel_indicator_type"])
        .fill_null(0).alias("operaton_amt_type_max_prev"),

        col("amount_clean").cum_max().shift(1).over(["customer_id", "event_desc"])
        .fill_null(0).alias("operaton_amt_desc_max_prev"),

        col("amount_clean").cum_max().shift(1).over(["customer_id", "channel_indicator_sub_type"])
        .fill_null(0).alias("operaton_amt_sub_max_prev"),

        col("amount_clean").cum_max().shift(1).over(["customer_id", "channel_type_subtype"])
        .fill_null(0).alias("operaton_amt_type_subtype_max_prev"),

        col("amount_clean").cum_max().shift(1).over(["customer_id", "evtype_channel"])
        .fill_null(0).alias("operaton_amt_evtype_channel_max_prev"),

        col("amount_clean").cum_max().shift(1).over(["customer_id", "evtype_subchannel"])
        .fill_null(0).alias("operaton_amt_evtype_subchannel_max_prev"),

        col("amount_clean").cum_max().shift(1).over(["customer_id", "event_type_nm", "mcc_code"])
        .fill_null(0).alias("operaton_amt_evtype_mcc_max_prev"),
    ])

    # ── Log-frequency features ────────────────────────────────────────────────
    # log1p(prior_count) compresses multi-order-of-magnitude ranges and exposes
    # a smoother gradient for tree splits on count columns.
    #
    # Two kinds:
    # (a) Log-transform of existing linear cumulative counts — the competitor
    #     shows these rank far higher than we'd expect from linear versions.
    # (b) New per-(customer, value) frequency for columns tracked elsewhere only
    #     as binary "is-new" flags.
    lf = lf.with_columns([
        # (a) Log of already-computed counts
        col("event_desc_user_freq").cast(pl.Float32).log1p().alias("event_desc_log_count"),
        col("mcc_freq_user_cum").cast(pl.Float32).log1p().alias("mcc_log_count"),
        col("pos_freq_user_cum").cast(pl.Float32).log1p().alias("pos_cd_log_count"),
        col("event_type_user_freq").cast(pl.Float32).log1p().alias("event_type_nm_log_count"),

        # (b) New per-user frequencies — inline cum_count - 1 = prior count
        (col("timezone").cum_count().over(["customer_id", "timezone"]) - 1)
        .cast(pl.Float32).log1p().alias("timezone_log_count"),

        (col("operating_system_type").cum_count().over(["customer_id", "operating_system_type"]) - 1)
        .cast(pl.Float32).log1p().alias("operating_system_type_log_count"),

        (col("channel_indicator_type").cum_count().over(["customer_id", "channel_indicator_type"]) - 1)
        .cast(pl.Float32).log1p().alias("channel_indicator_type_log_count"),

        (col("channel_type_subtype").cum_count().over(["customer_id", "channel_type_subtype"]) - 1)
        .cast(pl.Float32).log1p().alias("channel_type_subtype_log_count"),

        (col("evtype_channel").cum_count().over(["customer_id", "evtype_channel"]) - 1)
        .cast(pl.Float32).log1p().alias("evtype_channel_log_count"),

        (col("evtype_subchannel").cum_count().over(["customer_id", "evtype_subchannel"]) - 1)
        .cast(pl.Float32).log1p().alias("evtype_subchannel_log_count"),

        (col("event_type_nm").cum_count().over(["customer_id", "event_type_nm", "mcc_code"]) - 1)
        .cast(pl.Float32).log1p().alias("evtype_mcc_log_count"),

        (col("device_system_version").cum_count().over(["customer_id", "device_system_version"]) - 1)
        .cast(pl.Float32).log1p().alias("device_system_version_log_count"),

        # Device-state frequency: "has the customer always had compromised / dev-tools on?"
        (col("compromised").cum_count().over(["customer_id", "compromised"]) - 1)
        .cast(pl.Float32).log1p().alias("compromised_log_count"),

        (col("developer_tools").cum_count().over(["customer_id", "developer_tools"]) - 1)
        .cast(pl.Float32).log1p().alias("developer_tools_log_count"),

        (col("browser_language").cum_count().over(["customer_id", "browser_language"]) - 1)
        .cast(pl.Float32).log1p().alias("browser_language_log_count"),

        (col("currency_iso_cd").cum_count().over(["customer_id", "currency_iso_cd"]) - 1)
        .cast(pl.Float32).log1p().alias("currency_iso_cd_log_count"),

        (col("phone_voip_call_state").cum_count().over(["customer_id", "phone_voip_call_state"]) - 1)
        .cast(pl.Float32).log1p().alias("phone_voip_call_state_log_count"),

        (col("web_rdp_connection").cum_count().over(["customer_id", "web_rdp_connection"]) - 1)
        .cast(pl.Float32).log1p().alias("web_rdp_connection_log_count"),
    ])

    # ── "Same as previous" flags for columns without existing lag features ────
    # For event_type_nm, event_desc, channel_type, channel_sub_type we already
    # store the actual prior value (prev_1_op_*) which is strictly more informative.
    # These two columns have no lag features at all yet:
    lf = lf.with_columns([
        (col("pos_cd") == col("pos_cd").shift(1).over("customer_id"))
        .fill_null(False).cast(pl.Int8).alias("pos_cd_prev"),

        (col("currency_iso_cd") == col("currency_iso_cd").shift(1).over("customer_id"))
        .fill_null(False).cast(pl.Int8).alias("currency_iso_cd_prev"),
    ])

    return lf
