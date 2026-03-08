import polars as pl
from polars import col, when


def add_device_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Section E: Device & Risk."""
    lf = lf.with_columns([
        when(col("compromised") == "true").then(1).otherwise(0).cast(pl.Int8).alias("compromised_flag"),
        col("web_rdp_connection").fill_null(0).cast(pl.Int8).alias("web_rdp_connection_flag"),
        when(col("developer_tools") == "true").then(1).otherwise(0).cast(pl.Int8).alias("developer_tools_flag"),
        col("phone_voip_call_state").fill_null(0).cast(pl.Int8).alias("phone_voip_call_flag"),
        when(
            col("battery")
            .str.extract(r"(\d+(?:\.\d+)?)\s*%", 1)
            .cast(pl.Float64)
            .fill_null(100)
            < 15
        ).then(1).otherwise(0).cast(pl.Int8).alias("low_battery_flag"),
    ])

    lf = lf.with_columns([
        (col("operating_system_type").cum_count().over(["customer_id", "operating_system_type"]) - 1 == 0).cast(pl.Int8).alias("operating_system_is_new"),
        (col("device_system_version").cum_count().over(["customer_id", "device_system_version"]) - 1 == 0).cast(pl.Int8).alias("os_version_is_new"),
        (col("screen_size").cum_count().over(["customer_id", "screen_size"]) - 1 == 0).cast(pl.Int8).alias("screen_size_is_new"),
        (col("timezone").cum_count().over(["customer_id", "timezone"]) - 1 == 0).cast(pl.Int8).alias("timezone_is_new"),
        (col("accept_language").cum_count().over(["customer_id", "accept_language"]) - 1 == 0).cast(pl.Int8).alias("accept_language_is_new"),
        (col("compromised_flag") * 5 + col("web_rdp_connection_flag") * 3 + col("developer_tools_flag") * 2).cast(pl.Float64).alias("device_risk_score"),
        (col("event_id").cum_count().over(["customer_id", "session_id"]) - 1).alias("session_tx_count"),
        (col("amount_clean").cum_sum().over(["customer_id", "session_id"]) - col("amount_clean")).fill_null(0).alias("session_amount_sum"),
        (col("channel_indicator_type") != col("channel_indicator_type").shift(1).over(["customer_id", "session_id"])).fill_null(False).cast(pl.Int8).alias("session_channel_diversity"),
        (col("event_id").cum_count().over(["customer_id", "session_id"]) - 1).alias("session_length_estimate"),
    ])

    return lf
