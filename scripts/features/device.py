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
        # MCC switch within a session: jumping merchants in one session is suspicious
        (col("mcc_code") != col("mcc_code").shift(1).over(["customer_id", "session_id"])).fill_null(False).cast(pl.Int8).alias("session_mcc_switch"),
        # Session duration: minutes elapsed since the first transaction in this session.
        # cum_min on sorted data equals the session-start timestamp — leakage-free.
        (col("event_dttm") - col("event_dttm").cum_min().over(["customer_id", "session_id"])).dt.total_minutes().fill_null(0).alias("session_duration_minutes"),
        # Gap in seconds between consecutive transactions in the SAME session.
        # Captures burst behavior within a session that session_duration_minutes misses.
        # Competitor's pause_ses — different from pause_cus (customer-level gap).
        (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "session_id"])).dt.total_seconds().fill_null(0).alias("pause_ses"),
    ])

    # Screen dimensions parsed from "WxH" string (e.g. "1080x1920").
    # screen_size_is_new is a binary first-occurrence flag; the actual dimensions
    # carry a different signal (unusual screen resolution for this user/device type).
    lf = lf.with_columns([
        col("screen_size").str.extract(r"^(\d+)", 1).cast(pl.Int32).fill_null(0).alias("screen_w"),
        col("screen_size").str.extract(r"x(\d+)", 1).cast(pl.Int32).fill_null(0).alias("screen_h"),
    ])

    # Features that depend on session_tx_count / session_amount_sum computed above.
    # Continuous device×amount interactions replace the weak binary composite flags:
    # binary flags (new_device_and_night etc.) don't appear in top-60; continuous
    # products give the model a smooth signal to split on.
    lf = lf.with_columns([
        # Average spend per transaction so far in this session
        (col("session_amount_sum") / col("session_tx_count").clip(lower_bound=1)).fill_null(0).alias("session_avg_amount"),
        # RDP session depth: remote-controlled device × how deep into the session we are
        (col("web_rdp_connection_flag").cast(pl.Float64) * col("session_tx_count").cast(pl.Float64)).alias("rdp_x_session_depth"),
    ])

    return lf
