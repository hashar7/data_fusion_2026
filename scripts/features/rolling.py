import polars as pl
from polars import col, when


# Time windows for rolling statistics.
_WINDOWS = [
    ("15m", "15m"),
    ("1h",  "1h"),
    ("6h",  "6h"),
    ("12h", "12h"),
    ("1d",  "1d"),
    ("3d",  "3d"),
    ("7d",  "7d"),
    ("30d", "30d"),
    ("90d", "90d"),
]


def _inline_rolling_exprs(window_size: str, suffix: str) -> list[pl.Expr]:
    """Inline rolling_*_by() expressions for one window — no join required.

    Uses the Polars 1.x rolling_*_by() API with .over("customer_id") to
    compute per-customer rolling aggregations directly as column expressions.
    This avoids creating separate LazyFrames and joining them back.
    """
    kw = dict(window_size=window_size, closed="left", min_periods=1)
    by = "event_dttm"
    grp = "customer_id"
    return [
        col("amount_clean").rolling_mean_by(by, **kw).over(grp)
            .alias(f"amount_mean_{suffix}"),
        col("amount_clean").rolling_std_by(by, **kw).over(grp)
            .alias(f"amount_std_{suffix}"),
        col("amount_clean").rolling_median_by(by, **kw).over(grp)
            .alias(f"amount_median_{suffix}"),
        col("amount_clean").rolling_max_by(by, **kw).over(grp)
            .alias(f"amount_max_{suffix}"),
        col("amount_clean").rolling_min_by(by, **kw).over(grp)
            .alias(f"amount_min_{suffix}"),
        col("amount_clean").rolling_sum_by(by, **kw).over(grp)
            .fill_null(0.0).alias(f"cumulative_spend_{suffix}"),
        col("amount_card").rolling_sum_by(by, **kw).over(grp)
            .fill_null(0.0).alias(f"card_spend_{suffix}"),
        col("amount_p2p").rolling_sum_by(by, **kw).over(grp)
            .fill_null(0.0).alias(f"p2p_spend_{suffix}"),
        # tx_count: rolling count via sum-of-ones (no rolling_count_by in Polars)
        col("event_dttm").is_not_null().cast(pl.Int32)
            .rolling_sum_by(by, **kw).over(grp)
            .fill_null(0).alias(f"tx_count_{suffix}"),
    ]


def _build_nunique_table(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute n_unique (diversity) features via grouped .rolling().

    n_unique has no rolling_*_by() equivalent, so we use the grouped API.
    All 9 windows are merged into one wide table keyed by temp_row_idx,
    producing a single join to the main LazyFrame (instead of 9 joins).
    """
    merged = None
    for window_size, suffix in _WINDOWS:
        feat_names = [
            f"channel_diversity_{suffix}",
            f"device_diversity_{suffix}",
            f"merchant_diversity_{suffix}",
            f"event_desc_diversity_{suffix}",
            f"event_type_diversity_{suffix}",
        ]
        r = (
            lf.rolling(
                index_column="event_dttm",
                period=window_size,
                by="customer_id",
                closed="left",
            )
            .agg([
                col("channel_indicator_type").n_unique().alias(feat_names[0]),
                col("operating_system_type").n_unique().alias(feat_names[1]),
                col("mcc_code").n_unique().alias(feat_names[2]),
                col("event_desc").n_unique().alias(feat_names[3]),
                col("event_type_nm").n_unique().alias(feat_names[4]),
            ])
            .with_row_index("temp_row_idx")
            .select(["temp_row_idx"] + feat_names)
        )
        if merged is None:
            merged = r
        else:
            merged = merged.join(r, on="temp_row_idx", how="left")
    return merged


def add_rolling_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Section D: Rolling Statistics (optimized inline approach).

    Simple aggregations (sum, mean, std, median, min, max) use inline
    rolling_*_by() expressions — no intermediate LazyFrames or joins.
    Only n_unique (diversity) features use the grouped .rolling() path
    because Polars has no rolling_n_unique_by(). All n_unique results
    are merged into a single join (was 9 separate joins before).
    """
    # ── 1. n_unique (diversity) features via grouped rolling ─────────────
    # Compute first from the simpler LF (before inline columns are added).
    nq_table = _build_nunique_table(lf)
    lf = lf.join(nq_table, on="temp_row_idx", how="left")

    # ── 2. Inline rolling aggregations for all 9 windows ────────────────
    # All sum/mean/std/median/min/max use rolling_*_by() + .over().
    all_rolling_exprs = []
    for window_size, suffix in _WINDOWS:
        all_rolling_exprs.extend(_inline_rolling_exprs(window_size, suffix))
    lf = lf.with_columns(all_rolling_exprs)

    EPS = 1e-9

    # ── Sub-day derived features ──────────────────────────────────────────
    lf = lf.with_columns([
        (col("tx_count_1h")  > 5).cast(pl.Int8).alias("burst_flag_1h"),

        # Spend ratios: fraction of daily spend in short sub-windows.
        (col("cumulative_spend_15m") / (col("cumulative_spend_1d").fill_null(0) + EPS)).alias("spend_ratio_15m_vs_1d"),
        (col("cumulative_spend_1h")  / (col("cumulative_spend_1d").fill_null(0) + EPS)).alias("spend_ratio_1h_vs_1d"),
        (col("cumulative_spend_6h")  / (col("cumulative_spend_1d").fill_null(0) + EPS)).alias("spend_ratio_6h_vs_1d"),
        (col("cumulative_spend_12h") / (col("cumulative_spend_1d").fill_null(0) + EPS)).alias("spend_ratio_12h_vs_1d"),

        # TX-count ratios: concentration of activity in a short sub-window.
        (col("tx_count_15m") / (col("tx_count_1h").fill_null(0) + EPS)).alias("tx_count_ratio_15m_vs_1h"),
        (col("tx_count_1h")  / (col("tx_count_1d").fill_null(0) + EPS)).alias("tx_count_ratio_1h_vs_1d"),

        # Amount vs. short-window mean.
        (col("amount_clean") / (col("amount_mean_1h").fill_null(1) + EPS)).alias("amount_ratio_to_mean_1h"),
        (col("amount_clean") / (col("amount_mean_6h").fill_null(1) + EPS)).alias("amount_ratio_to_mean_6h"),
    ])

    # ── Day+ derived features ─────────────────────────────────────────────
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
        (col("cumulative_spend_1d") / (col("cumulative_spend_30d").fill_null(0) + 1e-9)).alias("spend_ratio_1d_vs_30d"),
        (col("cumulative_spend_1d") / (col("cumulative_spend_90d").fill_null(0) + 1e-9)).alias("spend_ratio_1d_vs_90d"),
        (col("cumulative_spend_7d") / (col("cumulative_spend_90d").fill_null(0) + 1e-9)).alias("spend_ratio_7d_vs_90d"),
        (col("cumulative_spend_7d")  / 7.0 ).alias("spend_velocity_7d"),
        (col("cumulative_spend_30d") / 30.0).alias("spend_velocity_30d"),
    ])

    lf = lf.with_columns([
        (col("amount_rank_percentile_30d") > 0.95).cast(pl.Int8).alias("amount_top5pct_30d"),
        (col("amount_rank_percentile_90d") > 0.99).cast(pl.Int8).alias("amount_top1pct_90d"),
        (col("amount_clean") > col("amount_max_90d").fill_null(0)).cast(pl.Int8).alias("amount_above_personal_max_flag"),
    ])

    # Per-channel cumulative stats (leakage-free: prior rows only, same -1 shift).
    lf = lf.with_columns([
        (col("amount_clean").cum_sum().over(["customer_id", "channel_indicator_type"]) - col("amount_clean")).fill_null(0).alias("spend_in_channel_lifetime"),
        (col("event_id").cum_count().over(["customer_id", "channel_indicator_type"]) - 1).alias("tx_count_in_channel_lifetime"),
        (col("amount_clean").cum_sum().over(["customer_id", "channel_type_subtype"]) - col("amount_clean")).fill_null(0).alias("spend_in_channel_type_subtype_lifetime"),
        (col("event_id").cum_count().over(["customer_id", "channel_type_subtype"]) - 1).alias("tx_count_in_channel_type_subtype_lifetime"),
        (col("amount_clean").cum_sum().over(["customer_id", "evtype_channel"]) - col("amount_clean")).fill_null(0).alias("spend_in_evtype_channel_lifetime"),
        (col("event_id").cum_count().over(["customer_id", "evtype_channel"]) - 1).alias("tx_count_in_evtype_channel_lifetime"),
        (col("amount_clean").cum_sum().over(["customer_id", "evtype_subchannel"]) - col("amount_clean")).fill_null(0).alias("spend_in_evtype_subchannel_lifetime"),
        (col("event_id").cum_count().over(["customer_id", "evtype_subchannel"]) - 1).alias("tx_count_in_evtype_subchannel_lifetime"),
        (col("amount_clean").cum_sum().over(["customer_id", "event_type_nm", "mcc_code"]) - col("amount_clean")).fill_null(0).alias("spend_in_evtype_mcc_lifetime"),
        (col("event_id").cum_count().over(["customer_id", "event_type_nm", "mcc_code"]) - 1).alias("tx_count_in_evtype_mcc_lifetime"),
    ])
    lf = lf.with_columns([
        (col("tx_count_in_channel_lifetime") / col("tx_count_lifetime").clip(lower_bound=1)).fill_null(0).alias("channel_usage_share"),
        (col("tx_count_in_channel_type_subtype_lifetime") / col("tx_count_lifetime").clip(lower_bound=1)).fill_null(0).alias("channel_type_subtype_usage_share"),
        (col("tx_count_in_evtype_channel_lifetime") / col("tx_count_lifetime").clip(lower_bound=1)).fill_null(0).alias("evtype_channel_usage_share"),
        (col("tx_count_in_evtype_subchannel_lifetime") / col("tx_count_lifetime").clip(lower_bound=1)).fill_null(0).alias("evtype_subchannel_usage_share"),
        (col("tx_count_in_evtype_mcc_lifetime") / col("tx_count_lifetime").clip(lower_bound=1)).fill_null(0).alias("evtype_mcc_usage_share"),
    ])

    EPS_CARD = 1e-9

    # ── Card vs P2P spend fractions and cross-window ratios ──────────────
    lf = lf.with_columns([
        (col("card_spend_1d")  / (col("cumulative_spend_1d").fill_null(0)  + EPS_CARD)).alias("card_fraction_1d"),
        (col("p2p_spend_1d")   / (col("cumulative_spend_1d").fill_null(0)  + EPS_CARD)).alias("p2p_fraction_1d"),
        (col("card_spend_7d")  / (col("cumulative_spend_7d").fill_null(0)  + EPS_CARD)).alias("card_fraction_7d"),
        (col("card_spend_30d") / (col("cumulative_spend_30d").fill_null(0) + EPS_CARD)).alias("card_fraction_30d"),
        (col("p2p_spend_30d")  / (col("cumulative_spend_30d").fill_null(0) + EPS_CARD)).alias("p2p_fraction_30d"),
        (col("card_spend_90d") / (col("cumulative_spend_90d").fill_null(0) + EPS_CARD)).alias("card_fraction_90d"),
        (col("p2p_spend_90d")  / (col("cumulative_spend_90d").fill_null(0) + EPS_CARD)).alias("p2p_fraction_90d"),

        # Cross-window ratios per payment type
        (col("card_spend_1d") / (col("card_spend_30d").fill_null(0) + EPS_CARD)).alias("card_spend_ratio_1d_vs_30d"),
        (col("p2p_spend_1d")  / (col("p2p_spend_30d").fill_null(0)  + EPS_CARD)).alias("p2p_spend_ratio_1d_vs_30d"),
        (col("card_spend_1d") / (col("card_spend_90d").fill_null(0) + EPS_CARD)).alias("card_spend_ratio_1d_vs_90d"),
        (col("p2p_spend_1d")  / (col("p2p_spend_90d").fill_null(0)  + EPS_CARD)).alias("p2p_spend_ratio_1d_vs_90d"),
        (col("card_spend_7d") / (col("card_spend_90d").fill_null(0) + EPS_CARD)).alias("card_spend_ratio_7d_vs_90d"),
        (col("p2p_spend_7d")  / (col("p2p_spend_90d").fill_null(0)  + EPS_CARD)).alias("p2p_spend_ratio_7d_vs_90d"),

        # Card-to-P2P balance ratio
        (col("card_spend_1d")  / (col("p2p_spend_1d").fill_null(0)  + EPS_CARD)).alias("card_vs_p2p_ratio_1d"),
        (col("card_spend_30d") / (col("p2p_spend_30d").fill_null(0) + EPS_CARD)).alias("card_vs_p2p_ratio_30d"),
        (col("card_spend_90d") / (col("p2p_spend_90d").fill_null(0) + EPS_CARD)).alias("card_vs_p2p_ratio_90d"),
    ])

    return lf
