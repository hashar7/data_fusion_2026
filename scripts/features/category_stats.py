import polars as pl
from polars import col, when
from scripts.features._helpers import prior_mean_expr, prior_std_expr


def add_category_stats_features(
    lf: pl.LazyFrame,
    global_stats: dict[str, pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    """Section H: Per-Category Cumulative Stats and Global Z-Scores.

    Computes leakage-free per-user statistics for each key categorical dimension:
    event_desc, event_type_nm, channel_indicator_sub_type, pos_cd, tx_type_group.

    For each category, we compute:
    - Lifetime cumulative spend and tx count for (customer, category) pairs
    - Prior mean and std of amount for (customer, category)
    - Z-score of current amount vs user's prior history in this category
    - Share of lifetime spend / tx count that went to this category
    - "Last seen N days ago" — how long since the user last did this operation type
    - Cross-category novelty flags (first time user combines two categorical values)
    - Global (population-level) z-scores when global_stats are provided

    All features are strictly historical — they exclude the current transaction.
    """
    EPS = 1e-9

    # ── Helper: prior lifetime spend per customer ─────────────────────────────
    # Used as denominator for spend-share features.
    lf = lf.with_columns([
        (col("amount_clean").cum_sum().over("customer_id") - col("amount_clean"))
        .fill_null(0)
        .alias("_prior_lifetime_spend"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # A. Per-(customer, event_desc) cumulative stats
    #    Note: event_desc_user_freq (cum_count - 1) already exists from behavioral.py
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        # Prior cumulative spend for this (customer, event_desc) pair
        (col("amount_clean").cum_sum().over(["customer_id", "event_desc"]) - col("amount_clean"))
        .fill_null(0)
        .alias("event_desc_spend_user"),

        # Prior mean and std of amount for this (customer, event_desc)
        prior_mean_expr(["customer_id", "event_desc"]).alias("event_desc_amount_mean_user"),
        prior_std_expr(["customer_id", "event_desc"]).alias("event_desc_amount_std_user"),

        # Days since last transaction of this event_desc for this customer
        (
            (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "event_desc"]))
            .dt.total_days()
        ).fill_null(999).alias("event_desc_last_seen_days"),
    ])

    lf = lf.with_columns([
        # Z-score: is this amount unusual for this event_desc type for this user?
        (
            (col("amount_clean") - col("event_desc_amount_mean_user")) /
            (col("event_desc_amount_std_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_event_desc"),

        # Share: what fraction of prior lifetime spend went to this event_desc?
        (col("event_desc_spend_user") / (col("_prior_lifetime_spend") + EPS))
        .fill_null(0)
        .alias("event_desc_spend_share_user"),

        # Approximate: lifetime spend in this event_desc vs recent 90d spend
        # Captures "how dominant is this event_desc in recent history"
        (col("event_desc_spend_user") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("event_desc_spend_vs_90d"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # B. Per-(customer, event_type_nm) cumulative stats
    #    Note: event_type_user_freq (cum_count - 1) already exists from behavioral.py
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        (col("amount_clean").cum_sum().over(["customer_id", "event_type_nm"]) - col("amount_clean"))
        .fill_null(0)
        .alias("event_type_spend_user"),

        prior_mean_expr(["customer_id", "event_type_nm"]).alias("event_type_amount_mean_user"),
        prior_std_expr(["customer_id", "event_type_nm"]).alias("event_type_amount_std_user"),

        (
            (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "event_type_nm"]))
            .dt.total_days()
        ).fill_null(999).alias("event_type_last_seen_days"),
    ])

    lf = lf.with_columns([
        (
            (col("amount_clean") - col("event_type_amount_mean_user")) /
            (col("event_type_amount_std_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_event_type"),

        (col("event_type_spend_user") / (col("_prior_lifetime_spend") + EPS))
        .fill_null(0)
        .alias("event_type_spend_share_user"),

        (col("event_type_spend_user") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("event_type_spend_vs_90d"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # C. Per-(customer, channel_indicator_sub_type) cumulative stats
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        (col("amount_clean").cum_sum().over(["customer_id", "channel_indicator_sub_type"]) - col("amount_clean"))
        .fill_null(0)
        .alias("spend_in_subchannel_lifetime"),

        (col("event_id").cum_count().over(["customer_id", "channel_indicator_sub_type"]) - 1)
        .alias("tx_count_in_subchannel_lifetime"),

        prior_mean_expr(["customer_id", "channel_indicator_sub_type"]).alias("amount_mean_subchannel_user"),
        prior_std_expr(["customer_id", "channel_indicator_sub_type"]).alias("amount_std_subchannel_user"),

        (
            (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "channel_indicator_sub_type"]))
            .dt.total_days()
        ).fill_null(999).alias("subchannel_last_seen_days"),
    ])

    lf = lf.with_columns([
        (col("tx_count_in_subchannel_lifetime") / col("tx_count_lifetime").clip(lower_bound=1))
        .fill_null(0)
        .alias("subchannel_usage_share"),

        (col("tx_count_in_subchannel_lifetime") == 0)
        .cast(pl.Int8)
        .alias("is_new_subchannel_for_user"),

        (
            (col("amount_clean") - col("amount_mean_subchannel_user")) /
            (col("amount_std_subchannel_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_subchannel"),

        (col("spend_in_subchannel_lifetime") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("subchannel_spend_vs_90d"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # D. Per-(customer, pos_cd) cumulative stats
    #    Note: pos_freq_user_cum (cum_count - 1) and pos_cd_transaction_share_user
    #    already exist in behavioral.py; we add the spend-side stats only.
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        (col("amount_clean").cum_sum().over(["customer_id", "pos_cd"]) - col("amount_clean"))
        .fill_null(0)
        .alias("spend_in_pos_lifetime"),

        prior_mean_expr(["customer_id", "pos_cd"]).alias("amount_mean_pos_user"),
        prior_std_expr(["customer_id", "pos_cd"]).alias("amount_std_pos_user"),

        (
            (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "pos_cd"]))
            .dt.total_days()
        ).fill_null(999).alias("pos_last_seen_days"),
    ])

    lf = lf.with_columns([
        (
            (col("amount_clean") - col("amount_mean_pos_user")) /
            (col("amount_std_pos_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_pos"),

        (col("spend_in_pos_lifetime") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("pos_spend_vs_90d"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # E. Per-(customer, tx_type_group) cumulative stats
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        (col("amount_clean").cum_sum().over(["customer_id", "tx_type_group"]) - col("amount_clean"))
        .fill_null(0)
        .alias("spend_in_tx_type_lifetime"),

        (col("event_id").cum_count().over(["customer_id", "tx_type_group"]) - 1)
        .alias("tx_count_in_tx_type_lifetime"),

        prior_mean_expr(["customer_id", "tx_type_group"]).alias("amount_mean_tx_type_user"),
        prior_std_expr(["customer_id", "tx_type_group"]).alias("amount_std_tx_type_user"),
    ])

    lf = lf.with_columns([
        (col("tx_count_in_tx_type_lifetime") / col("tx_count_lifetime").clip(lower_bound=1))
        .fill_null(0)
        .alias("tx_type_usage_share"),

        (
            (col("amount_clean") - col("amount_mean_tx_type_user")) /
            (col("amount_std_tx_type_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_tx_type"),

        (col("spend_in_tx_type_lifetime") / (col("_prior_lifetime_spend") + EPS))
        .fill_null(0)
        .alias("tx_type_spend_share"),

        (col("spend_in_tx_type_lifetime") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("tx_type_spend_vs_90d"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # F. Cross-category novelty flags
    #    First time a customer uses a specific combination of two categorical values.
    #    These capture unusual compound contexts that single-field novelty flags miss.
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        # Is (channel_type, event_desc) combination new for this user?
        ((col("event_id").cum_count().over(["customer_id", "channel_indicator_type", "event_desc"]) - 1) == 0)
        .cast(pl.Int8)
        .alias("is_new_channel_desc_combo"),

        # Is (channel_type, event_type_nm) combination new for this user?
        ((col("event_id").cum_count().over(["customer_id", "channel_indicator_type", "event_type_nm"]) - 1) == 0)
        .cast(pl.Int8)
        .alias("is_new_channel_type_combo"),

        # Is (channel_sub_type, event_type_nm) combination new for this user?
        ((col("event_id").cum_count().over(["customer_id", "channel_indicator_sub_type", "event_type_nm"]) - 1) == 0)
        .cast(pl.Int8)
        .alias("is_new_subchannel_type_combo"),

        # Is (event_type_nm, event_desc) combination new for this user?
        ((col("event_id").cum_count().over(["customer_id", "event_type_nm", "event_desc"]) - 1) == 0)
        .cast(pl.Int8)
        .alias("is_new_type_desc_combo"),

        # Is (tx_type_group, channel_type) combination new for this user?
        # E.g., card payment through a channel usually used for non-payments.
        ((col("event_id").cum_count().over(["customer_id", "tx_type_group", "channel_indicator_type"]) - 1) == 0)
        .cast(pl.Int8)
        .alias("is_new_txtype_channel_combo"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # G. Global (population-level) z-scores
    #    When global_stats are provided, compute how unusual this transaction's
    #    amount is relative to the population baseline for its category.
    # ══════════════════════════════════════════════════════════════════════════
    def _to_lazy(df_or_lf):
        return df_or_lf.lazy() if isinstance(df_or_lf, pl.DataFrame) else df_or_lf

    if global_stats:
        if "event_desc_stats_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["event_desc_stats_global"]), on="event_desc", how="left")
            lf = lf.with_columns([
                (
                    (col("amount_clean") - col("event_desc_global_mean").fill_null(0)) /
                    (col("event_desc_global_std").fill_null(1) + EPS)
                ).alias("amount_zscore_event_desc_global"),
            ])
            lf = lf.drop(["event_desc_global_mean", "event_desc_global_std"])

        if "event_type_stats_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["event_type_stats_global"]), on="event_type_nm", how="left")
            lf = lf.with_columns([
                (
                    (col("amount_clean") - col("event_type_global_mean").fill_null(0)) /
                    (col("event_type_global_std").fill_null(1) + EPS)
                ).alias("amount_zscore_event_type_global"),
            ])
            lf = lf.drop(["event_type_global_mean", "event_type_global_std"])

        if "subchannel_stats_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["subchannel_stats_global"]), on="channel_indicator_sub_type", how="left")
            lf = lf.with_columns([
                (
                    (col("amount_clean") - col("subchannel_global_mean").fill_null(0)) /
                    (col("subchannel_global_std").fill_null(1) + EPS)
                ).alias("amount_zscore_subchannel_global"),
            ])
            lf = lf.drop(["subchannel_global_mean", "subchannel_global_std"])

        if "pos_stats_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["pos_stats_global"]), on="pos_cd", how="left")
            lf = lf.with_columns([
                (
                    (col("amount_clean") - col("pos_global_mean").fill_null(0)) /
                    (col("pos_global_std").fill_null(1) + EPS)
                ).alias("amount_zscore_pos_global"),
            ])
            lf = lf.drop(["pos_global_mean", "pos_global_std"])

        if "subchannel_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["subchannel_global"]), on="channel_indicator_sub_type", how="left")
            lf = lf.with_columns([
                col("global_subchannel_freq").fill_null(0),
            ])
    else:
        # Fallback when no global_stats: constant sentinel so schema is stable
        lf = lf.with_columns([
            pl.lit(0.0).alias("amount_zscore_event_desc_global"),
            pl.lit(0.0).alias("amount_zscore_event_type_global"),
            pl.lit(0.0).alias("amount_zscore_subchannel_global"),
            pl.lit(0.0).alias("amount_zscore_pos_global"),
            pl.lit(0).cast(pl.Int32).alias("global_subchannel_freq"),
        ])

    # Drop intermediate helper columns
    lf = lf.drop(["_prior_lifetime_spend"])

    return lf
