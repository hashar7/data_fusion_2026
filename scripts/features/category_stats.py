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

        # Log count complement to the linear tx_count_in_subchannel_lifetime.
        # Competitor's channel_indicator_sub_type_log_count ranks 22nd (1.27% gain).
        col("tx_count_in_subchannel_lifetime").cast(pl.Float32).log1p()
        .alias("channel_indicator_sub_type_log_count"),

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
    # C2. Per-(customer, channel_type_subtype) cumulative stats
    #     Combined channel type + subtype captures type×subtype interactions.
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        prior_mean_expr(["customer_id", "channel_type_subtype"]).alias("amount_mean_channel_type_subtype_user"),
        prior_std_expr(["customer_id", "channel_type_subtype"]).alias("amount_std_channel_type_subtype_user"),

        (
            (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "channel_type_subtype"]))
            .dt.total_days()
        ).fill_null(999).alias("channel_type_subtype_last_seen_days"),

        (col("event_id").cum_count().over(["customer_id", "channel_type_subtype"]) - 1 == 0)
        .cast(pl.Int8)
        .alias("is_new_channel_type_subtype_for_user"),
    ])

    lf = lf.with_columns([
        (
            (col("amount_clean") - col("amount_mean_channel_type_subtype_user")) /
            (col("amount_std_channel_type_subtype_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_channel_type_subtype"),

        (col("spend_in_channel_type_subtype_lifetime") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("channel_type_subtype_spend_vs_90d"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # C3. Per-(customer, evtype_channel) cumulative stats
    #     Combined event_type_nm × channel_indicator_type interactions.
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        prior_mean_expr(["customer_id", "evtype_channel"]).alias("amount_mean_evtype_channel_user"),
        prior_std_expr(["customer_id", "evtype_channel"]).alias("amount_std_evtype_channel_user"),

        (
            (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "evtype_channel"]))
            .dt.total_days()
        ).fill_null(999).alias("evtype_channel_last_seen_days"),

        (col("event_id").cum_count().over(["customer_id", "evtype_channel"]) - 1 == 0)
        .cast(pl.Int8)
        .alias("is_new_evtype_channel_for_user"),
    ])

    lf = lf.with_columns([
        (
            (col("amount_clean") - col("amount_mean_evtype_channel_user")) /
            (col("amount_std_evtype_channel_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_evtype_channel"),

        (col("spend_in_evtype_channel_lifetime") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("evtype_channel_spend_vs_90d"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # C4. Per-(customer, evtype_subchannel) cumulative stats
    #     Combined event_type_nm × channel_indicator_sub_type interactions.
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        prior_mean_expr(["customer_id", "evtype_subchannel"]).alias("amount_mean_evtype_subchannel_user"),
        prior_std_expr(["customer_id", "evtype_subchannel"]).alias("amount_std_evtype_subchannel_user"),

        (
            (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "evtype_subchannel"]))
            .dt.total_days()
        ).fill_null(999).alias("evtype_subchannel_last_seen_days"),

        (col("event_id").cum_count().over(["customer_id", "evtype_subchannel"]) - 1 == 0)
        .cast(pl.Int8)
        .alias("is_new_evtype_subchannel_for_user"),
    ])

    lf = lf.with_columns([
        (
            (col("amount_clean") - col("amount_mean_evtype_subchannel_user")) /
            (col("amount_std_evtype_subchannel_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_evtype_subchannel"),

        (col("spend_in_evtype_subchannel_lifetime") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("evtype_subchannel_spend_vs_90d"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # C5. Per-(customer, event_type_nm, mcc_code) cumulative stats
    #     Two-column .over() since mcc_code is String type.
    # ══════════════════════════════════════════════════════════════════════════
    lf = lf.with_columns([
        prior_mean_expr(["customer_id", "event_type_nm", "mcc_code"]).alias("amount_mean_evtype_mcc_user"),
        prior_std_expr(["customer_id", "event_type_nm", "mcc_code"]).alias("amount_std_evtype_mcc_user"),

        (
            (col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "event_type_nm", "mcc_code"]))
            .dt.total_days()
        ).fill_null(999).alias("evtype_mcc_last_seen_days"),

        (col("event_id").cum_count().over(["customer_id", "event_type_nm", "mcc_code"]) - 1 == 0)
        .cast(pl.Int8)
        .alias("is_new_evtype_mcc_for_user"),
    ])

    lf = lf.with_columns([
        (
            (col("amount_clean") - col("amount_mean_evtype_mcc_user")) /
            (col("amount_std_evtype_mcc_user").fill_null(1) + EPS)
        ).alias("amount_zscore_given_evtype_mcc"),

        (col("spend_in_evtype_mcc_lifetime") / (col("cumulative_spend_90d").fill_null(0) + EPS))
        .fill_null(0)
        .alias("evtype_mcc_spend_vs_90d"),
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

        if "channel_type_subtype_stats_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["channel_type_subtype_stats_global"]), on="channel_type_subtype", how="left")
            lf = lf.with_columns([
                (
                    (col("amount_clean") - col("channel_type_subtype_global_mean").fill_null(0)) /
                    (col("channel_type_subtype_global_std").fill_null(1) + EPS)
                ).alias("amount_zscore_channel_type_subtype_global"),
            ])
            lf = lf.drop(["channel_type_subtype_global_mean", "channel_type_subtype_global_std"])

        if "channel_type_subtype_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["channel_type_subtype_global"]), on="channel_type_subtype", how="left")
            lf = lf.with_columns([
                col("global_channel_type_subtype_freq").fill_null(0),
            ])

        if "evtype_channel_stats_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["evtype_channel_stats_global"]), on="evtype_channel", how="left")
            lf = lf.with_columns([
                (
                    (col("amount_clean") - col("evtype_channel_global_mean").fill_null(0)) /
                    (col("evtype_channel_global_std").fill_null(1) + EPS)
                ).alias("amount_zscore_evtype_channel_global"),
            ])
            lf = lf.drop(["evtype_channel_global_mean", "evtype_channel_global_std"])

        if "evtype_channel_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["evtype_channel_global"]), on="evtype_channel", how="left")
            lf = lf.with_columns([
                col("global_evtype_channel_freq").fill_null(0),
            ])

        if "evtype_subchannel_stats_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["evtype_subchannel_stats_global"]), on="evtype_subchannel", how="left")
            lf = lf.with_columns([
                (
                    (col("amount_clean") - col("evtype_subchannel_global_mean").fill_null(0)) /
                    (col("evtype_subchannel_global_std").fill_null(1) + EPS)
                ).alias("amount_zscore_evtype_subchannel_global"),
            ])
            lf = lf.drop(["evtype_subchannel_global_mean", "evtype_subchannel_global_std"])

        if "evtype_subchannel_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["evtype_subchannel_global"]), on="evtype_subchannel", how="left")
            lf = lf.with_columns([
                col("global_evtype_subchannel_freq").fill_null(0),
            ])

        if "evtype_mcc_stats_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["evtype_mcc_stats_global"]), on=["event_type_nm", "mcc_code"], how="left")
            lf = lf.with_columns([
                (
                    (col("amount_clean") - col("evtype_mcc_global_mean").fill_null(0)) /
                    (col("evtype_mcc_global_std").fill_null(1) + EPS)
                ).alias("amount_zscore_evtype_mcc_global"),
            ])
            lf = lf.drop(["evtype_mcc_global_mean", "evtype_mcc_global_std"])

        if "evtype_mcc_global" in global_stats:
            lf = lf.join(_to_lazy(global_stats["evtype_mcc_global"]), on=["event_type_nm", "mcc_code"], how="left")
            lf = lf.with_columns([
                col("global_evtype_mcc_freq").fill_null(0),
            ])
    else:
        # Fallback when no global_stats: constant sentinel so schema is stable
        lf = lf.with_columns([
            pl.lit(0.0).alias("amount_zscore_event_desc_global"),
            pl.lit(0.0).alias("amount_zscore_event_type_global"),
            pl.lit(0.0).alias("amount_zscore_subchannel_global"),
            pl.lit(0.0).alias("amount_zscore_pos_global"),
            pl.lit(0).cast(pl.Int32).alias("global_subchannel_freq"),
            pl.lit(0.0).alias("amount_zscore_channel_type_subtype_global"),
            pl.lit(0).cast(pl.Int32).alias("global_channel_type_subtype_freq"),
            pl.lit(0.0).alias("amount_zscore_evtype_channel_global"),
            pl.lit(0).cast(pl.Int32).alias("global_evtype_channel_freq"),
            pl.lit(0.0).alias("amount_zscore_evtype_subchannel_global"),
            pl.lit(0).cast(pl.Int32).alias("global_evtype_subchannel_freq"),
            pl.lit(0.0).alias("amount_zscore_evtype_mcc_global"),
            pl.lit(0).cast(pl.Int32).alias("global_evtype_mcc_freq"),
        ])

    # Drop intermediate helper columns
    lf = lf.drop(["_prior_lifetime_spend"])

    # ══════════════════════════════════════════════════════════════════════════
    # I. Bayesian-smoothed target encodings
    #    Precomputed on the labeled training set in global_stats.py.
    #    Each feature is the smoothed fraud rate for that category value.
    #    Unseen values at test time are filled with the stored global rate.
    # ══════════════════════════════════════════════════════════════════════════
    _SINGLE_TE = [
        # (stats key,                join column,                  feature name)
        ("evtype_target_enc",        "event_type_nm",              "event_type_nm_target_enc"),
        ("evdesc_target_enc",        "event_desc",                 "event_desc_target_enc"),
        ("channel_type_target_enc",  "channel_indicator_type",     "channel_type_target_enc"),
        ("channel_subtype_target_enc","channel_indicator_sub_type","channel_subtype_target_enc"),
        # MCC fraud rate — strong signal for card transactions; fills with global rate for non-card (null mcc_code)
        ("mcc_target_enc",           "mcc_code",                   "mcc_target_enc"),
        # Combined channel type × subtype target encoding
        ("channel_type_subtype_target_enc", "channel_type_subtype", "channel_type_subtype_target_enc"),
        # Combined event_type × channel / subchannel target encodings
        ("evtype_channel_target_enc", "evtype_channel", "evtype_channel_target_enc"),
        ("evtype_subchannel_target_enc", "evtype_subchannel", "evtype_subchannel_target_enc"),
    ]

    if global_stats:
        # Retrieve the global fraud rate (fallback for unseen category values).
        _fallback = 0.5  # neutral prior when global_fraud_rate key is missing
        if "global_fraud_rate" in global_stats:
            _gfr_df = _to_lazy(global_stats["global_fraud_rate"]).collect()
            _fallback = float(_gfr_df["global_fraud_rate"][0])

        for key, join_col, feat_name in _SINGLE_TE:
            if key in global_stats:
                lf = lf.join(
                    _to_lazy(global_stats[key]),
                    on=join_col,
                    how="left",
                ).with_columns(
                    pl.col(feat_name).fill_null(_fallback)
                )
            else:
                lf = lf.with_columns(pl.lit(_fallback).alias(feat_name))

        if "type_desc_target_enc" in global_stats:
            lf = lf.join(
                _to_lazy(global_stats["type_desc_target_enc"]),
                on=["event_type_nm", "event_desc"],
                how="left",
            ).with_columns(
                pl.col("type_desc_pair_target_enc").fill_null(_fallback)
            )
        else:
            lf = lf.with_columns(pl.lit(_fallback).alias("type_desc_pair_target_enc"))

        # Within-group channel encodings — two-column join key
        _WITHIN_GROUP_TE = [
            # (stats key,                    join columns,                                          feature name)
            ("channel_type_within_group_te",    ["tx_type_group", "channel_indicator_type"],     "channel_type_fraud_rate_within_group"),
            ("channel_subtype_within_group_te", ["tx_type_group", "channel_indicator_sub_type"], "channel_subtype_fraud_rate_within_group"),
        ]
        for key, join_cols, feat_name in _WITHIN_GROUP_TE:
            if key in global_stats:
                lf = lf.join(
                    _to_lazy(global_stats[key]),
                    on=join_cols,
                    how="left",
                ).with_columns(
                    pl.col(feat_name).fill_null(_fallback)
                )
            else:
                lf = lf.with_columns(pl.lit(_fallback).alias(feat_name))

        # evtype_mcc pair target encoding — two-column join (mcc_code is String)
        if "evtype_mcc_target_enc" in global_stats:
            lf = lf.join(
                _to_lazy(global_stats["evtype_mcc_target_enc"]),
                on=["event_type_nm", "mcc_code"],
                how="left",
            ).with_columns(
                pl.col("evtype_mcc_target_enc").fill_null(_fallback)
            )
        else:
            lf = lf.with_columns(pl.lit(_fallback).alias("evtype_mcc_target_enc"))
    else:
        # No global_stats at all: emit neutral constant so the schema is stable
        for _, _, feat_name in _SINGLE_TE:
            lf = lf.with_columns(pl.lit(0.5).alias(feat_name))
        lf = lf.with_columns(pl.lit(0.5).alias("type_desc_pair_target_enc"))
        lf = lf.with_columns(pl.lit(0.5).alias("channel_type_fraud_rate_within_group"))
        lf = lf.with_columns(pl.lit(0.5).alias("channel_subtype_fraud_rate_within_group"))
        lf = lf.with_columns(pl.lit(0.5).alias("mcc_target_enc"))
        lf = lf.with_columns(pl.lit(0.5).alias("channel_type_subtype_target_enc"))
        lf = lf.with_columns(pl.lit(0.5).alias("evtype_mcc_target_enc"))

    return lf
