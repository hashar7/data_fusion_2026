import polars as pl
from polars import col

# Bayesian smoothing strength (pseudo-observation count).
# With ~300 labeled rows per event_desc value on average, alpha=20 pulls
# rare-value estimates ~6 % toward the global mean — conservative and safe.
_ALPHA = 20


def compute_global_stats(
    train_lf: pl.LazyFrame,
    labels_lf: pl.LazyFrame | None = None,
) -> dict[str, pl.LazyFrame]:
    """
    Compute population-level frequency tables from the TRAINING set only.

    Usage
    ─────
        labels_lf = pl.scan_parquet("../../data/train_labels.parquet")
        stats = compute_global_stats(train_lf, labels_lf=labels_lf)
        # save to disk for reuse
        for key, lf in stats.items():
            lf.collect().write_parquet(f"stats_{key}.parquet")

        # at inference time, reload:
        stats = {
            "mcc_global": pl.scan_parquet("stats_mcc_global.parquet"),
            ...
        }
        features_lf = generate_fraud_features(inference_lf, global_stats=stats)

    Parameters
    ──────────
    train_lf   : LazyFrame of transaction data (pretrain + train).
    labels_lf  : Optional LazyFrame with columns (event_id, target).
                 When provided, Bayesian-smoothed target encodings are
                 computed for event_type_nm, event_desc,
                 channel_indicator_type, channel_indicator_sub_type,
                 and the (event_type_nm, event_desc) pair.
                 These are joined at feature-engineering time and give
                 LGBM a direct continuous fraud-risk signal per category.
    """
    train_lf = train_lf.with_columns(
        col("operaton_amt").fill_null(0).alias("amount_clean")
    )

    col_cfg = [
        # (raw_col,                stat_alias,               key)
        ("mcc_code",               "global_mcc_freq",        "mcc_global"),
        ("channel_indicator_type", "global_channel_freq",    "channel_global"),
        ("operating_system_type",  "global_device_os_freq",  "os_global"),
        ("timezone",               "global_timezone_freq",   "timezone_global"),
        ("accept_language",        "global_language_freq",   "language_global"),
        ("pos_cd",                 "global_pos_cd_freq",     "pos_global"),
        ("event_type_nm",          "global_event_type_freq", "evtype_global"),
        ("event_desc",             "global_event_desc_freq", "evdesc_global"),
    ]

    stats: dict[str, pl.LazyFrame] = {}
    for raw_col, stat_alias, key in col_cfg:
        stats[key] = (
            train_lf
            .group_by(raw_col)
            .agg(pl.len().alias(stat_alias))
        )

    stats["channel_stats_global"] = (
        train_lf
        .group_by("channel_indicator_type")
        .agg([
            col("amount_clean").mean().alias("channel_mean"),
            col("amount_clean").std().alias("channel_std"),
        ])
    )
    stats["mcc_stats_global"] = (
        train_lf
        .group_by("mcc_code")
        .agg([
            col("amount_clean").mean().alias("mcc_mean"),
            col("amount_clean").std().alias("mcc_std"),
        ])
    )

    stats["combination_global"] = (
        train_lf
        .group_by(["channel_indicator_type", "operating_system_type"])
        .agg(pl.len().alias("global_combination_freq"))
    )

    # ── Per-category amount distribution statistics ────────────────────────────
    # Used by category_stats.py to compute global (population-level) z-scores.
    stats["event_desc_stats_global"] = (
        train_lf
        .group_by("event_desc")
        .agg([
            col("amount_clean").mean().alias("event_desc_global_mean"),
            col("amount_clean").std().alias("event_desc_global_std"),
        ])
    )

    stats["event_type_stats_global"] = (
        train_lf
        .group_by("event_type_nm")
        .agg([
            col("amount_clean").mean().alias("event_type_global_mean"),
            col("amount_clean").std().alias("event_type_global_std"),
        ])
    )

    stats["subchannel_stats_global"] = (
        train_lf
        .group_by("channel_indicator_sub_type")
        .agg([
            col("amount_clean").mean().alias("subchannel_global_mean"),
            col("amount_clean").std().alias("subchannel_global_std"),
        ])
    )

    stats["pos_stats_global"] = (
        train_lf
        .group_by("pos_cd")
        .agg([
            col("amount_clean").mean().alias("pos_global_mean"),
            col("amount_clean").std().alias("pos_global_std"),
        ])
    )

    stats["subchannel_global"] = (
        train_lf
        .group_by("channel_indicator_sub_type")
        .agg(pl.len().alias("global_subchannel_freq"))
    )

    # ── Combined channel_type × subtype stats ──────────────────────────────
    # Derived inline (same formula as transaction.py) so we don't depend on
    # the feature-engineering pipeline having already added the column.
    _train_with_cts = train_lf.with_columns(
        (col("channel_indicator_type") * 1000 + col("channel_indicator_sub_type"))
        .cast(pl.Int32)
        .alias("channel_type_subtype")
    )
    stats["channel_type_subtype_stats_global"] = (
        _train_with_cts
        .group_by("channel_type_subtype")
        .agg([
            col("amount_clean").mean().alias("channel_type_subtype_global_mean"),
            col("amount_clean").std().alias("channel_type_subtype_global_std"),
        ])
    )
    stats["channel_type_subtype_global"] = (
        _train_with_cts
        .group_by("channel_type_subtype")
        .agg(pl.len().alias("global_channel_type_subtype_freq"))
    )

    # ── Combined evtype_channel / evtype_subchannel stats ─────────────────────
    _train_with_ev = train_lf.with_columns([
        (col("event_type_nm") * 1000 + col("channel_indicator_type"))
        .cast(pl.Int32)
        .alias("evtype_channel"),

        (col("event_type_nm") * 1000 + col("channel_indicator_sub_type"))
        .cast(pl.Int32)
        .alias("evtype_subchannel"),
    ])
    stats["evtype_channel_stats_global"] = (
        _train_with_ev
        .group_by("evtype_channel")
        .agg([
            col("amount_clean").mean().alias("evtype_channel_global_mean"),
            col("amount_clean").std().alias("evtype_channel_global_std"),
        ])
    )
    stats["evtype_channel_global"] = (
        _train_with_ev
        .group_by("evtype_channel")
        .agg(pl.len().alias("global_evtype_channel_freq"))
    )
    stats["evtype_subchannel_stats_global"] = (
        _train_with_ev
        .group_by("evtype_subchannel")
        .agg([
            col("amount_clean").mean().alias("evtype_subchannel_global_mean"),
            col("amount_clean").std().alias("evtype_subchannel_global_std"),
        ])
    )
    stats["evtype_subchannel_global"] = (
        _train_with_ev
        .group_by("evtype_subchannel")
        .agg(pl.len().alias("global_evtype_subchannel_freq"))
    )

    # ── Combined event_type_nm × mcc_code stats (two-column, mcc is String) ─
    stats["evtype_mcc_stats_global"] = (
        train_lf
        .group_by(["event_type_nm", "mcc_code"])
        .agg([
            col("amount_clean").mean().alias("evtype_mcc_global_mean"),
            col("amount_clean").std().alias("evtype_mcc_global_std"),
        ])
    )
    stats["evtype_mcc_global"] = (
        train_lf
        .group_by(["event_type_nm", "mcc_code"])
        .agg(pl.len().alias("global_evtype_mcc_freq"))
    )

    # ── Bayesian-smoothed target encodings ────────────────────────────────────
    # Requires labels_lf (event_id → target).  Skipped when not provided.
    if labels_lf is not None:
        _labels = (
            labels_lf.lazy()
            if isinstance(labels_lf, pl.DataFrame)
            else labels_lf
        )
        # Inner join: keep only labeled training rows
        labeled_tx = train_lf.join(_labels, on="event_id", how="inner")

        # tx_type_group is a derived column added by add_transaction_features()
        # during feature engineering, so it is absent from the raw transaction
        # LazyFrame passed here.  Derive it inline using the same logic so the
        # within-group channel encodings can group_by it.
        labeled_tx = labeled_tx.with_columns(
            pl.when(col("operaton_amt").is_null())
            .then(pl.lit(0))
            .when(col("mcc_code").is_not_null() & (col("mcc_code") != ""))
            .then(pl.lit(1))
            .otherwise(pl.lit(2))
            .cast(pl.Int8)
            .alias("tx_type_group")
        )

        # Global fraud rate — used as the Bayesian prior mean.
        # Collected once here (tiny scalar) so we can embed it in each formula.
        _agg = labeled_tx.select([
            pl.len().alias("n"),
            pl.col("target").sum().alias("s"),
        ]).collect()
        _global_rate = float(_agg["s"][0]) / float(_agg["n"][0])

        # Store global rate so category_stats.py can use it as a null-fill
        # for unseen category values at test time.
        stats["global_fraud_rate"] = pl.DataFrame(
            {"global_fraud_rate": [_global_rate]}
        ).lazy()

        # ── Single-column target encodings ────────────────────────────────────
        # Formula (additive Bayesian smoothing toward the global mean):
        #   smoothed = (fraud_count + α * global_rate) / (labeled_count + α)
        # This is equivalent to starting with α pseudo-observations split at
        # the global fraud rate and then updating with the observed counts.
        _te_single = [
            # (join column,                 output feature name,                stats key)
            ("event_type_nm",            "event_type_nm_target_enc",      "evtype_target_enc"),
            ("event_desc",               "event_desc_target_enc",          "evdesc_target_enc"),
            ("channel_indicator_type",   "channel_type_target_enc",        "channel_type_target_enc"),
            ("channel_indicator_sub_type","channel_subtype_target_enc",    "channel_subtype_target_enc"),
            # mcc_code is null for non-card transactions; unseen values filled with global rate at join time
            ("mcc_code",                  "mcc_target_enc",                 "mcc_target_enc"),
        ]

        # Derive combined columns on labeled_tx so the TEs can group by them.
        labeled_tx = labeled_tx.with_columns([
            (col("channel_indicator_type") * 1000 + col("channel_indicator_sub_type"))
            .cast(pl.Int32)
            .alias("channel_type_subtype"),

            (col("event_type_nm") * 1000 + col("channel_indicator_type"))
            .cast(pl.Int32)
            .alias("evtype_channel"),

            (col("event_type_nm") * 1000 + col("channel_indicator_sub_type"))
            .cast(pl.Int32)
            .alias("evtype_subchannel"),
        ])
        _te_single.extend([
            ("channel_type_subtype",  "channel_type_subtype_target_enc", "channel_type_subtype_target_enc"),
            ("evtype_channel",        "evtype_channel_target_enc",       "evtype_channel_target_enc"),
            ("evtype_subchannel",     "evtype_subchannel_target_enc",    "evtype_subchannel_target_enc"),
        ])
        for join_col, feat_name, key in _te_single:
            stats[key] = (
                labeled_tx
                .group_by(join_col)
                .agg([
                    col("target").sum().cast(pl.Float64).alias("_fraud"),
                    pl.len().cast(pl.Float64).alias("_n"),
                ])
                .with_columns([
                    (
                        (col("_fraud") + _ALPHA * _global_rate) /
                        (col("_n") + _ALPHA)
                    ).alias(feat_name)
                ])
                .select([join_col, feat_name])
            )

        # ── Pair target encoding: (event_type_nm × event_desc) ────────────────
        # The pair has the strongest signal (0 %–100 % fraud rate range) and
        # is too large an interaction space for trees to learn unaided.
        stats["type_desc_target_enc"] = (
            labeled_tx
            .group_by(["event_type_nm", "event_desc"])
            .agg([
                col("target").sum().cast(pl.Float64).alias("_fraud"),
                pl.len().cast(pl.Float64).alias("_n"),
            ])
            .with_columns([
                (
                    (col("_fraud") + _ALPHA * _global_rate) /
                    (col("_n") + _ALPHA)
                ).alias("type_desc_pair_target_enc")
            ])
            .select(["event_type_nm", "event_desc", "type_desc_pair_target_enc"])
        )

        # ── Pair target encoding: (event_type_nm × mcc_code) ─────────────────
        # Two-column join because mcc_code is String type.
        stats["evtype_mcc_target_enc"] = (
            labeled_tx
            .group_by(["event_type_nm", "mcc_code"])
            .agg([
                col("target").sum().cast(pl.Float64).alias("_fraud"),
                pl.len().cast(pl.Float64).alias("_n"),
            ])
            .with_columns([
                (
                    (col("_fraud") + _ALPHA * _global_rate) /
                    (col("_n") + _ALPHA)
                ).alias("evtype_mcc_target_enc")
            ])
            .select(["event_type_nm", "mcc_code", "evtype_mcc_target_enc"])
        )

        # ── Within-group channel target encodings ─────────────────────────────
        # The same channel can have very different fraud rates across tx_type_groups.
        # Example: channel_indicator_type=0 in p2p → 90.5 % fraud,
        #          channel_indicator_type=0 in nonpayment → 52.5 % fraud.
        # A raw integer encoding forces LGBM to re-learn this via deep interactions;
        # these smoothed rates expose it as a single continuous feature per group.
        _te_within_group = [
            # (group-by columns,                              feature name,                            stats key)
            (["tx_type_group", "channel_indicator_type"],     "channel_type_fraud_rate_within_group",  "channel_type_within_group_te"),
            (["tx_type_group", "channel_indicator_sub_type"], "channel_subtype_fraud_rate_within_group","channel_subtype_within_group_te"),
        ]
        for join_cols, feat_name, key in _te_within_group:
            stats[key] = (
                labeled_tx
                .group_by(join_cols)
                .agg([
                    col("target").sum().cast(pl.Float64).alias("_fraud"),
                    pl.len().cast(pl.Float64).alias("_n"),
                ])
                .with_columns([
                    (
                        (col("_fraud") + _ALPHA * _global_rate) /
                        (col("_n") + _ALPHA)
                    ).alias(feat_name)
                ])
                .select(join_cols + [feat_name])
            )

    return stats
