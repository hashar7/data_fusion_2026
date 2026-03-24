import polars as pl
from polars import col
from pathlib import Path
import shutil

# Bayesian smoothing strength (pseudo-observation count).
# With ~300 labeled rows per event_desc value on average, alpha=20 pulls
# rare-value estimates ~6 % toward the global mean — conservative and safe.
_ALPHA = 20


TX_KEY_COLS = [
    "channel_indicator_type",
    "channel_indicator_sub_type",
    "event_type_nm",
    "event_desc",
]
TX_KEY_MAP_FILENAME = "transaction_type_key_map.parquet"
TX_KEY_MAP_PATH = Path(__file__).resolve().parent / "../../../data/misc" / TX_KEY_MAP_FILENAME


def learn_transaction_type_keys(history_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Learn transaction type grouping from historical data using a composite key.

    Rules by key:
        0 = non-payment:
            all rows with this key have no operaton_amt, no mcc_code, no pos_cd
        1 = card:
            all rows with this key have mcc_code or pos_cd
        2 = p2p / other payment:
            everything else
    """
    has_amt_expr = pl.col("operaton_amt").is_not_null() & (pl.col("operaton_amt").cast(pl.Float64) > 0)
    has_mcc_expr = pl.col("mcc_code").is_not_null() & (pl.col("mcc_code").str != "")
    has_pos_expr = pl.col("pos_cd").is_not_null()# & (pl.col("pos_cd") != "")
    has_card_marker_expr = has_mcc_expr | has_pos_expr

    key_map_lf = (
        history_lf
        .select(
            TX_KEY_COLS
            + [
                has_amt_expr.alias("has_amt"),
                has_mcc_expr.alias("has_mcc"),
                has_pos_expr.alias("has_pos"),
                has_card_marker_expr.alias("has_card_marker"),
            ]
        )
        .group_by(TX_KEY_COLS)
        .agg([
            pl.col("has_amt").sum().alias("amt_cnt"),
            pl.col("has_mcc").sum().alias("mcc_cnt"),
            pl.col("has_pos").sum().alias("pos_cnt"),
            pl.col("has_card_marker").sum().alias("card_marker_cnt"),
            pl.len().alias("key_row_cnt"),
        ])
        .with_columns([
            pl.when(
                (pl.col("amt_cnt") == 0)
                & (pl.col("mcc_cnt") == 0)
                & (pl.col("pos_cnt") == 0)
            )
            .then(pl.lit(0))
            .when(
                pl.col("card_marker_cnt") > 0#== pl.col("key_row_cnt")
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(2))
            .cast(pl.Int8)
            .alias("tx_type_group"),
        ])
        .select(TX_KEY_COLS + ["tx_type_group"])
    )

    output_path = Path(TX_KEY_MAP_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_map_lf.collect().write_parquet(output_path)

    return key_map_lf


def compute_global_stats(
    train_lf: pl.LazyFrame,
    labels_lf: pl.LazyFrame | None = None,
    # key_map_lf: pl.LazyFrame | None = None,
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

    # learn tx_type_groups on train+pretrain data
    tx_key_map_lf = learn_transaction_type_keys(train_lf)

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
        
        # labeled_tx = labeled_tx.with_columns(
        #     pl.when(col("operaton_amt").is_null())
        #     .then(pl.lit(0))
        #     .when(col("mcc_code").is_not_null() & (col("mcc_code") != ""))
        #     .then(pl.lit(1))
        #     .otherwise(pl.lit(2))
        #     .cast(pl.Int8)
        #     .alias("tx_type_group")
        # )

        labeled_tx = (
            labeled_tx
                .join(tx_key_map_lf, on=TX_KEY_COLS, how="left")
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
