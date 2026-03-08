import polars as pl
from polars import col


def compute_global_stats(train_lf: pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    """
    Compute population-level frequency tables from the TRAINING set only.

    Usage
    ─────
        stats = compute_global_stats(train_lf)
        # save to disk for reuse
        for key, lf in stats.items():
            lf.collect().write_parquet(f"stats_{key}.parquet")

        # at inference time, reload:
        stats = {
            "mcc_global": pl.scan_parquet("stats_mcc_global.parquet"),
            ...
        }
        features_lf = generate_fraud_features(inference_lf, global_stats=stats)
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

    return stats
