import polars as pl
from polars import col, lit, when


def add_transaction_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Section A: Transaction-Level Features."""
    lf = lf.with_columns([
        pl.col("event_dttm").str.strptime(pl.Datetime, strict=False),
    ])

    lf = lf.with_columns([
        col("event_dttm").dt.hour().alias("hour"),
        col("event_dttm").dt.weekday().alias("day_of_week"),
        col("event_dttm").dt.day().alias("day_of_month"),
        col("event_dttm").dt.week().alias("week_of_year"),
        col("operaton_amt").fill_null(0).alias("amount_clean"),
    ])

    lf = lf.with_columns([
        col("hour").alias("hour_of_day"),
        (col("day_of_week") >= 6).cast(pl.Int8).alias("is_weekend"),
        (col("hour") <= 6).cast(pl.Int8).alias("is_night"),
        col("hour").is_between(9, 18).cast(pl.Int8).alias("is_working_hour"),
        (col("hour") * 60 + col("event_dttm").dt.minute()).alias("minutes_from_midnight"),
        col("amount_clean").log1p().alias("log_amount"),
        col("amount_clean").abs().alias("amount_abs"),
        (col("amount_clean") % 100 == 0).cast(pl.Int8).alias("amount_round_100"),
        (col("amount_clean") % 1000 == 0).cast(pl.Int8).alias("amount_round_1000"),
        (col("amount_clean") % 1 == 0).cast(pl.Int8).alias("amount_is_integer"),
        col("operaton_amt").is_null().cast(pl.Int8).alias("amount_missing_flag"),
        when(col("currency_iso_cd").is_null()).then(1).otherwise(0).cast(pl.Int8).alias("amount_currency_mismatch_flag"),
        lit(None).cast(pl.Float64).alias("amount_usd_normalized"),
    ])

    return lf
