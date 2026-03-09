import polars as pl
from polars import col, lit, when

# Russian federal public holidays encoded as month*100 + day (year-independent).
# Source: Labour Code of the Russian Federation, Art. 112.
# Covers all fixed-date holidays; transfer days (перенос выходных) are not included
# because they change annually and the fixed dates already carry the main signal.
_RU_HOLIDAY_MMDD = [
    101, 102, 103, 104, 105, 106, 107, 108,  # Jan 1–8  : New Year holidays + Orthodox Christmas
    223,   # Feb 23 : Defender of the Fatherland Day
    308,   # Mar 8  : International Women's Day
    501,   # May 1  : Spring and Labour Day
    509,   # May 9  : Victory Day
    612,   # Jun 12 : Russia Day
    1104,  # Nov 4  : National Unity Day
]


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
        # Calendar position — fraudsters tend to strike near paydays / month boundaries
        col("day_of_month").is_in([1, 2, 3]).cast(pl.Int8).alias("is_month_start"),
        col("day_of_month").is_in([28, 29, 30, 31]).cast(pl.Int8).alias("is_month_end"),
        col("day_of_month").is_in([1, 15, 25]).cast(pl.Int8).alias("is_payday"),
        # Russian public holiday flag.
        # Encoded as month*100 + day so the check is a single integer is_in() scan.
        (
            (col("event_dttm").dt.month().cast(pl.Int32) * 100 + col("event_dttm").dt.day().cast(pl.Int32))
            .is_in(_RU_HOLIDAY_MMDD)
            .cast(pl.Int8)
            .alias("is_holiday")
        ),
    ])

    return lf
