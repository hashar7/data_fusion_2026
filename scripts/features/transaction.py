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
        when(col("currency_iso_cd").is_null()).then(1).otherwise(0).cast(pl.Int8).alias("amount_currency_mismatch_flag"),
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

    # ── Transaction type grouping ─────────────────────────────────────────────
    # 0 = non-payment (no amount), 1 = card (has amount + MCC), 2 = P2P (has amount, no MCC)
    lf = lf.with_columns([
        when(col("operaton_amt").is_null())
        .then(lit(0))
        .when(col("mcc_code").is_not_null() & (col("mcc_code") != ""))
        .then(lit(1))
        .otherwise(lit(2))
        .cast(pl.Int8)
        .alias("tx_type_group"),
    ])

    # ── Model routing group ────────────────────────────────────────────────────
    # Splits nonpayment into two sub-groups based on event_type_nm because the
    # within-nonpayment fraud rate varies from 22 % to 79 % across event types
    # and event_type_nm=7 (70 M rows, 58 % fraud) dominates the population,
    # causing a single nonpayment model to under-fit the rarer types.
    #
    # 0 = nonpayment-type7  (tx_type_group==0 AND event_type_nm==7)
    # 1 = nonpayment-other  (tx_type_group==0 AND event_type_nm!=7)
    # 2 = card              (tx_type_group==1)
    # 3 = p2p               (tx_type_group==2)
    #
    # model_group is used only for routing to the correct booster at train/predict
    # time.  It is excluded from the model feature set via NON_FEATURE_COLS.
    lf = lf.with_columns([
        when((col("tx_type_group") == 0) & (col("event_type_nm") == 7))
        .then(lit(0))
        .when((col("tx_type_group") == 0) & (col("event_type_nm") != 7))
        .then(lit(1))
        .when(col("tx_type_group") == 1)
        .then(lit(2))
        .otherwise(lit(3))
        .cast(pl.Int8)
        .alias("model_group"),
    ])

    # ── Combined categorical interaction columns ────────────────────────────────
    # Encode common categorical pairs as single integers so downstream features
    # can use one-column group-by/over instead of two-column.
    lf = lf.with_columns([
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

    # Masked amounts for rolling aggregation by transaction type.
    # amount_card / amount_p2p contribute their amount in the relevant window sums;
    # the other type contributes 0, so window sums give per-type spending directly.
    lf = lf.with_columns([
        when(col("tx_type_group") == 1).then(col("amount_clean")).otherwise(lit(0.0)).alias("amount_card"),
        when(col("tx_type_group") == 2).then(col("amount_clean")).otherwise(lit(0.0)).alias("amount_p2p"),
    ])

    return lf
