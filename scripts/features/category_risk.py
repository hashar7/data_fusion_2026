"""
Section J: Hardcoded binary risk flags derived from categorical analysis.

All thresholds come from the empirical fraud-rate study of pretrain+train data
(notebooks/categorical_stats/).  Each flag gives LGBM a clean one-node split
on a strong signal it would otherwise need many trees to approximate.

No global_stats required — values are compiled into the code and never change
across train / val / test.
"""
import polars as pl
from polars import col, when


# ── event_desc sets (encoded integers, schema: Int32) ─────────────────────────
# All fraud rates quoted from within-group labeled rows, ≥50 labeled samples.

# >89 % fraud in their primary group — near-certain fraud indicators.
_VERY_HIGH_RISK_DESC = [
    60,   # card  : 100 % fraud (150 K rows)
    41,   # card  :  97 % fraud (124 K rows)
    27,   # p2p   :  96 % fraud   (6 K rows)
    73,   # card  :  93 % fraud (413 K rows)
    103,  # card  :  89 % fraud  (33 K rows)
]

# 67 %–89 % fraud in their primary group — high-confidence fraud indicators.
_HIGH_RISK_DESC = [
    68,   # card  :  84 % fraud (288 K rows)
    29,   # p2p   :  83 % fraud (336 K rows)
    31,   # p2p   :  82 % fraud (360 K rows)
    119,  # p2p   :  80 % fraud (186 K rows)
    113,  # p2p   :  76 % fraud  (55 K rows)
    109,  # p2p   :  73 % fraud (102 K rows)
    37,   # p2p   :  68 % fraud (1.95 M rows)
]

# <20 % fraud — low-risk descriptors; helps the model rule out false positives.
_LOW_RISK_DESC = [
    51,   # p2p   :  12 % fraud (489 K rows)
    5,    # p2p   :  12 % fraud (273 K rows)
    4,    # p2p   :   5 % fraud  (39 K rows)
    69,   # p2p   :   0 % fraud   (6 K rows, 25 labeled)
    97,   # p2p   :   0 % fraud   (1 K rows, 29 labeled)
    32,   # p2p   :  16 % fraud (187 K rows)
]

# ── (event_type_nm, event_desc) pairs with near-certain fraud ─────────────────
# These three pairs collectively cover ~700 K card-group rows at 93–100 % fraud.
_NEAR_CERTAIN_FRAUD_TYPE_DESC = [
    (14, 60),  # card : 100 % fraud (150 K rows)
    (14, 41),  # card :  97 % fraud (124 K rows)
    (14, 73),  # card :  93 % fraud (413 K rows)
]


def add_category_risk_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Section J: binary risk flags for high/low-fraud categorical combinations."""

    # ── event_desc risk tier ───────────────────────────────────────────────────
    lf = lf.with_columns([
        col("event_desc").is_in(_VERY_HIGH_RISK_DESC).cast(pl.Int8).alias("is_very_high_risk_desc"),
        col("event_desc").is_in(_HIGH_RISK_DESC).cast(pl.Int8).alias("is_high_risk_desc"),
        col("event_desc").is_in(_LOW_RISK_DESC).cast(pl.Int8).alias("is_low_risk_desc"),
    ])

    # ── channel risk flags ─────────────────────────────────────────────────────
    # channel_indicator_type == 6  → 85.5 % global fraud rate
    # channel_indicator_sub_type == 11 → 87.8 % fraud (mostly card)
    lf = lf.with_columns([
        (
            (col("channel_indicator_type") == 6) |
            (col("channel_indicator_sub_type") == 11)
        ).cast(pl.Int8).alias("is_high_risk_channel"),

        # P2P transactions through sub_type=5: 90.5 % fraud in the p2p group.
        # The same sub_type in card has a different (lower) risk profile,
        # so we gate this flag on tx_type_group == 2.
        (
            (col("tx_type_group") == 2) &
            (col("channel_indicator_sub_type") == 5)
        ).cast(pl.Int8).alias("is_p2p_danger_channel"),
    ])

    # ── near-certain fraud (event_type_nm, event_desc) pair ───────────────────
    # Expressed as a single OR over the three known fraud pairs rather than
    # a struct join, to keep the lazy plan simple and fast.
    near_certain = (
        ((col("event_type_nm") == 14) & (col("event_desc") == 60)) |
        ((col("event_type_nm") == 14) & (col("event_desc") == 41)) |
        ((col("event_type_nm") == 14) & (col("event_desc") == 73))
    )
    lf = lf.with_columns([
        near_certain.cast(pl.Int8).alias("is_near_certain_fraud_pair"),
    ])

    return lf
