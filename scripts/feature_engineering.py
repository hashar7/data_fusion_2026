import polars as pl
from scripts.features.transaction  import add_transaction_features
from scripts.features.behavioral   import add_behavioral_features
from scripts.features.rolling      import add_rolling_features
from scripts.features.device       import add_device_features
from scripts.features.temporal     import add_temporal_features
from scripts.features.zscore       import add_zscore_features
from scripts.features.global_stats import compute_global_stats  # re-exported for callers

_COLS_TO_DROP = ["temp_row_idx", "amount_clean", "channel_mean", "channel_std", "mcc_mean", "mcc_std"]


def generate_fraud_features(
    lf: pl.LazyFrame,
    global_stats: dict[str, pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    lf = lf.sort(["event_dttm", "customer_id"]).with_row_index("temp_row_idx")
    lf = add_transaction_features(lf)
    lf = add_behavioral_features(lf)
    lf = add_rolling_features(lf)
    lf = add_device_features(lf)
    lf = add_temporal_features(lf, global_stats)
    lf = add_zscore_features(lf, global_stats)
    return lf.drop(_COLS_TO_DROP)


# Backward-compatible alias — feature_calculation.py and the notebook use this name
generate_fraud_features_v4 = generate_fraud_features
