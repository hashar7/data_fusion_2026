import polars as pl
from polars import col, lit, when


def generate_fraud_features_v4(
    lf: pl.LazyFrame,
    global_stats: dict[str, pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    """
    Generates fraud detection features.
    Compatible with Polars 1.x.

    Changes vs v3:
    ─────────────────────────────────────────────────────────────────────────────
    FIX 1 – Global frequency leakage (Section F)
        Previously: col(c).cum_count().over(c)
            → counts ALL rows in the dataset (including future ones) grouped by
              the categorical value, so an early transaction gets credit for
              transactions that haven't happened yet.
        Now: cumulative count is computed PER CUSTOMER over time
            col(c).cum_count().over(["customer_id", c])
            This gives "how many times has THIS customer seen this value so far",
            which is strictly in-the-past and safe at inference time.

        For true global priors (population-level frequency of each MCC, channel,
        etc.) you should precompute them on the TRAINING SET ONLY and pass them
        in via the `global_stats` dict. Example keys expected:
            "mcc_global"      → DataFrame with columns [mcc_code, global_mcc_freq]
            "channel_global"  → DataFrame with columns [channel_indicator_type, global_channel_freq]
            "os_global"       → DataFrame with columns [operating_system_type, global_device_os_freq]
            "timezone_global" → DataFrame with columns [timezone, global_timezone_freq]
            "language_global" → DataFrame with columns [accept_language, global_language_freq]
            "pos_global"      → DataFrame with columns [pos_cd, global_pos_cd_freq]
            "evtype_global"   → DataFrame with columns [event_type_nm, global_event_type_freq]
            "evdesc_global"   → DataFrame with columns [event_desc, global_event_desc_freq]
        If not provided, the per-customer cumulative count is used as a proxy.

    FIX 2 – Duplicate features removed
        The following aliases were computing identical expressions:
            amount_round_100  / amount_is_round_100        → kept amount_round_100
            amount_round_1000 / amount_is_round_1000       → kept amount_round_1000
            amount_is_integer / amount_integer             → kept amount_is_integer
            mcc_is_new_for_user / merchant_category_is_new_for_user → kept mcc_is_new_for_user
        device_entropy_ratio was computed twice (Sections F and final block)
            → kept only the final-block version.

    FIX 3 – amount_rank_percentile in rolling agg
        Previously: col("amount_clean").rank("dense")
            → rank() inside .rolling().agg() returns a List column (rank of each
              element within the window), not a scalar, causing a schema error or
              silently producing a list that can't be joined back.
        Now: replaced with a proper within-window percentile proxy:
            (current_amount - window_min) / (window_max - window_min + 1e-9)
            This is computed as a derived column AFTER the join, using the
            already-aggregated window min/max scalars, which are correct scalars.
    ─────────────────────────────────────────────────────────────────────────────
    """

    # ── helpers ──────────────────────────────────────────────────────────────
    def s_num(c):
        return col(c).fill_null(0)

    def s_str(c):
        return col(c).fill_null("Unknown")

    # ── 0. Sort & row index ───────────────────────────────────────────────────
    lf = lf.sort(["event_dttm", "customer_id"]).with_row_index("temp_row_idx")

    # =========================================================================
    # SECTION A: Transaction-Level Features
    # =========================================================================
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
        # FIX 2: removed duplicates amount_is_round_100, amount_is_round_1000,
        #         amount_integer – kept the first occurrence of each.
        (col("amount_clean") % 100 == 0).cast(pl.Int8).alias("amount_round_100"),
        (col("amount_clean") % 1000 == 0).cast(pl.Int8).alias("amount_round_1000"),
        (col("amount_clean") % 1 == 0).cast(pl.Int8).alias("amount_is_integer"),
        col("operaton_amt").is_null().cast(pl.Int8).alias("amount_missing_flag"),
        when(col("currency_iso_cd").is_null()).then(1).otherwise(0).cast(pl.Int8).alias("amount_currency_mismatch_flag"),
        lit(None).cast(pl.Float64).alias("amount_usd_normalized"),
    ])

    # =========================================================================
    # SECTION B: Behavioral & User History (Cumulative)
    # =========================================================================
    lf = lf.with_columns([
        # Subtract 1 so each value reflects the count of transactions BEFORE the
        # current row (i.e. strictly historical).  cum_count() includes the current
        # row by design, which would leak the current transaction into its own feature.
        (col("event_id").cum_count().over("customer_id") - 1).alias("tx_count_lifetime"),
        (col("event_dttm").diff().over("customer_id").dt.total_minutes().fill_null(0)).alias("time_since_last_tx_minutes"),
        (col("mcc_code").cum_count().over(["customer_id", "mcc_code"]) - 1).alias("mcc_freq_user_cum"),
        (col("pos_cd").cum_count().over(["customer_id", "pos_cd"]) - 1).alias("pos_freq_user_cum"),
    ])

    lf = lf.with_columns([
        # FIX 2: merchant_category_is_new_for_user was identical to mcc_is_new_for_user → removed.
        # After the -1 shift, 0 means "never seen before this transaction" (was 1).
        (col("mcc_freq_user_cum") == 0).cast(pl.Int8).alias("mcc_is_new_for_user"),
        (col("pos_freq_user_cum") == 0).cast(pl.Int8).alias("pos_cd_is_new"),
        # fill_null(0): tx_count_lifetime == 0 on the very first transaction → 0/0 → 0.
        (col("mcc_freq_user_cum") / col("tx_count_lifetime")).fill_null(0).alias("mcc_transaction_share_user"),
        (col("pos_freq_user_cum") / col("tx_count_lifetime")).fill_null(0).alias("pos_cd_transaction_share_user"),
        (col("mcc_code") != col("mcc_code").shift(1).over("customer_id")).fill_null(False).cast(pl.Int8).alias("merchant_switch_flag"),
        col("time_since_last_tx_minutes").rolling_mean(window_size=3).over("customer_id").alias("time_since_last_3_tx_mean"),
        col("mcc_freq_user_cum").alias("mcc_frequency_user"),
    ])

    # =========================================================================
    # SECTION D: Rolling Statistics (Lazy-compatible)
    # =========================================================================
    # FIX 3: amount_rank_percentile removed from the agg (it produced a List,
    #         not a scalar).  A proper percentile proxy is computed AFTER the
    #         join using already-aggregated min/max scalars.
    def get_rolling_stats(base_lf: pl.LazyFrame, period: str, suffix: str) -> pl.LazyFrame:
        return (
            base_lf.rolling(
                index_column="event_dttm",
                period=period,
                by="customer_id",
                # closed="left" → window is [t - period, t), which excludes the
                # current transaction's timestamp.  The default "right" uses
                # (t - period, t], meaning the current row is always in its own
                # window — leaking the current amount/channel/etc. into the very
                # aggregates used to judge how unusual it is.
                closed="left",
            )
            .agg([
                pl.len().alias(f"tx_count_{suffix}"),
                col("amount_clean").mean().alias(f"amount_mean_{suffix}"),
                col("amount_clean").std().alias(f"amount_std_{suffix}"),
                col("amount_clean").median().alias(f"amount_median_{suffix}"),
                col("amount_clean").max().alias(f"amount_max_{suffix}"),
                col("amount_clean").min().alias(f"amount_min_{suffix}"),
                col("amount_clean").sum().alias(f"cumulative_spend_{suffix}"),
                col("channel_indicator_type").n_unique().alias(f"channel_diversity_{suffix}"),
                col("operating_system_type").n_unique().alias(f"device_diversity_{suffix}"),
                col("mcc_code").n_unique().alias(f"merchant_diversity_{suffix}"),
            ])
            # Rolling preserves input row order → with_row_index produces the same
            # 0-based sequence as the temp_row_idx already on lf, guaranteeing a
            # 1-to-1 join even when multiple transactions share the same event_dttm.
            # The old approach (col("temp_row_idx").last() inside .agg()) returned
            # the *last* temp_row_idx within the window, so timestamp-tied rows all
            # got the same key → many-to-many join → duplicate output rows.
            .with_row_index("temp_row_idx")
            .select([col("temp_row_idx"), pl.exclude(["customer_id", "event_dttm", "temp_row_idx"])])
        )

    r1d  = get_rolling_stats(lf, "1d",  "1d")
    r3d  = get_rolling_stats(lf, "3d",  "3d")
    r7d  = get_rolling_stats(lf, "7d",  "7d")
    r30d = get_rolling_stats(lf, "30d", "30d")
    r90d = get_rolling_stats(lf, "90d", "90d")

    lf = lf.join(r1d,  on="temp_row_idx", how="left")
    lf = lf.join(r3d,  on="temp_row_idx", how="left")
    lf = lf.join(r7d,  on="temp_row_idx", how="left")
    lf = lf.join(r30d, on="temp_row_idx", how="left")
    lf = lf.join(r90d, on="temp_row_idx", how="left")

    # FIX 3: Compute within-window rank percentile AFTER join, using scalar min/max.
    #   Formula: (amount - window_min) / (window_max - window_min + ε)
    #   Produces a value in [0, 1] indicating where this transaction sits
    #   relative to the observed range within each window. Computed for 7d, 30d.
    EPS = 1e-9
    lf = lf.with_columns([
        (
            (col("amount_clean") - col("amount_min_7d")) /
            (col("amount_max_7d") - col("amount_min_7d") + EPS)
        ).alias("amount_rank_percentile_7d"),
        (
            (col("amount_clean") - col("amount_min_30d")) /
            (col("amount_max_30d") - col("amount_min_30d") + EPS)
        ).alias("amount_rank_percentile_30d"),
        (
            (col("amount_clean") - col("amount_min_90d")) /
            (col("amount_max_90d") - col("amount_min_90d") + EPS)
        ).alias("amount_rank_percentile_90d"),
    ])

    # Derived Rolling Features
    lf = lf.with_columns([
        ((col("amount_clean") - col("amount_mean_30d")) / col("amount_std_30d").fill_null(1)).alias("amount_zscore_30d"),
        (col("amount_clean") / col("amount_mean_30d").fill_null(1)).alias("amount_ratio_to_mean_30d"),
        (col("tx_count_30d") / 30.0).alias("avg_tx_per_day_30d"),
        (col("cumulative_spend_30d") / 30.0).alias("spend_velocity_1d"),
        (col("tx_count_7d") / col("tx_count_90d").fill_null(1)).alias("tx_count_ratio_7d_vs_90d"),
        (col("amount_mean_7d") / col("amount_mean_30d").fill_null(1)).alias("amount_mean_ratio_7d_vs_90d"),
        (col("amount_clean") - col("amount_clean").shift(1).over("customer_id")).fill_null(0).alias("amount_diff_from_prev"),
        (col("amount_clean") / col("amount_clean").shift(1).over("customer_id").fill_null(1)).alias("amount_ratio_prev"),
        when(col("time_since_last_tx_minutes") < 5).then(1).otherwise(0).cast(pl.Int8).alias("burst_flag"),
    ])

    # =========================================================================
    # SECTION E: Device & Risk
    # =========================================================================
    lf = lf.with_columns([
        when(col("compromised") == "true").then(1).otherwise(0).cast(pl.Int8).alias("compromised_flag"),
        col("web_rdp_connection").fill_null(0).cast(pl.Int8).alias("web_rdp_connection_flag"),
        when(col("developer_tools") == "true").then(1).otherwise(0).cast(pl.Int8).alias("developer_tools_flag"),
        col("phone_voip_call_state").fill_null(0).cast(pl.Int8).alias("phone_voip_call_flag"),
        when(
            col("battery")
            .str.extract(r"(\d+(?:\.\d+)?)\s*%", 1)   # pull the first number before '%'
            .cast(pl.Float64)
            .fill_null(100)                             # treat missing/unparseable as 100 (not low)
            < 15
        ).then(1).otherwise(0).cast(pl.Int8).alias("low_battery_flag"),
    ])

    lf = lf.with_columns([
        # Same -1 shift as Section B: check == 0 (zero prior uses) rather than
        # == 1 (first occurrence including the current row).
        (col("operating_system_type").cum_count().over(["customer_id", "operating_system_type"]) - 1 == 0).cast(pl.Int8).alias("operating_system_is_new"),
        (col("device_system_version").cum_count().over(["customer_id", "device_system_version"]) - 1 == 0).cast(pl.Int8).alias("os_version_is_new"),
        (col("screen_size").cum_count().over(["customer_id", "screen_size"]) - 1 == 0).cast(pl.Int8).alias("screen_size_is_new"),
        (col("timezone").cum_count().over(["customer_id", "timezone"]) - 1 == 0).cast(pl.Int8).alias("timezone_is_new"),
        (col("accept_language").cum_count().over(["customer_id", "accept_language"]) - 1 == 0).cast(pl.Int8).alias("accept_language_is_new"),
        (col("compromised_flag") * 5 + col("web_rdp_connection_flag") * 3 + col("developer_tools_flag") * 2).cast(pl.Float64).alias("device_risk_score"),
        # Session features: len/sum/n_unique over "session_id" aggregated the FULL
        # session (including future transactions not yet seen at scoring time).
        # Replaced with cumulative-within-session expressions that use only
        # transactions that occurred BEFORE the current one.
        (col("event_id").cum_count().over(["customer_id", "session_id"]) - 1).alias("session_tx_count"),
        (col("amount_clean").cum_sum().over(["customer_id", "session_id"]) - col("amount_clean")).fill_null(0).alias("session_amount_sum"),
        # n_unique over a growing window isn't directly expressible in Polars;
        # replaced with a binary proxy: did the channel change from the previous
        # transaction in this session? Same directional fraud signal, no leakage.
        (col("channel_indicator_type") != col("channel_indicator_type").shift(1).over(["customer_id", "session_id"])).fill_null(False).cast(pl.Int8).alias("session_channel_diversity"),
        (col("event_id").cum_count().over(["customer_id", "session_id"]) - 1).alias("session_length_estimate"),
    ])

    # =========================================================================
    # SECTION F: Temporal & Global (Leakage-Free)
    # =========================================================================
    # FIX 1: Global frequency leakage.
    #
    #   OLD (leaky):
    #       col(c).cum_count().over(c)
    #   This partitions the FULL dataset by the categorical value c and counts
    #   rows cumulatively — but cumulative here means "within this sorted dataset",
    #   not "up to the current point in time per customer". Because the dataset is
    #   sorted by [event_dttm, customer_id], an early transaction for customer A
    #   will have a low count, but only because other customers haven't appeared
    #   yet — not because of any time-based restriction. At inference time on a
    #   new batch, the counts would be completely different, causing train/test
    #   distribution shift.
    #
    #   NEW (leakage-free, two options):
    #
    #   Option A – Per-customer cumulative count (always available, zero leakage):
    #       col(c).cum_count().over(["customer_id", c])
    #   Meaning: "how many times has THIS customer transacted with this value
    #   before (and including) now?"  Pure historical signal, identical at train
    #   and inference time.
    #
    #   Option B – Static global prior from training set (best for population freq):
    #       Precompute on TRAIN SET ONLY → join as a lookup table.
    #       Pass via the `global_stats` dict argument.
    #   This gives a true population frequency without any leakage because the
    #   numbers are frozen at training time and reused unchanged at inference.
    #
    #   We implement Option A as the default and Option B when global_stats is
    #   provided.

    global_col_map = [
        # (raw_col,                    alias,                    global_stats_key)
        ("mcc_code",                   "global_mcc_freq",        "mcc_global"),
        ("channel_indicator_type",     "global_channel_freq",    "channel_global"),
        ("operating_system_type",      "global_device_os_freq",  "os_global"),
        ("timezone",                   "global_timezone_freq",   "timezone_global"),
        ("accept_language",            "global_language_freq",   "language_global"),
        ("pos_cd",                     "global_pos_cd_freq",     "pos_global"),
        ("event_type_nm",              "global_event_type_freq", "evtype_global"),
        ("event_desc",                 "global_event_desc_freq", "evdesc_global"),
    ]

    if global_stats:
        # Option B: join precomputed train-set frequencies (recommended).
        for raw_col, alias, key in global_col_map:
            if key in global_stats:
                lf = lf.join(
                    global_stats[key].lazy() if isinstance(global_stats[key], pl.DataFrame)
                    else global_stats[key],
                    on=raw_col,
                    how="left",
                ).rename({alias: alias})   # column already named correctly in the lookup
            else:
                # Fallback to Option A for missing keys.
                lf = lf.with_columns([
                    (col(raw_col).cum_count().over(["customer_id", raw_col]) - 1).alias(alias)
                ])
    else:
        # Option A: per-customer cumulative count — no leakage, always usable.
        # Subtract 1 so the value reflects the count of PRIOR uses (current row
        # excluded), consistent with the -1 shift applied in Sections B and E.
        lf = lf.with_columns([
            (col(raw_col).cum_count().over(["customer_id", raw_col]) - 1).alias(alias)
            for raw_col, alias, _ in global_col_map
        ])

    # Additional temporal features (leakage-free — all use per-customer history)
    lf = lf.with_columns([
        # circadian_deviation_score: how far is the current hour from the
        # customer's TYPICAL hour?  The old mean().over("customer_id") used all
        # transactions including future ones.  Fix: expanding mean of PRIOR hours
        # = cum_sum shifted by 1 (excludes current row) / tx_count_lifetime
        # (already the count of prior transactions after the Section B fix).
        (
            col("hour_of_day") -
            col("hour_of_day").cum_sum().over("customer_id").shift(1).fill_null(0) /
            col("tx_count_lifetime").clip(lower_bound=1)
        ).abs().alias("circadian_deviation_score"),
        (col("channel_indicator_type") != col("channel_indicator_type").shift(1).over("customer_id")).fill_null(False).cast(pl.Int8).alias("channel_shift_score"),
        when(col("tx_count_1d") > col("avg_tx_per_day_30d") * 2).then(1).otherwise(0).cast(pl.Int8).alias("velocity_change_flag"),
        # time_gap_variance_30d: old std().over("customer_id") used the customer's
        # full all-time history including future transactions.  Fix: rolling_std
        # over a window of 90 prior gaps (shift(1) excludes the current gap).
        col("time_since_last_tx_minutes").shift(1).rolling_std(window_size=90, min_periods=2).over("customer_id").fill_null(0).alias("time_gap_variance_30d"),
        col("operating_system_is_new").alias("new_device_flag"),
        col("mcc_is_new_for_user").alias("new_mcc_flag"),
        # Same -1 shift: 0 prior uses of this channel = new channel for customer.
        (col("channel_indicator_type").cum_count().over(["customer_id", "channel_indicator_type"]) - 1 == 0).cast(pl.Int8).alias("new_channel_flag"),
        # Merchant entropy: distinct MCCs seen so far / total transactions so far (per customer).
        (col("mcc_freq_user_cum") / col("tx_count_lifetime")).fill_null(0).alias("merchant_entropy_user"),
    ])

    lf = lf.with_columns([
        when(col("accept_language") != col("browser_language")).then(1).otherwise(0).cast(pl.Int8).alias("browser_language_mismatch"),
    ])

    lf = lf.with_columns([
        when((col("is_night") == 1) & (col("new_device_flag") == 1)).then(1).otherwise(0).cast(pl.Int8).alias("new_device_and_night_flag"),
        when((col("web_rdp_connection_flag") == 1) & (col("amount_clean") > 1000)).then(1).otherwise(0).cast(pl.Int8).alias("rdp_and_large_amount_flag"),
        # amount_zscore_given_channel/mcc/device: old mean().over(category) computed
        # the global mean across ALL dataset rows for that category, including future
        # transactions.  Replaced with the customer's own prior mean for each
        # category (cum_sum - current) / (cum_count - 1), which is leakage-free and
        # arguably more informative for fraud (personalised baseline vs. global one).
        # fill_null(0): first encounter of this category for the customer → prior
        # count = 0, clipped to 1 to avoid div/0, sum = 0, result = 0.
        (
            (col("amount_clean").cum_sum().over(["customer_id", "channel_indicator_type"]) - col("amount_clean")) /
            (col("event_id").cum_count().over(["customer_id", "channel_indicator_type"]) - 1).clip(lower_bound=1)
        ).fill_null(0).alias("amount_zscore_given_channel"),
        (
            (col("amount_clean").cum_sum().over(["customer_id", "mcc_code"]) - col("amount_clean")) /
            (col("event_id").cum_count().over(["customer_id", "mcc_code"]) - 1).clip(lower_bound=1)
        ).fill_null(0).alias("amount_zscore_given_mcc"),
        (
            (col("amount_clean").cum_sum().over(["customer_id", "operating_system_type"]) - col("amount_clean")) /
            (col("event_id").cum_count().over(["customer_id", "operating_system_type"]) - 1).clip(lower_bound=1)
        ).fill_null(0).alias("amount_zscore_given_device"),
        # tx_time_zscore_given_user: old std().over("customer_id") included future
        # transactions.  Fix: rolling_std over prior 90 hours (shift(1) excludes
        # the current row's hour from the window).
        col("hour_of_day").shift(1).rolling_std(window_size=90, min_periods=2).over("customer_id").fill_null(0).alias("tx_time_zscore_given_user"),
    ])

    # ============================================================
    # SECTION G: Channel / MCC Conditional Z-Scores
    # ============================================================
    # channel_mean/std and mcc_mean/std: the old mean/std.over(category) computes
    # the statistic across ALL rows with that category value — including transactions
    # that happen AFTER the current one.  Fix (two options):
    #
    #   Option B (preferred): join precomputed training-set stats passed via
    #       global_stats["channel_stats_global"] / global_stats["mcc_stats_global"].
    #       compute_global_stats() now produces these tables.
    #
    #   Fallback: customer-level expanding mean/std (only prior rows, no leakage).
    #       Computed using cumulative sum and sum-of-squares, which give the exact
    #       sample mean/std for all transactions seen before the current one.

    def _prior_mean_expr(group_cols: list) -> pl.Expr:
        """Expanding mean of amount_clean for group_cols, excluding the current row."""
        n = (col("event_id").cum_count().over(group_cols) - 1).clip(lower_bound=1)
        s = col("amount_clean").cum_sum().over(group_cols) - col("amount_clean")
        return (s / n).fill_null(0)

    def _prior_std_expr(group_cols: list) -> pl.Expr:
        """Expanding sample std of amount_clean for group_cols, excluding the current row."""
        n   = col("event_id").cum_count().over(group_cols) - 1
        s   = col("amount_clean").cum_sum().over(group_cols) - col("amount_clean")
        sq  = (col("amount_clean") ** 2).cum_sum().over(group_cols) - col("amount_clean") ** 2
        # Sample variance: (Σx² − (Σx)²/n) / (n−1)
        var = (sq - s ** 2 / n.clip(lower_bound=1)) / (n - 1).clip(lower_bound=1)
        return var.clip(lower_bound=0).sqrt()

    if global_stats and "channel_stats_global" in global_stats and "mcc_stats_global" in global_stats:
        _ch_sf  = (global_stats["channel_stats_global"].lazy()
                   if isinstance(global_stats["channel_stats_global"], pl.DataFrame)
                   else global_stats["channel_stats_global"])
        _mcc_sf = (global_stats["mcc_stats_global"].lazy()
                   if isinstance(global_stats["mcc_stats_global"], pl.DataFrame)
                   else global_stats["mcc_stats_global"])
        lf = lf.join(_ch_sf,  on="channel_indicator_type", how="left")
        lf = lf.join(_mcc_sf, on="mcc_code",               how="left")
    else:
        lf = lf.with_columns([
            _prior_mean_expr(["customer_id", "channel_indicator_type"]).alias("channel_mean"),
            _prior_std_expr( ["customer_id", "channel_indicator_type"]).alias("channel_std"),
            _prior_mean_expr(["customer_id", "mcc_code"]).alias("mcc_mean"),
            _prior_std_expr( ["customer_id", "mcc_code"]).alias("mcc_std"),
        ])

    lf = lf.with_columns([
        ((col("amount_clean") - col("channel_mean")) / col("channel_std").fill_null(1)).alias("amount_zscore_channel"),
        ((col("amount_clean") - col("mcc_mean"))     / col("mcc_std").fill_null(1)).alias("amount_zscore_mcc"),
    ])

    # global_combination_freq: the old count().over([channel, os]) scanned ALL
    # rows in the batch for each (channel, os) pair — leaky.
    # Option B: join precomputed freq from training data (global_stats["combination_global"]).
    # Fallback: per-customer cumulative count for this (channel, os) combo.
    if global_stats and "combination_global" in global_stats:
        _combo_sf = (global_stats["combination_global"].lazy()
                     if isinstance(global_stats["combination_global"], pl.DataFrame)
                     else global_stats["combination_global"])
        lf = lf.join(_combo_sf, on=["channel_indicator_type", "operating_system_type"], how="left")
        lf = lf.with_columns(col("global_combination_freq").fill_null(0))
    else:
        lf = lf.with_columns(
            (col("event_id").cum_count().over(["customer_id", "channel_indicator_type", "operating_system_type"]) - 1)
            .alias("global_combination_freq")
        )

    lf = lf.with_columns([
        # session_first_tx_flag: old min(event_id).over([customer, session]) == event_id
        # inspected ALL event_ids in the session (including future transactions).
        # Fix: "is this the first transaction seen so far in this session?" is simply
        # cum_count within (customer, session) == 1 (first occurrence, no future look).
        (col("event_id").cum_count().over(["customer_id", "session_id"]) == 1).cast(pl.Int8).alias("session_first_tx_flag"),
        (col("accept_language") != col("accept_language").shift(1).over("customer_id")).cast(pl.Int8).fill_null(0).alias("language_change_flag"),
        (col("operating_system_type") != col("operating_system_type").shift(1).over("customer_id")).cast(pl.Int8).fill_null(0).alias("os_change_flag"),
        (col("timezone") != col("timezone").shift(1).over("customer_id")).cast(pl.Int8).fill_null(0).alias("timezone_change_flag"),
        (col("time_since_last_tx_minutes") < 5).cast(pl.Int8).alias("rapid_sequence_flag"),
        (col("device_risk_score") > 1).cast(pl.Int8).alias("suspicious_env_flag"),
    ])

    lf = lf.with_columns([
        when(col("global_mcc_freq") < 100).then(1).otherwise(0).cast(pl.Int8).alias("mcc_rare_global_flag"),
        (col("global_combination_freq") < 10).cast(pl.Int8).alias("rare_combination_flag"),
        (col("operating_system_is_new") & (s_num("operaton_amt") > 1000)).cast(pl.Int8).alias("device_change_and_large_amount_flag"),
        (col("phone_voip_call_flag") & col("mcc_is_new_for_user")).cast(pl.Int8).alias("voip_and_new_mcc_flag"),
        (col("compromised_flag") & (s_num("operaton_amt") > col("amount_mean_7d") * 2)).cast(pl.Int8).alias("compromised_and_high_amount_flag"),
        (col("session_first_tx_flag") & (s_num("operaton_amt") > 1000)).cast(pl.Int8).alias("session_first_tx_large_flag"),
        (col("timezone") - col("timezone").shift(1).over("customer_id")).abs().fill_null(0).alias("geo_jump_proxy"),
        # FIX 2: device_entropy_ratio was computed twice (Sections F & final block).
        #        Keeping only this final version.
        (col("device_diversity_30d") / col("tx_count_30d").fill_null(1)).fill_null(0).alias("device_entropy_ratio"),
        (s_str("accept_language") != s_str("browser_language")).cast(pl.Int8).alias("language_mismatch"),
        col("timezone").is_null().cast(pl.Int8).alias("timezone_mismatch"),
        ((col("event_dttm") - col("event_dttm").shift(1).over("customer_id")).dt.total_days()).fill_null(999).alias("merchant_last_seen_days"),
        ((col("event_dttm") - col("event_dttm").shift(1).over(["customer_id", "mcc_code"])).dt.total_days()).fill_null(999).alias("mcc_last_seen_days"),
    ])

    return lf.drop([
        "temp_row_idx",
        "amount_clean",
        "channel_mean",
        "channel_std",
        "mcc_mean",
        "mcc_std",
    ])


# =============================================================================
# Helper: precompute global stats from the training LazyFrame
# (call once on training data, reuse at inference time)
# =============================================================================

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
        features_lf = generate_fraud_features_v4(inference_lf, global_stats=stats)
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

    # Channel and MCC amount statistics — used by Section G to compute leakage-free
    # z-scores.  Column names match what the rest of Section G expects so that the
    # join works without any renaming.
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

    # (channel, OS) combination frequency — used for global_combination_freq /
    # rare_combination_flag in Section G.
    stats["combination_global"] = (
        train_lf
        .group_by(["channel_indicator_type", "operating_system_type"])
        .agg(pl.len().alias("global_combination_freq"))
    )

    return stats