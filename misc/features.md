# Feature Catalogue

All engineered features produced by the pipeline (`scripts/features/`).
Organized by section in the order they are computed.

Columns dropped before final output: `temp_row_idx`, `amount_clean`, `channel_mean`, `channel_std`, `mcc_mean`, `mcc_std`.
String raw columns (`mcc_code`, `accept_language`, `browser_language`, `battery`, `device_system_version`, `screen_size`, `developer_tools`, `compromised`) are excluded from the LightGBM feature set via `NON_FEATURE_COLS` in `training/config.py`.

---

## Section A — Transaction-Level (`transaction.py`)

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) extracted from `event_dttm` |
| `day_of_week` | Day of week (0=Monday … 6=Sunday) |
| `day_of_month` | Day of month (1–31) |
| `week_of_year` | ISO week number within the year |
| `hour_of_day` | Alias of `hour` |
| `is_weekend` | 1 if transaction falls on Saturday or Sunday |
| `is_night` | 1 if hour ≤ 6 |
| `is_working_hour` | 1 if hour is between 9 and 18 |
| `minutes_from_midnight` | Minutes elapsed since midnight (hour × 60 + minute) |
| `log_amount` | log1p of transaction amount; compresses large outliers |
| `amount_abs` | Absolute value of transaction amount |
| `amount_round_100` | 1 if amount is divisible by 100 (round-number flag) |
| `amount_round_1000` | 1 if amount is divisible by 1000 |
| `amount_is_integer` | 1 if amount has no fractional part |
| `amount_missing_flag` | 1 if original `operaton_amt` was null |
| `amount_currency_mismatch_flag` | 1 if `currency_iso_cd` is null |
| `amount_usd_normalized` | Placeholder — always null (reserved for future FX normalisation) |
| `is_month_start` | 1 if day ∈ {1, 2, 3} — early-month payday period |
| `is_month_end` | 1 if day ∈ {28, 29, 30, 31} — end-of-month period |
| `is_payday` | 1 if day ∈ {1, 15, 25} — common Russian salary days |
| `is_holiday` | 1 if date matches a Russian federal public holiday |

---

## Section B — Behavioural & User History (`behavioral.py`)

All features are computed cumulatively (prior rows only; current row excluded via `cum_count() - 1` shift).

### Cumulative counts

| Feature | Description |
|---------|-------------|
| `tx_count_lifetime` | Total number of prior transactions for this customer |
| `time_since_last_tx_minutes` | Minutes elapsed since the customer's previous transaction |
| `mcc_freq_user_cum` | How many times this customer has used this MCC code before |
| `pos_freq_user_cum` | How many times this customer has used this POS code before |
| `event_desc_user_freq` | How many times this customer has used this `event_desc` before |
| `event_type_user_freq` | How many times this customer has used this `event_type_nm` before |

### Derived behavioural

| Feature | Description |
|---------|-------------|
| `mcc_is_new_for_user` | 1 if this is the first occurrence of this MCC for the customer |
| `pos_cd_is_new` | 1 if this is the first occurrence of this POS code for the customer |
| `mcc_transaction_share_user` | Fraction of customer's prior transactions at this MCC |
| `pos_cd_transaction_share_user` | Fraction of customer's prior transactions at this POS |
| `merchant_switch_flag` | 1 if MCC code differs from the previous transaction |
| `time_since_last_3_tx_mean` | Rolling mean of the last 3 inter-transaction gaps (minutes) |
| `mcc_frequency_user` | Alias of `mcc_freq_user_cum` |
| `event_desc_is_new_for_user` | 1 if this `event_desc` has never appeared in this customer's history |
| `event_type_is_new_for_user` | 1 if this `event_type_nm` has never appeared in this customer's history |
| `event_desc_share_user` | Fraction of customer's prior transactions with this `event_desc` |

### Lag features (n = 1 … 5)

| Feature | Description |
|---------|-------------|
| `prev_{n}_op_type` | `event_type_nm` of the n-th previous transaction (-1 if absent) |
| `prev_{n}_op_desc` | `event_desc` of the n-th previous transaction (-1 if absent) |
| `prev_{n}_op_channel` | `channel_indicator_type` of the n-th previous transaction (-1 if absent) |
| `prev_{n}_op_subchannel` | `channel_indicator_sub_type` of the n-th previous transaction (-1 if absent) |
| `prev_1_op_timediff` | Minutes since last transaction (alias of `time_since_last_tx_minutes`) |

---

## Section D — Rolling Window Statistics (`rolling.py`)

All windows use `closed="left"` — the interval is `[t − period, t)`, so the current transaction is never included (no leakage).

### Base statistics — 9 windows × 10 metrics = 90 features

Windows: **15m**, **1h**, **6h**, **12h**, **1d**, **3d**, **7d**, **30d**, **90d**

For each window suffix `{W}` the following columns are produced:

| Feature | Description |
|---------|-------------|
| `tx_count_{W}` | Number of transactions in the window |
| `amount_mean_{W}` | Mean transaction amount in the window |
| `amount_std_{W}` | Standard deviation of amounts in the window |
| `amount_median_{W}` | Median transaction amount in the window |
| `amount_max_{W}` | Maximum transaction amount in the window |
| `amount_min_{W}` | Minimum transaction amount in the window |
| `cumulative_spend_{W}` | Total spend in the window |
| `channel_diversity_{W}` | Number of distinct channel types used in the window |
| `device_diversity_{W}` | Number of distinct OS types used in the window |
| `merchant_diversity_{W}` | Number of distinct MCC codes used in the window |

### Sub-day derived features

| Feature | Description |
|---------|-------------|
| `burst_flag_15m` | 1 if 3 or more transactions in the last 15 minutes |
| `burst_flag_1h` | 1 if 6 or more transactions in the last 1 hour |
| `spend_ratio_15m_vs_1d` | Fraction of daily spend that occurred in the last 15 minutes |
| `spend_ratio_1h_vs_1d` | Fraction of daily spend that occurred in the last 1 hour |
| `spend_ratio_6h_vs_1d` | Fraction of daily spend that occurred in the last 6 hours |
| `spend_ratio_12h_vs_1d` | Fraction of daily spend that occurred in the last 12 hours |
| `tx_count_ratio_15m_vs_1h` | Fraction of the hour's transaction count in the last 15 minutes |
| `tx_count_ratio_1h_vs_1d` | Fraction of the day's transaction count in the last 1 hour |
| `amount_ratio_to_mean_1h` | Current amount divided by the mean amount over the last 1 hour |
| `amount_ratio_to_mean_6h` | Current amount divided by the mean amount over the last 6 hours |

### Day+ derived features

| Feature | Description |
|---------|-------------|
| `amount_rank_percentile_7d` | Position of current amount in the 7d [min, max] range (0–1) |
| `amount_rank_percentile_30d` | Position of current amount in the 30d [min, max] range (0–1) |
| `amount_rank_percentile_90d` | Position of current amount in the 90d [min, max] range (0–1) |
| `amount_zscore_30d` | Z-score of current amount relative to 30d mean and std |
| `amount_ratio_to_mean_30d` | Current amount divided by the 30d mean amount |
| `avg_tx_per_day_30d` | Average number of transactions per day over last 30 days |
| `spend_velocity_1d` | Average daily spend derived from 30d total (`cumulative_spend_30d / 30`) |
| `tx_count_ratio_7d_vs_90d` | Ratio of 7d transaction count to 90d transaction count |
| `amount_mean_ratio_7d_vs_90d` | Ratio of 7d mean amount to 30d mean amount |
| `amount_diff_from_prev` | Difference between current and previous transaction amount |
| `amount_ratio_prev` | Ratio of current to previous transaction amount |
| `burst_flag` | 1 if time since previous transaction < 5 minutes |
| `spend_ratio_1d_vs_30d` | Fraction of 30d spend that occurred in the last 1 day |
| `spend_ratio_1d_vs_90d` | Fraction of 90d spend that occurred in the last 1 day |
| `spend_ratio_7d_vs_90d` | Fraction of 90d spend that occurred in the last 7 days |
| `spend_velocity_7d` | Average daily spend over last 7 days (`cumulative_spend_7d / 7`) |
| `spend_velocity_30d` | Average daily spend over last 30 days (`cumulative_spend_30d / 30`) |
| `amount_top5pct_30d` | 1 if amount is in the top 5% of the 30d min-max range |
| `amount_top1pct_90d` | 1 if amount is in the top 1% of the 90d min-max range |
| `amount_above_personal_max_flag` | 1 if amount exceeds the customer's personal maximum in the last 90 days |
| `spend_in_channel_lifetime` | Cumulative amount spent in this channel by this customer (prior transactions) |
| `tx_count_in_channel_lifetime` | Cumulative transaction count in this channel by this customer (prior transactions) |
| `channel_usage_share` | Fraction of lifetime transactions in this channel |

---

## Section E — Device & Session Risk (`device.py`)

### Device state flags

| Feature | Description |
|---------|-------------|
| `compromised_flag` | 1 if device has root/jailbreak access (`compromised = "true"`) |
| `web_rdp_connection_flag` | 1 if device is under remote desktop control |
| `developer_tools_flag` | 1 if developer settings are enabled on the device |
| `phone_voip_call_flag` | 1 if a VoIP call was active during the transaction |
| `low_battery_flag` | 1 if device battery level is below 15% |

### Device novelty flags (first-seen for this customer)

| Feature | Description |
|---------|-------------|
| `operating_system_is_new` | 1 if this OS type has never been seen for this customer |
| `os_version_is_new` | 1 if this OS version string has never been seen for this customer |
| `screen_size_is_new` | 1 if this screen resolution has never been seen for this customer |
| `timezone_is_new` | 1 if this timezone has never been seen for this customer |
| `accept_language_is_new` | 1 if this HTTP Accept-Language header has never been seen for this customer |

### Device composite & session features

| Feature | Description |
|---------|-------------|
| `device_risk_score` | Weighted sum: `compromised×5 + rdp×3 + dev_tools×2` |
| `session_tx_count` | Number of prior transactions within the same session |
| `session_amount_sum` | Total amount of prior transactions within the same session |
| `session_channel_diversity` | 1 if channel type differs from the previous transaction in this session |
| `session_length_estimate` | Alias of `session_tx_count` |
| `session_mcc_switch` | 1 if MCC code differs from the previous transaction in this session |
| `session_duration_minutes` | Minutes elapsed since the first transaction in this session |
| `session_avg_amount` | Average spend per transaction so far in this session |
| `compromised_x_amount_ratio` | `compromised_flag × (amount / 30d mean amount)` — continuous risk interaction |
| `rdp_x_session_depth` | `rdp_flag × session_tx_count` — remote session depth signal |

---

## Section F — Temporal & Global Frequencies (`temporal.py`)

### Global frequency lookups (population-level, precomputed on training data)

| Feature | Description |
|---------|-------------|
| `global_mcc_freq` | How often this MCC code appears across all training transactions |
| `global_channel_freq` | How often this channel type appears across all training transactions |
| `global_device_os_freq` | How often this OS type appears across all training transactions |
| `global_timezone_freq` | How often this timezone appears across all training transactions |
| `global_language_freq` | How often this Accept-Language header appears across all training transactions |
| `global_pos_cd_freq` | How often this POS code appears across all training transactions |
| `global_event_type_freq` | How often this event type appears across all training transactions |
| `global_event_desc_freq` | How often this event description appears across all training transactions |

*If `global_stats` is not provided, each is replaced by a per-customer cumulative count (leakage-free fallback).*

### Temporal & velocity features

| Feature | Description |
|---------|-------------|
| `circadian_deviation_score` | Absolute difference between current hour and customer's historical mean hour |
| `channel_shift_score` | 1 if channel type differs from the previous transaction |
| `velocity_change_flag` | 1 if today's transaction count exceeds 2× the customer's 30d daily average |
| `time_gap_variance_30d` | Std of inter-transaction gaps over the last 90 transactions (rolling) |
| `time_gap_mean_30d` | Mean of inter-transaction gaps over the last 90 transactions (rolling) |
| `time_gap_min_10tx` | Minimum inter-transaction gap among the last 10 gaps |
| `time_gap_cv_30d` | Coefficient of variation of gaps: `std / mean` (scale-free irregularity) |
| `new_device_flag` | 1 if this OS type is new for this customer (alias of `operating_system_is_new`) |
| `new_mcc_flag` | 1 if this MCC is new for this customer (alias of `mcc_is_new_for_user`) |
| `new_channel_flag` | 1 if this channel type is new for this customer |
| `merchant_entropy_user` | Fraction of customer's transactions at this MCC (alias of `mcc_transaction_share_user`) |
| `browser_language_mismatch` | 1 if `accept_language` ≠ `browser_language` |

### Combined risk signals

| Feature | Description |
|---------|-------------|
| `new_device_and_night_flag` | 1 if new device detected and transaction is at night |
| `rdp_and_large_amount_flag` | 1 if RDP active and amount > 1000 |
| `amount_zscore_given_channel` | Amount vs. customer's cumulative mean spend in this channel |
| `amount_zscore_given_mcc` | Amount vs. customer's cumulative mean spend at this MCC |
| `amount_zscore_given_device` | Amount vs. customer's cumulative mean spend on this OS |
| `tx_time_zscore_given_user` | Rolling std of transaction hours for this customer (temporal irregularity) |

---

## Section G — Z-Scores & Composite Flags (`zscore.py`)

### Amount z-scores (global channel/MCC baselines)

| Feature | Description |
|---------|-------------|
| `amount_zscore_channel` | Z-score of amount relative to the global mean/std for this channel |
| `amount_zscore_mcc` | Z-score of amount relative to the global mean/std for this MCC |

### Combination & session

| Feature | Description |
|---------|-------------|
| `global_combination_freq` | Population-level frequency of the (channel × OS) combination |
| `session_first_tx_flag` | 1 if this is the first transaction in the session |

### Change flags (vs. previous transaction)

| Feature | Description |
|---------|-------------|
| `language_change_flag` | 1 if `accept_language` differs from the previous transaction |
| `os_change_flag` | 1 if OS type differs from the previous transaction |
| `timezone_change_flag` | 1 if timezone differs from the previous transaction |
| `rapid_sequence_flag` | 1 if time since the last transaction < 5 minutes |
| `suspicious_env_flag` | 1 if `device_risk_score` > 1 (at least one risk indicator present) |

### Composite risk flags

| Feature | Description |
|---------|-------------|
| `mcc_rare_global_flag` | 1 if this MCC appears fewer than 100 times in the training population |
| `rare_combination_flag` | 1 if the (channel × OS) combination appears fewer than 10 times globally |
| `device_change_and_large_amount_flag` | 1 if new OS type detected and amount > 1000 |
| `voip_and_new_mcc_flag` | 1 if VoIP call active and MCC is new for this customer |
| `compromised_and_high_amount_flag` | 1 if device is compromised and amount > 2× the 7d mean |
| `session_first_tx_large_flag` | 1 if first transaction in session and amount > 1000 |

### Distance & entropy

| Feature | Description |
|---------|-------------|
| `geo_jump_proxy` | Absolute timezone difference from the previous transaction (approximate geo-movement) |
| `device_entropy_ratio` | Distinct OS types in last 30d divided by tx count in last 30d |
| `language_mismatch` | 1 if `accept_language` ≠ `browser_language` (null-safe version) |
| `timezone_mismatch` | 1 if `timezone` is null |
| `merchant_last_seen_days` | Days since the customer's previous transaction (any merchant); 999 if first |
| `mcc_last_seen_days` | Days since the customer's previous transaction at this MCC; 999 if first |

---

## Feature count summary

| Section | File | Features |
|---------|------|----------|
| A — Transaction-level | `transaction.py` | 21 |
| B — Behavioural & history | `behavioral.py` | 37 |
| D — Rolling window base stats | `rolling.py` | 90 (10 metrics × 9 windows) |
| D — Rolling window derived | `rolling.py` | 33 |
| E — Device & session | `device.py` | 20 |
| F — Temporal & global freq | `temporal.py` | 26 |
| G — Z-scores & composite | `zscore.py` | 21 |
| **Total** | | **~248** |
