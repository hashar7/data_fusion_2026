# Feature Catalogue

All engineered features produced by the pipeline (`scripts/features/`).
Organized by section in the order they are computed: A -> B -> K -> D -> E -> F -> G -> H -> I -> J.

Columns dropped before final output: `temp_row_idx`, `amount_clean`, `channel_mean`, `channel_std`, `mcc_mean`, `mcc_std`, `_fb_target`, `_fb_is_red`, `_fb_is_yellow`, `_fb_is_labeled`.
String raw columns (`mcc_code`, `accept_language`, `browser_language`, `battery`, `device_system_version`, `screen_size`, `developer_tools`, `compromised`) are excluded from the LightGBM feature set via `NON_FEATURE_COLS` in `training/config.py`.
`model_group` is a routing key only, also excluded from the model feature set.

---

## Section A -- Transaction-Level (`transaction.py`)

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0-23) extracted from `event_dttm` |
| `day_of_week` | Day of week (0=Monday ... 6=Sunday) |
| `day_of_month` | Day of month (1-31) |
| `week_of_year` | ISO week number within the year |
| `hour_of_day` | Alias of `hour` |
| `is_weekend` | 1 if transaction falls on Saturday or Sunday |
| `is_night` | 1 if hour <= 6 |
| `is_working_hour` | 1 if hour is between 9 and 18 |
| `minutes_from_midnight` | Minutes elapsed since midnight (hour x 60 + minute) |
| `log_amount` | log1p of transaction amount; compresses large outliers |
| `amount_abs` | Absolute value of transaction amount |
| `amount_round_100` | 1 if amount is divisible by 100 (round-number flag) |
| `amount_round_1000` | 1 if amount is divisible by 1000 |
| `amount_currency_mismatch_flag` | 1 if `currency_iso_cd` is null |
| `is_month_start` | 1 if day in {1, 2, 3} -- early-month payday period |
| `is_month_end` | 1 if day in {28, 29, 30, 31} -- end-of-month period |
| `is_payday` | 1 if day in {1, 15, 25} -- common Russian salary days |
| `is_holiday` | 1 if date matches a Russian federal public holiday |
| `tx_type_group` | Transaction type: 0=non-payment, 1=card, 2=P2P |
| `model_group` | Model routing key: 0=np_type7, 1=np_other, 2=card, 3=p2p (**not a model feature**) |
| `channel_type_subtype` | Combined `channel_indicator_type * 1000 + channel_indicator_sub_type` (Int32) |
| `evtype_channel` | Combined `event_type_nm * 1000 + channel_indicator_type` (Int32) |
| `evtype_subchannel` | Combined `event_type_nm * 1000 + channel_indicator_sub_type` (Int32) |
| `amount_card` | Transaction amount if card (`tx_type_group==1`), else 0; used for rolling card spend |
| `amount_p2p` | Transaction amount if P2P (`tx_type_group==2`), else 0; used for rolling P2P spend |

---

## Section B -- Behavioural & User History (`behavioral.py`)

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
| `mcc_transaction_share_user` | Fraction of customer's prior transactions at this MCC |
| `pos_cd_transaction_share_user` | Fraction of customer's prior transactions at this POS |
| `merchant_switch_flag` | 1 if MCC code differs from the previous transaction |
| `time_since_last_3_tx_mean` | Rolling mean of the last 3 inter-transaction gaps (minutes) |
| `mcc_frequency_user` | Alias of `mcc_freq_user_cum` |
| `event_desc_is_new_for_user` | 1 if this `event_desc` has never appeared in this customer's history |
| `event_type_is_new_for_user` | 1 if this `event_type_nm` has never appeared in this customer's history |
| `event_desc_share_user` | Fraction of customer's prior transactions with this `event_desc` |

### Lag features (n = 1 ... 5)

| Feature | Description |
|---------|-------------|
| `prev_{n}_op_type` | `event_type_nm` of the n-th previous transaction (-1 if absent) |
| `prev_{n}_op_desc` | `event_desc` of the n-th previous transaction (-1 if absent) |
| `prev_{n}_op_channel` | `channel_indicator_type` of the n-th previous transaction (-1 if absent) |
| `prev_{n}_op_subchannel` | `channel_indicator_sub_type` of the n-th previous transaction (-1 if absent) |
| `prev_1_op_timediff` | Minutes since last transaction (alias of `time_since_last_tx_minutes`) |

### Running-max features (leakage-free via `cum_max().shift(1)`)

| Feature | Description |
|---------|-------------|
| `operaton_amt_max_prev` | Maximum amount from all prior customer transactions |
| `operaton_amt_mcc_max_prev` | Maximum amount prior for this (customer, MCC) pair |
| `operaton_amt_type_max_prev` | Maximum amount prior for this (customer, channel_indicator_type) pair |
| `operaton_amt_desc_max_prev` | Maximum amount prior for this (customer, event_desc) pair |
| `operaton_amt_sub_max_prev` | Maximum amount prior for this (customer, channel_indicator_sub_type) pair |
| `operaton_amt_type_subtype_max_prev` | Maximum amount prior for this (customer, channel_type_subtype) pair |
| `operaton_amt_evtype_channel_max_prev` | Maximum amount prior for this (customer, evtype_channel) pair |
| `operaton_amt_evtype_subchannel_max_prev` | Maximum amount prior for this (customer, evtype_subchannel) pair |
| `operaton_amt_evtype_mcc_max_prev` | Maximum amount prior for this (customer, event_type_nm, mcc_code) triple |

### Log-frequency features (`log1p(prior_count)`)

| Feature | Description |
|---------|-------------|
| `event_desc_log_count` | log1p of `event_desc_user_freq` |
| `mcc_log_count` | log1p of `mcc_freq_user_cum` |
| `pos_cd_log_count` | log1p of `pos_freq_user_cum` |
| `event_type_nm_log_count` | log1p of `event_type_user_freq` |
| `timezone_log_count` | log1p of per-(customer, timezone) prior count |
| `operating_system_type_log_count` | log1p of per-(customer, OS) prior count |
| `channel_indicator_type_log_count` | log1p of per-(customer, channel_type) prior count |
| `channel_type_subtype_log_count` | log1p of per-(customer, channel_type_subtype) prior count |
| `evtype_channel_log_count` | log1p of per-(customer, evtype_channel) prior count |
| `evtype_subchannel_log_count` | log1p of per-(customer, evtype_subchannel) prior count |
| `evtype_mcc_log_count` | log1p of per-(customer, event_type_nm, mcc_code) prior count |
| `device_system_version_log_count` | log1p of per-(customer, OS version) prior count |
| `compromised_log_count` | log1p of per-(customer, compromised state) prior count |
| `developer_tools_log_count` | log1p of per-(customer, developer_tools) prior count |
| `browser_language_log_count` | log1p of per-(customer, browser_language) prior count |
| `currency_iso_cd_log_count` | log1p of per-(customer, currency) prior count |
| `phone_voip_call_state_log_count` | log1p of per-(customer, VoIP state) prior count |
| `web_rdp_connection_log_count` | log1p of per-(customer, RDP state) prior count |

### Same-as-previous flags

| Feature | Description |
|---------|-------------|
| `pos_cd_prev` | 1 if POS code matches the previous transaction |
| `currency_iso_cd_prev` | 1 if currency matches the previous transaction |

---

## Section K -- Dynamic Label Feedback (`feedback.py`)

Per-customer cumulative statistics from the label history (red=fraud, yellow=confirmed non-fraud).
All features use strictly prior information (`cum_sum() - current` or `shift(1) + forward_fill`).
When `labels_lf` is not provided, all features default to zero/smoothing prior.

### Per-customer label counts and rates

| Feature | Description |
|---------|-------------|
| `fb_cust_prev_red_cnt` | Count of prior fraud labels for this customer |
| `fb_cust_prev_yellow_cnt` | Count of prior non-fraud labels |
| `fb_cust_prev_labeled_cnt` | Count of prior labeled transactions |
| `fb_cust_prev_red_rate` | Smoothed fraud rate among prior labeled: `(red + 0.1) / (labeled + 1)` |
| `fb_cust_prev_yellow_rate` | Smoothed non-fraud rate among prior labeled |
| `fb_cust_prev_susp_rate` | Fraction of all prior events that were labeled |
| `fb_cust_prev_any_red` | Ever had a fraud label (binary) |
| `fb_cust_prev_any_yellow` | Ever had a non-fraud label (binary) |

### Time since last label

| Feature | Description |
|---------|-------------|
| `fb_sec_since_prev_red` | Seconds since last fraud label (-1 if none) |
| `fb_sec_since_prev_yellow` | Seconds since last non-fraud label (-1 if none) |

### Per-(customer, event_desc) label stats

| Feature | Description |
|---------|-------------|
| `fb_desc_prev_red_cnt` | Prior fraud count for (customer, event_desc) |
| `fb_desc_prev_yellow_cnt` | Prior non-fraud count for (customer, event_desc) |
| `fb_desc_prev_labeled_cnt` | Prior labeled count for (customer, event_desc) |
| `fb_desc_prev_red_rate` | Smoothed fraud rate for (customer, event_desc) |

### Per-(customer, event_type_nm) label stats

| Feature | Description |
|---------|-------------|
| `fb_type_prev_red_cnt` | Prior fraud count for (customer, event_type_nm) |
| `fb_type_prev_labeled_cnt` | Prior labeled count for (customer, event_type_nm) |
| `fb_type_prev_red_rate` | Smoothed fraud rate for (customer, event_type_nm) |

### Per-(customer, channel_indicator_sub_type) label stats

| Feature | Description |
|---------|-------------|
| `fb_subchan_prev_red_cnt` | Prior fraud count for (customer, subchannel) |
| `fb_subchan_prev_labeled_cnt` | Prior labeled count for (customer, subchannel) |
| `fb_subchan_prev_red_rate` | Smoothed fraud rate for (customer, subchannel) |

---

## Section D -- Rolling Window Statistics (`rolling.py`)

All windows use `closed="left"` -- the interval is `[t - period, t)`, so the current transaction is never included (no leakage).

### Base rolling statistics -- 9 windows x 14 metrics = 126 features

Windows: **15m**, **1h**, **6h**, **12h**, **1d**, **3d**, **7d**, **30d**, **90d**

For each window suffix `{W}` the following columns are produced:

| Feature | Description |
|---------|-------------|
| `amount_mean_{W}` | Mean transaction amount in the window |
| `amount_std_{W}` | Standard deviation of amounts in the window |
| `amount_median_{W}` | Median transaction amount in the window |
| `amount_max_{W}` | Maximum transaction amount in the window |
| `amount_min_{W}` | Minimum transaction amount in the window |
| `cumulative_spend_{W}` | Total spend in the window |
| `card_spend_{W}` | Total card-type spend in the window |
| `p2p_spend_{W}` | Total P2P-type spend in the window |
| `tx_count_{W}` | Number of transactions in the window |
| `channel_diversity_{W}` | Number of distinct channel types used in the window |
| `device_diversity_{W}` | Number of distinct OS types used in the window |
| `merchant_diversity_{W}` | Number of distinct MCC codes used in the window |
| `event_desc_diversity_{W}` | Number of distinct event_desc values used in the window |
| `event_type_diversity_{W}` | Number of distinct event_type_nm values used in the window |

### Sub-day derived features

| Feature | Description |
|---------|-------------|
| `burst_flag_1h` | 1 if more than 5 transactions in the last 1 hour |
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
| `amount_rank_percentile_7d` | Position of current amount in the 7d [min, max] range (0-1) |
| `amount_rank_percentile_30d` | Position of current amount in the 30d [min, max] range (0-1) |
| `amount_rank_percentile_90d` | Position of current amount in the 90d [min, max] range (0-1) |
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

### Per-combination cumulative lifetime stats (leakage-free)

| Feature | Description |
|---------|-------------|
| `spend_in_channel_lifetime` | Prior cumulative spend for this (customer, channel_indicator_type) |
| `tx_count_in_channel_lifetime` | Prior cumulative tx count for this (customer, channel_indicator_type) |
| `spend_in_channel_type_subtype_lifetime` | Prior cumulative spend for this (customer, channel_type_subtype) |
| `tx_count_in_channel_type_subtype_lifetime` | Prior cumulative tx count for this (customer, channel_type_subtype) |
| `spend_in_evtype_channel_lifetime` | Prior cumulative spend for this (customer, evtype_channel) |
| `tx_count_in_evtype_channel_lifetime` | Prior cumulative tx count for this (customer, evtype_channel) |
| `spend_in_evtype_subchannel_lifetime` | Prior cumulative spend for this (customer, evtype_subchannel) |
| `tx_count_in_evtype_subchannel_lifetime` | Prior cumulative tx count for this (customer, evtype_subchannel) |
| `spend_in_evtype_mcc_lifetime` | Prior cumulative spend for this (customer, event_type_nm, mcc_code) |
| `tx_count_in_evtype_mcc_lifetime` | Prior cumulative tx count for this (customer, event_type_nm, mcc_code) |

### Usage shares

| Feature | Description |
|---------|-------------|
| `channel_usage_share` | Fraction of lifetime transactions in this channel_indicator_type |
| `channel_type_subtype_usage_share` | Fraction of lifetime transactions in this channel_type_subtype |
| `evtype_channel_usage_share` | Fraction of lifetime transactions in this evtype_channel |
| `evtype_subchannel_usage_share` | Fraction of lifetime transactions in this evtype_subchannel |
| `evtype_mcc_usage_share` | Fraction of lifetime transactions in this (event_type_nm, mcc_code) |

### Card vs P2P spend fractions

| Feature | Description |
|---------|-------------|
| `card_fraction_1d` | Card spend / total spend in last 1d |
| `p2p_fraction_1d` | P2P spend / total spend in last 1d |
| `card_fraction_7d` | Card spend / total spend in last 7d |
| `card_fraction_30d` | Card spend / total spend in last 30d |
| `p2p_fraction_30d` | P2P spend / total spend in last 30d |
| `card_fraction_90d` | Card spend / total spend in last 90d |
| `p2p_fraction_90d` | P2P spend / total spend in last 90d |

### Card / P2P cross-window spend ratios

| Feature | Description |
|---------|-------------|
| `card_spend_ratio_1d_vs_30d` | Card spend 1d / card spend 30d |
| `p2p_spend_ratio_1d_vs_30d` | P2P spend 1d / P2P spend 30d |
| `card_spend_ratio_1d_vs_90d` | Card spend 1d / card spend 90d |
| `p2p_spend_ratio_1d_vs_90d` | P2P spend 1d / P2P spend 90d |
| `card_spend_ratio_7d_vs_90d` | Card spend 7d / card spend 90d |
| `p2p_spend_ratio_7d_vs_90d` | P2P spend 7d / P2P spend 90d |

### Card-to-P2P balance ratios

| Feature | Description |
|---------|-------------|
| `card_vs_p2p_ratio_1d` | Card spend 1d / P2P spend 1d |
| `card_vs_p2p_ratio_30d` | Card spend 30d / P2P spend 30d |
| `card_vs_p2p_ratio_90d` | Card spend 90d / P2P spend 90d |

---

## Section E -- Device & Session Risk (`device.py`)

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
| `device_risk_score` | Weighted sum: `compromised x 5 + rdp x 3 + dev_tools x 2` |
| `session_tx_count` | Number of prior transactions within the same session |
| `session_amount_sum` | Total amount of prior transactions within the same session |
| `session_channel_diversity` | 1 if channel type differs from the previous transaction in this session |
| `session_length_estimate` | Alias of `session_tx_count` |
| `session_mcc_switch` | 1 if MCC code differs from the previous transaction in this session |
| `session_duration_minutes` | Minutes elapsed since the first transaction in this session |
| `pause_ses` | Seconds since the previous transaction in the same session |
| `screen_w` | Screen width in pixels (parsed from `screen_size` "WxH" string) |
| `screen_h` | Screen height in pixels (parsed from `screen_size` "WxH" string) |
| `session_avg_amount` | Average spend per transaction so far in this session |
| `rdp_x_session_depth` | `rdp_flag x session_tx_count` -- remote session depth signal |

---

## Section F -- Temporal & Global Frequencies (`temporal.py`)

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
| `velocity_change_flag` | 1 if today's transaction count exceeds 2x the customer's 30d daily average |
| `time_gap_variance_30d` | Std of inter-transaction gaps over the last 90 transactions (rolling) |
| `time_gap_mean_30d` | Mean of inter-transaction gaps over the last 90 transactions (rolling) |
| `time_gap_min_10tx` | Minimum inter-transaction gap among the last 10 gaps |
| `time_gap_cv_30d` | Coefficient of variation of gaps: `std / mean` (scale-free irregularity) |
| `new_device_flag` | 1 if this OS type is new for this customer (alias of `operating_system_is_new`) |
| `merchant_entropy_user` | Fraction of customer's transactions at this MCC (`mcc_freq_user_cum / tx_count_lifetime`) |
| `browser_language_mismatch` | 1 if `accept_language` != `browser_language` |

### Combined risk signals

| Feature | Description |
|---------|-------------|
| `rdp_and_large_amount_flag` | 1 if RDP active and amount > 1000 |
| `amount_zscore_given_channel` | Amount vs. customer's cumulative mean spend in this channel |
| `amount_zscore_given_mcc` | Amount vs. customer's cumulative mean spend at this MCC |
| `amount_zscore_given_device` | Amount vs. customer's cumulative mean spend on this OS |
| `tx_time_zscore_given_user` | Rolling std of transaction hours for this customer (temporal irregularity) |

---

## Section G -- Z-Scores & Composite Flags (`zscore.py`)

### Amount z-scores (global channel/MCC baselines)

| Feature | Description |
|---------|-------------|
| `amount_zscore_channel` | Z-score of amount relative to the global mean/std for this channel |
| `amount_zscore_mcc` | Z-score of amount relative to the global mean/std for this MCC |

### Combination & session

| Feature | Description |
|---------|-------------|
| `global_combination_freq` | Population-level frequency of the (channel x OS) combination |
| `session_first_tx_flag` | 1 if this is the first transaction in the session |

### Change flags (vs. previous transaction)

| Feature | Description |
|---------|-------------|
| `language_change_flag` | 1 if `accept_language` differs from the previous transaction |
| `os_change_flag` | 1 if OS type differs from the previous transaction |
| `timezone_change_flag` | 1 if timezone differs from the previous transaction |
| `rapid_sequence_flag` | 1 if time since the last transaction < 5 minutes |

### Composite risk flags

| Feature | Description |
|---------|-------------|
| `device_change_and_large_amount_flag` | 1 if new OS type detected and amount > 1000 |
| `session_first_tx_large_flag` | 1 if first transaction in session and amount > 1000 |

### Distance & entropy

| Feature | Description |
|---------|-------------|
| `geo_jump_proxy` | Absolute timezone difference from the previous transaction (approximate geo-movement) |
| `device_entropy_ratio` | Distinct OS types in last 30d divided by tx count in last 30d |
| `language_mismatch` | 1 if `accept_language` != `browser_language` (null-safe version) |
| `merchant_last_seen_days` | Days since the customer's previous transaction (any merchant); 999 if first |
| `mcc_last_seen_days` | Days since the customer's previous transaction at this MCC; 999 if first |

---

## Section H -- Per-Category Cumulative Stats (`category_stats.py`)

Leakage-free per-user statistics for key categorical dimensions.
Prior mean/std use `cum_sum().shift(1) / cum_count().shift(1)` to exclude the current row.

### Per-(customer, event_desc)

| Feature | Description |
|---------|-------------|
| `event_desc_spend_user` | Prior cumulative spend for this (customer, event_desc) pair |
| `event_desc_amount_mean_user` | Prior mean amount for this event_desc |
| `event_desc_amount_std_user` | Prior std of amount for this event_desc |
| `event_desc_last_seen_days` | Days since last transaction with this event_desc; 999 if first |
| `amount_zscore_given_event_desc` | Z-score of amount vs user's prior history for this event_desc |
| `event_desc_spend_share_user` | Fraction of prior lifetime spend that went to this event_desc |
| `event_desc_spend_vs_90d` | Lifetime event_desc spend / 90d cumulative spend |

### Per-(customer, event_type_nm)

| Feature | Description |
|---------|-------------|
| `event_type_spend_user` | Prior cumulative spend for this event_type |
| `event_type_amount_mean_user` | Prior mean amount for this event_type |
| `event_type_amount_std_user` | Prior std of amount for this event_type |
| `event_type_last_seen_days` | Days since last transaction with this event_type; 999 if first |
| `amount_zscore_given_event_type` | Z-score of amount vs user's prior for this event_type |
| `event_type_spend_share_user` | Fraction of prior lifetime spend for this event_type |
| `event_type_spend_vs_90d` | Lifetime event_type spend / 90d cumulative spend |

### Per-(customer, channel_indicator_sub_type)

| Feature | Description |
|---------|-------------|
| `spend_in_subchannel_lifetime` | Prior cumulative spend for this subchannel |
| `tx_count_in_subchannel_lifetime` | Prior cumulative tx count for this subchannel |
| `amount_mean_subchannel_user` | Prior mean amount for this subchannel |
| `amount_std_subchannel_user` | Prior std of amount for this subchannel |
| `subchannel_last_seen_days` | Days since last transaction with this subchannel; 999 if first |
| `subchannel_usage_share` | Fraction of lifetime transactions in this subchannel |
| `channel_indicator_sub_type_log_count` | log1p of `tx_count_in_subchannel_lifetime` |
| `is_new_subchannel_for_user` | 1 if this subchannel has never been seen for this customer |
| `amount_zscore_given_subchannel` | Z-score of amount vs user's prior for this subchannel |
| `subchannel_spend_vs_90d` | Lifetime subchannel spend / 90d cumulative spend |

### Per-(customer, channel_type_subtype)

| Feature | Description |
|---------|-------------|
| `amount_mean_channel_type_subtype_user` | Prior mean amount for this type+subtype combo |
| `amount_std_channel_type_subtype_user` | Prior std of amount for this type+subtype combo |
| `channel_type_subtype_last_seen_days` | Days since last transaction with this type+subtype; 999 if first |
| `is_new_channel_type_subtype_for_user` | 1 if this type+subtype is new for this customer |
| `amount_zscore_given_channel_type_subtype` | Z-score of amount vs prior for this type+subtype |
| `channel_type_subtype_spend_vs_90d` | Lifetime type+subtype spend / 90d cumulative spend |

### Per-(customer, evtype_channel)

| Feature | Description |
|---------|-------------|
| `amount_mean_evtype_channel_user` | Prior mean amount for this evtype+channel combo |
| `amount_std_evtype_channel_user` | Prior std of amount for this evtype+channel combo |
| `evtype_channel_last_seen_days` | Days since last transaction with this evtype+channel; 999 if first |
| `is_new_evtype_channel_for_user` | 1 if this evtype+channel is new for this customer |
| `amount_zscore_given_evtype_channel` | Z-score of amount vs prior for this evtype+channel |
| `evtype_channel_spend_vs_90d` | Lifetime evtype+channel spend / 90d cumulative spend |

### Per-(customer, evtype_subchannel)

| Feature | Description |
|---------|-------------|
| `amount_mean_evtype_subchannel_user` | Prior mean amount for this evtype+subchannel combo |
| `amount_std_evtype_subchannel_user` | Prior std of amount for this evtype+subchannel combo |
| `evtype_subchannel_last_seen_days` | Days since last with this evtype+subchannel; 999 if first |
| `is_new_evtype_subchannel_for_user` | 1 if this evtype+subchannel is new for this customer |
| `amount_zscore_given_evtype_subchannel` | Z-score of amount vs prior for this evtype+subchannel |
| `evtype_subchannel_spend_vs_90d` | Lifetime evtype+subchannel spend / 90d cumulative spend |

### Per-(customer, event_type_nm, mcc_code)

| Feature | Description |
|---------|-------------|
| `amount_mean_evtype_mcc_user` | Prior mean amount for this evtype+MCC combo |
| `amount_std_evtype_mcc_user` | Prior std of amount for this evtype+MCC combo |
| `evtype_mcc_last_seen_days` | Days since last with this evtype+MCC; 999 if first |
| `is_new_evtype_mcc_for_user` | 1 if this evtype+MCC is new for this customer |
| `amount_zscore_given_evtype_mcc` | Z-score of amount vs prior for this evtype+MCC |
| `evtype_mcc_spend_vs_90d` | Lifetime evtype+MCC spend / 90d cumulative spend |

### Per-(customer, pos_cd)

| Feature | Description |
|---------|-------------|
| `spend_in_pos_lifetime` | Prior cumulative spend for this POS code |
| `amount_mean_pos_user` | Prior mean amount for this POS code |
| `amount_std_pos_user` | Prior std of amount for this POS code |
| `pos_last_seen_days` | Days since last transaction with this POS code; 999 if first |
| `amount_zscore_given_pos` | Z-score of amount vs prior for this POS code |
| `pos_spend_vs_90d` | Lifetime POS spend / 90d cumulative spend |

### Per-(customer, tx_type_group)

| Feature | Description |
|---------|-------------|
| `spend_in_tx_type_lifetime` | Prior cumulative spend for this tx_type_group |
| `tx_count_in_tx_type_lifetime` | Prior cumulative tx count for this tx_type_group |
| `amount_mean_tx_type_user` | Prior mean amount for this tx_type_group |
| `amount_std_tx_type_user` | Prior std of amount for this tx_type_group |
| `tx_type_usage_share` | Fraction of lifetime transactions in this tx_type_group |
| `amount_zscore_given_tx_type` | Z-score of amount vs prior for this tx_type_group |
| `tx_type_spend_share` | Fraction of prior lifetime spend for this tx_type_group |
| `tx_type_spend_vs_90d` | Lifetime tx_type spend / 90d cumulative spend |

### Cross-category novelty flags

| Feature | Description |
|---------|-------------|
| `is_new_channel_desc_combo` | 1 if (channel_type, event_desc) combination is new for this customer |
| `is_new_channel_type_combo` | 1 if (channel_type, event_type_nm) combination is new |
| `is_new_subchannel_type_combo` | 1 if (subchannel, event_type_nm) combination is new |
| `is_new_type_desc_combo` | 1 if (event_type_nm, event_desc) combination is new |
| `is_new_txtype_channel_combo` | 1 if (tx_type_group, channel_type) combination is new |

### Global (population-level) z-scores

Available when `global_stats` is provided; otherwise constant 0.

| Feature | Description |
|---------|-------------|
| `amount_zscore_event_desc_global` | Z-score of amount vs global population mean/std for this event_desc |
| `amount_zscore_event_type_global` | Z-score of amount vs global mean/std for this event_type |
| `amount_zscore_subchannel_global` | Z-score of amount vs global mean/std for this subchannel |
| `amount_zscore_pos_global` | Z-score of amount vs global mean/std for this POS code |
| `global_subchannel_freq` | Global frequency of this subchannel across training population |
| `amount_zscore_channel_type_subtype_global` | Z-score of amount vs global mean/std for this channel_type_subtype |
| `global_channel_type_subtype_freq` | Global frequency of this channel_type_subtype |
| `amount_zscore_evtype_channel_global` | Z-score of amount vs global mean/std for this evtype_channel |
| `global_evtype_channel_freq` | Global frequency of this evtype_channel |
| `amount_zscore_evtype_subchannel_global` | Z-score of amount vs global mean/std for this evtype_subchannel |
| `global_evtype_subchannel_freq` | Global frequency of this evtype_subchannel |
| `amount_zscore_evtype_mcc_global` | Z-score of amount vs global mean/std for this (event_type, mcc_code) |
| `global_evtype_mcc_freq` | Global frequency of this (event_type, mcc_code) |

---

## Section I -- Bayesian Target Encodings (`category_stats.py`)

Precomputed on the labeled training set via `compute_global_stats()`. Each feature is the Bayesian-smoothed fraud rate for that category value: `smoothed = (fraud_count + alpha * global_rate) / (labeled_count + alpha)`, alpha=20. Unseen values at test time are filled with the stored global fraud rate; 0.5 when no `global_stats` provided.

| Feature | Join key | Description |
|---------|----------|-------------|
| `event_type_nm_target_enc` | `event_type_nm` | Smoothed fraud rate per event type |
| `event_desc_target_enc` | `event_desc` | Smoothed fraud rate per event description |
| `channel_type_target_enc` | `channel_indicator_type` | Smoothed fraud rate per channel type |
| `channel_subtype_target_enc` | `channel_indicator_sub_type` | Smoothed fraud rate per subchannel |
| `mcc_target_enc` | `mcc_code` | Smoothed fraud rate per MCC; null (non-card) filled with global rate |
| `channel_type_subtype_target_enc` | `channel_type_subtype` | Smoothed fraud rate per (type x subtype) |
| `evtype_channel_target_enc` | `evtype_channel` | Smoothed fraud rate per (event_type x channel) |
| `evtype_subchannel_target_enc` | `evtype_subchannel` | Smoothed fraud rate per (event_type x subchannel) |
| `type_desc_pair_target_enc` | `(event_type_nm, event_desc)` | Pair-level fraud rate -- strongest single signal |
| `channel_type_fraud_rate_within_group` | `(tx_type_group, channel_indicator_type)` | Channel fraud rate conditional on tx_type_group |
| `channel_subtype_fraud_rate_within_group` | `(tx_type_group, channel_indicator_sub_type)` | Subtype fraud rate conditional on tx_type_group |
| `evtype_mcc_target_enc` | `(event_type_nm, mcc_code)` | Per-(event_type x MCC) smoothed fraud rate |

---

## Section J -- Binary Risk Flags (`category_risk.py`)

Hardcoded from empirical fraud-rate analysis. No `global_stats` dependency.

| Feature | Condition |
|---------|-----------|
| `is_very_high_risk_desc` | `event_desc` in {60, 41, 27, 73, 103} -- all >89% fraud |
| `is_high_risk_desc` | `event_desc` in {68, 29, 31, 119, 113, 109, 37} -- all >67% fraud |
| `is_low_risk_desc` | `event_desc` in {51, 5, 4, 69, 97, 32} -- all <20% fraud |
| `is_high_risk_channel` | `channel_indicator_type == 6` OR `channel_indicator_sub_type == 11` |
| `is_p2p_danger_channel` | `tx_type_group == 2` AND `channel_indicator_sub_type == 5` (90.5% fraud in P2P) |
| `is_near_certain_fraud_pair` | `(event_type_nm, event_desc)` in {(14,60), (14,41), (14,73)} -- 93-100% fraud |

---

## Feature count summary

| Section | File | Features |
|---------|------|----------|
| A -- Transaction-level | `transaction.py` | 25 (incl. model_group routing key) |
| B -- Behavioural & history | `behavioral.py` | 64 (6 counts + 8 derived + 21 lags + 9 running-max + 18 log-freq + 2 same-as-prev) |
| K -- Label feedback | `feedback.py` | 20 |
| D -- Rolling window base stats | `rolling.py` | 126 (14 metrics x 9 windows) |
| D -- Rolling window derived | `rolling.py` | 60 (9 sub-day + 20 day+ + 10 lifetime + 5 usage + 7 card/p2p frac + 6 cross-window + 3 balance) |
| E -- Device & session | `device.py` | 22 |
| F -- Temporal & global freq | `temporal.py` | 23 |
| G -- Z-scores & composite | `zscore.py` | 15 |
| H -- Per-category stats | `category_stats.py` | 67 (7+7+10+6+6+6+6+6+8+5 cumulative) |
| H -- Global z-scores | `category_stats.py` | 13 |
| I -- Target encodings | `category_stats.py` | 12 |
| J -- Binary risk flags | `category_risk.py` | 6 |
| **Total** | | **~453** |

*Minus 1 `model_group` (non-feature routing key) and 1 `amount_clean` (dropped intermediate) = ~451 usable feature columns.*

---

## Feature blacklist (zero-gain, excluded from model input)

These features were removed from the code or blacklisted in `config.py` after showing zero split gain:

`compromised_and_high_amount_flag`, `timezone_mismatch`, `burst_flag_15m`, `voip_and_new_mcc_flag`, `suspicious_env_flag`, `mcc_rare_global_flag`, `rare_combination_flag`, `compromised_x_amount_ratio`, `pos_cd_is_new`, `mcc_is_new_for_user`, `amount_usd_normalized`, `amount_missing_flag`, `new_device_and_night_flag`, `new_mcc_flag`, `new_channel_flag`.

Note: `compromised_flag` and `developer_tools_flag` are still produced by `device.py` but are in the blacklist. All other blacklisted features have been removed from the pipeline entirely.
