# Каталог признаков

Все инженерные признаки, производимые пайплайном (`scripts/features/`).
Организованы по секциям в порядке вычисления: A -> B -> K -> D -> E -> F -> G -> H -> I -> J.

Колонки, удаляемые перед финальным выводом: `temp_row_idx`, `amount_clean`, `channel_mean`, `channel_std`, `mcc_mean`, `mcc_std`, `_fb_target`, `_fb_is_red`, `_fb_is_yellow`, `_fb_is_labeled`.
Строковые исходные колонки (`mcc_code`, `accept_language`, `browser_language`, `battery`, `device_system_version`, `screen_size`, `developer_tools`, `compromised`) исключены из набора признаков LightGBM через `NON_FEATURE_COLS` в `training/config.py`.
`model_group` — только ключ маршрутизации, также исключён из набора признаков модели.

---

## Секция A -- Уровень транзакции (`transaction.py`)

| Признак | Описание |
|---------|----------|
| `hour` | Час суток (0-23), извлечённый из `event_dttm` |
| `day_of_week` | День недели (0=понедельник ... 6=воскресенье) |
| `day_of_month` | День месяца (1-31) |
| `week_of_year` | ISO-номер недели в году |
| `hour_of_day` | Псевдоним `hour` |
| `is_weekend` | 1, если транзакция совершена в субботу или воскресенье |
| `is_night` | 1, если час <= 6 |
| `is_working_hour` | 1, если час от 9 до 18 |
| `minutes_from_midnight` | Минут с начала суток (hour x 60 + minute) |
| `log_amount` | log1p суммы транзакции; сжимает крупные выбросы |
| `amount_abs` | Абсолютное значение суммы транзакции |
| `amount_round_100` | 1, если сумма делится на 100 (признак круглой суммы) |
| `amount_round_1000` | 1, если сумма делится на 1000 |
| `amount_currency_mismatch_flag` | 1, если `currency_iso_cd` равно null |
| `is_month_start` | 1, если день входит в {1, 2, 3} — начало месяца / зарплатный период |
| `is_month_end` | 1, если день входит в {28, 29, 30, 31} — конец месяца |
| `is_payday` | 1, если день входит в {1, 15, 25} — типичные дни выплаты зарплаты в России |
| `is_holiday` | 1, если дата совпадает с федеральным праздником России |
| `tx_type_group` | Тип транзакции: 0=non-payment, 1=card, 2=P2P |
| `model_group` | Ключ маршрутизации модели: 0=np_type7, 1=np_other, 2=card, 3=p2p (**не является признаком модели**) |
| `channel_type_subtype` | Комбинация `channel_indicator_type * 1000 + channel_indicator_sub_type` (Int32) |
| `evtype_channel` | Комбинация `event_type_nm * 1000 + channel_indicator_type` (Int32) |
| `evtype_subchannel` | Комбинация `event_type_nm * 1000 + channel_indicator_sub_type` (Int32) |
| `amount_card` | Сумма транзакции для card (`tx_type_group==1`), иначе 0; используется для rolling card spend |
| `amount_p2p` | Сумма транзакции для P2P (`tx_type_group==2`), иначе 0; используется для rolling P2P spend |

---

## Секция B -- Поведение и история клиента (`behavioral.py`)

Все признаки вычисляются накопительно (только предыдущие строки; текущая строка исключена через сдвиг `cum_count() - 1`).

### Накопительные счётчики

| Признак | Описание |
|---------|----------|
| `tx_count_lifetime` | Суммарное число предыдущих транзакций данного клиента |
| `time_since_last_tx_minutes` | Минут с момента предыдущей транзакции клиента |
| `mcc_freq_user_cum` | Сколько раз клиент использовал этот MCC ранее |
| `pos_freq_user_cum` | Сколько раз клиент использовал этот POS ранее |
| `event_desc_user_freq` | Сколько раз клиент использовал этот `event_desc` ранее |
| `event_type_user_freq` | Сколько раз клиент использовал этот `event_type_nm` ранее |

### Производные поведенческие признаки

| Признак | Описание |
|---------|----------|
| `mcc_transaction_share_user` | Доля предыдущих транзакций клиента в данном MCC |
| `pos_cd_transaction_share_user` | Доля предыдущих транзакций клиента в данном POS |
| `merchant_switch_flag` | 1, если MCC отличается от предыдущей транзакции |
| `time_since_last_3_tx_mean` | Скользящее среднее последних 3 межтранзакционных интервалов (минуты) |
| `mcc_frequency_user` | Псевдоним `mcc_freq_user_cum` |
| `event_desc_is_new_for_user` | 1, если этот `event_desc` ни разу не встречался в истории клиента |
| `event_type_is_new_for_user` | 1, если этот `event_type_nm` ни разу не встречался в истории клиента |
| `event_desc_share_user` | Доля предыдущих транзакций клиента с данным `event_desc` |

### Лаговые признаки (n = 1 ... 5)

| Признак | Описание |
|---------|----------|
| `prev_{n}_op_type` | `event_type_nm` n-й предыдущей транзакции (-1, если отсутствует) |
| `prev_{n}_op_desc` | `event_desc` n-й предыдущей транзакции (-1, если отсутствует) |
| `prev_{n}_op_channel` | `channel_indicator_type` n-й предыдущей транзакции (-1, если отсутствует) |
| `prev_{n}_op_subchannel` | `channel_indicator_sub_type` n-й предыдущей транзакции (-1, если отсутствует) |
| `prev_1_op_timediff` | Минут с последней транзакции (псевдоним `time_since_last_tx_minutes`) |

### Running-max признаки (без утечки данных, через `cum_max().shift(1)`)

| Признак | Описание |
|---------|----------|
| `operaton_amt_max_prev` | Максимальная сумма среди всех предыдущих транзакций клиента |
| `operaton_amt_mcc_max_prev` | Максимальная сумма ранее для пары (клиент, MCC) |
| `operaton_amt_type_max_prev` | Максимальная сумма ранее для пары (клиент, channel_indicator_type) |
| `operaton_amt_desc_max_prev` | Максимальная сумма ранее для пары (клиент, event_desc) |
| `operaton_amt_sub_max_prev` | Максимальная сумма ранее для пары (клиент, channel_indicator_sub_type) |
| `operaton_amt_type_subtype_max_prev` | Максимальная сумма ранее для пары (клиент, channel_type_subtype) |
| `operaton_amt_evtype_channel_max_prev` | Максимальная сумма ранее для пары (клиент, evtype_channel) |
| `operaton_amt_evtype_subchannel_max_prev` | Максимальная сумма ранее для пары (клиент, evtype_subchannel) |
| `operaton_amt_evtype_mcc_max_prev` | Максимальная сумма ранее для тройки (клиент, event_type_nm, mcc_code) |

### Log-частотные признаки (`log1p(prior_count)`)

| Признак | Описание |
|---------|----------|
| `event_desc_log_count` | log1p от `event_desc_user_freq` |
| `mcc_log_count` | log1p от `mcc_freq_user_cum` |
| `pos_cd_log_count` | log1p от `pos_freq_user_cum` |
| `event_type_nm_log_count` | log1p от `event_type_user_freq` |
| `timezone_log_count` | log1p от числа предыдущих вхождений (клиент, timezone) |
| `operating_system_type_log_count` | log1p от числа предыдущих вхождений (клиент, OS) |
| `channel_indicator_type_log_count` | log1p от числа предыдущих вхождений (клиент, channel_type) |
| `channel_type_subtype_log_count` | log1p от числа предыдущих вхождений (клиент, channel_type_subtype) |
| `evtype_channel_log_count` | log1p от числа предыдущих вхождений (клиент, evtype_channel) |
| `evtype_subchannel_log_count` | log1p от числа предыдущих вхождений (клиент, evtype_subchannel) |
| `evtype_mcc_log_count` | log1p от числа предыдущих вхождений (клиент, event_type_nm, mcc_code) |
| `device_system_version_log_count` | log1p от числа предыдущих вхождений (клиент, версия OS) |
| `compromised_log_count` | log1p от числа предыдущих вхождений (клиент, состояние compromised) |
| `developer_tools_log_count` | log1p от числа предыдущих вхождений (клиент, developer_tools) |
| `browser_language_log_count` | log1p от числа предыдущих вхождений (клиент, browser_language) |
| `currency_iso_cd_log_count` | log1p от числа предыдущих вхождений (клиент, валюта) |
| `phone_voip_call_state_log_count` | log1p от числа предыдущих вхождений (клиент, VoIP state) |
| `web_rdp_connection_log_count` | log1p от числа предыдущих вхождений (клиент, RDP state) |

### Флаги совпадения с предыдущей транзакцией

| Признак | Описание |
|---------|----------|
| `pos_cd_prev` | 1, если POS совпадает с предыдущей транзакцией |
| `currency_iso_cd_prev` | 1, если валюта совпадает с предыдущей транзакцией |

---

## Секция K -- Динамическая обратная связь по меткам (`feedback.py`)

Накопительная статистика по клиенту на основе истории меток (red=мошенничество, yellow=подтверждённая легитимная транзакция).
Все признаки используют строго предшествующую информацию (`cum_sum() - current` или `shift(1) + forward_fill`).
Если `labels_lf` не передан, все признаки по умолчанию равны нулю / значению сглаживающего prior.

### Счётчики и ставки меток по клиенту

| Признак | Описание |
|---------|----------|
| `fb_cust_prev_red_cnt` | Число предыдущих меток мошенничества у данного клиента |
| `fb_cust_prev_yellow_cnt` | Число предыдущих меток подтверждённых легитимных транзакций |
| `fb_cust_prev_labeled_cnt` | Число предыдущих размеченных транзакций |
| `fb_cust_prev_red_rate` | Сглаженная частота мошенничества среди размеченных: `(red + 0.1) / (labeled + 1)` |
| `fb_cust_prev_yellow_rate` | Сглаженная частота легитимных среди размеченных |
| `fb_cust_prev_susp_rate` | Доля всех предыдущих событий, которые были размечены |
| `fb_cust_prev_any_red` | Хотя бы одна метка мошенничества была ранее (бинарный) |
| `fb_cust_prev_any_yellow` | Хотя бы одна метка легитимной транзакции была ранее (бинарный) |

### Время с момента последней метки

| Признак | Описание |
|---------|----------|
| `fb_sec_since_prev_red` | Секунд с момента последней метки мошенничества (-1, если не было) |
| `fb_sec_since_prev_yellow` | Секунд с момента последней легитимной метки (-1, если не было) |

### Статистики меток по (клиент, event_desc)

| Признак | Описание |
|---------|----------|
| `fb_desc_prev_red_cnt` | Число предыдущих меток мошенничества для (клиент, event_desc) |
| `fb_desc_prev_yellow_cnt` | Число предыдущих легитимных меток для (клиент, event_desc) |
| `fb_desc_prev_labeled_cnt` | Число предыдущих размеченных записей для (клиент, event_desc) |
| `fb_desc_prev_red_rate` | Сглаженная частота мошенничества для (клиент, event_desc) |

### Статистики меток по (клиент, event_type_nm)

| Признак | Описание |
|---------|----------|
| `fb_type_prev_red_cnt` | Число предыдущих меток мошенничества для (клиент, event_type_nm) |
| `fb_type_prev_labeled_cnt` | Число предыдущих размеченных записей для (клиент, event_type_nm) |
| `fb_type_prev_red_rate` | Сглаженная частота мошенничества для (клиент, event_type_nm) |

### Статистики меток по (клиент, channel_indicator_sub_type)

| Признак | Описание |
|---------|----------|
| `fb_subchan_prev_red_cnt` | Число предыдущих меток мошенничества для (клиент, subchannel) |
| `fb_subchan_prev_labeled_cnt` | Число предыдущих размеченных записей для (клиент, subchannel) |
| `fb_subchan_prev_red_rate` | Сглаженная частота мошенничества для (клиент, subchannel) |

---

## Секция D -- Статистики скользящего окна (`rolling.py`)

Все окна используют `closed="left"` — интервал `[t - period, t)`, поэтому текущая транзакция никогда не включается в своё окно (нет утечки данных).

### Базовые rolling-статистики -- 9 окон x 14 метрик = 126 признаков

Окна: **15m**, **1h**, **6h**, **12h**, **1d**, **3d**, **7d**, **30d**, **90d**

Для каждого суффикса окна `{W}` производятся следующие колонки:

| Признак | Описание |
|---------|----------|
| `amount_mean_{W}` | Среднее значение суммы транзакций в окне |
| `amount_std_{W}` | Стандартное отклонение суммы в окне |
| `amount_median_{W}` | Медиана суммы транзакций в окне |
| `amount_max_{W}` | Максимальная сумма транзакции в окне |
| `amount_min_{W}` | Минимальная сумма транзакции в окне |
| `cumulative_spend_{W}` | Суммарные траты в окне |
| `card_spend_{W}` | Суммарные card-траты в окне |
| `p2p_spend_{W}` | Суммарные P2P-траты в окне |
| `tx_count_{W}` | Число транзакций в окне |
| `channel_diversity_{W}` | Число уникальных типов каналов в окне |
| `device_diversity_{W}` | Число уникальных типов OS в окне |
| `merchant_diversity_{W}` | Число уникальных MCC кодов в окне |
| `event_desc_diversity_{W}` | Число уникальных значений event_desc в окне |
| `event_type_diversity_{W}` | Число уникальных значений event_type_nm в окне |

### Производные признаки субдневного масштаба

| Признак | Описание |
|---------|----------|
| `burst_flag_1h` | 1, если за последний час совершено более 5 транзакций |
| `spend_ratio_15m_vs_1d` | Доля дневных трат, совершённых за последние 15 минут |
| `spend_ratio_1h_vs_1d` | Доля дневных трат, совершённых за последний час |
| `spend_ratio_6h_vs_1d` | Доля дневных трат, совершённых за последние 6 часов |
| `spend_ratio_12h_vs_1d` | Доля дневных трат, совершённых за последние 12 часов |
| `tx_count_ratio_15m_vs_1h` | Доля часовых транзакций, совершённых за последние 15 минут |
| `tx_count_ratio_1h_vs_1d` | Доля дневных транзакций, совершённых за последний час |
| `amount_ratio_to_mean_1h` | Текущая сумма, делённая на среднюю сумму за последний час |
| `amount_ratio_to_mean_6h` | Текущая сумма, делённая на среднюю сумму за последние 6 часов |

### Производные признаки масштаба день+

| Признак | Описание |
|---------|----------|
| `amount_rank_percentile_7d` | Позиция текущей суммы в диапазоне [min, max] за 7 дней (0-1) |
| `amount_rank_percentile_30d` | Позиция текущей суммы в диапазоне [min, max] за 30 дней (0-1) |
| `amount_rank_percentile_90d` | Позиция текущей суммы в диапазоне [min, max] за 90 дней (0-1) |
| `amount_zscore_30d` | Z-score текущей суммы относительно среднего и std за 30 дней |
| `amount_ratio_to_mean_30d` | Текущая сумма, делённая на среднюю за 30 дней |
| `avg_tx_per_day_30d` | Среднее число транзакций в день за последние 30 дней |
| `spend_velocity_1d` | Средние дневные траты из 30-дневного итога (`cumulative_spend_30d / 30`) |
| `tx_count_ratio_7d_vs_90d` | Соотношение числа транзакций за 7 дней к числу за 90 дней |
| `amount_mean_ratio_7d_vs_90d` | Соотношение средней суммы за 7 дней к средней за 30 дней |
| `amount_diff_from_prev` | Разница между текущей и предыдущей суммой транзакции |
| `amount_ratio_prev` | Отношение текущей суммы к предыдущей |
| `burst_flag` | 1, если с предыдущей транзакции прошло менее 5 минут |
| `spend_ratio_1d_vs_30d` | Доля 30-дневных трат, совершённых за последние сутки |
| `spend_ratio_1d_vs_90d` | Доля 90-дневных трат, совершённых за последние сутки |
| `spend_ratio_7d_vs_90d` | Доля 90-дневных трат, совершённых за последние 7 дней |
| `spend_velocity_7d` | Средние дневные траты за последние 7 дней (`cumulative_spend_7d / 7`) |
| `spend_velocity_30d` | Средние дневные траты за последние 30 дней (`cumulative_spend_30d / 30`) |
| `amount_top5pct_30d` | 1, если сумма входит в топ 5% диапазона min-max за 30 дней |
| `amount_top1pct_90d` | 1, если сумма входит в топ 1% диапазона min-max за 90 дней |
| `amount_above_personal_max_flag` | 1, если сумма превышает личный максимум клиента за последние 90 дней |

### Накопительная статистика по комбинациям (без утечки данных)

| Признак | Описание |
|---------|----------|
| `spend_in_channel_lifetime` | Предыдущие накопленные траты для (клиент, channel_indicator_type) |
| `tx_count_in_channel_lifetime` | Предыдущее накопленное число транзакций для (клиент, channel_indicator_type) |
| `spend_in_channel_type_subtype_lifetime` | Предыдущие накопленные траты для (клиент, channel_type_subtype) |
| `tx_count_in_channel_type_subtype_lifetime` | Предыдущее накопленное число транзакций для (клиент, channel_type_subtype) |
| `spend_in_evtype_channel_lifetime` | Предыдущие накопленные траты для (клиент, evtype_channel) |
| `tx_count_in_evtype_channel_lifetime` | Предыдущее накопленное число транзакций для (клиент, evtype_channel) |
| `spend_in_evtype_subchannel_lifetime` | Предыдущие накопленные траты для (клиент, evtype_subchannel) |
| `tx_count_in_evtype_subchannel_lifetime` | Предыдущее накопленное число транзакций для (клиент, evtype_subchannel) |
| `spend_in_evtype_mcc_lifetime` | Предыдущие накопленные траты для (клиент, event_type_nm, mcc_code) |
| `tx_count_in_evtype_mcc_lifetime` | Предыдущее накопленное число транзакций для (клиент, event_type_nm, mcc_code) |

### Доли использования

| Признак | Описание |
|---------|----------|
| `channel_usage_share` | Доля транзакций за всё время в данном channel_indicator_type |
| `channel_type_subtype_usage_share` | Доля транзакций за всё время в данном channel_type_subtype |
| `evtype_channel_usage_share` | Доля транзакций за всё время в данном evtype_channel |
| `evtype_subchannel_usage_share` | Доля транзакций за всё время в данном evtype_subchannel |
| `evtype_mcc_usage_share` | Доля транзакций за всё время в данной паре (event_type_nm, mcc_code) |

### Доли трат: card vs P2P

| Признак | Описание |
|---------|----------|
| `card_fraction_1d` | Card spend / суммарные траты за последние 1 д |
| `p2p_fraction_1d` | P2P spend / суммарные траты за последние 1 д |
| `card_fraction_7d` | Card spend / суммарные траты за последние 7 д |
| `card_fraction_30d` | Card spend / суммарные траты за последние 30 д |
| `p2p_fraction_30d` | P2P spend / суммарные траты за последние 30 д |
| `card_fraction_90d` | Card spend / суммарные траты за последние 90 д |
| `p2p_fraction_90d` | P2P spend / суммарные траты за последние 90 д |

### Кросс-оконные соотношения трат card / P2P

| Признак | Описание |
|---------|----------|
| `card_spend_ratio_1d_vs_30d` | Card spend 1д / card spend 30д |
| `p2p_spend_ratio_1d_vs_30d` | P2P spend 1д / P2P spend 30д |
| `card_spend_ratio_1d_vs_90d` | Card spend 1д / card spend 90д |
| `p2p_spend_ratio_1d_vs_90d` | P2P spend 1д / P2P spend 90д |
| `card_spend_ratio_7d_vs_90d` | Card spend 7д / card spend 90д |
| `p2p_spend_ratio_7d_vs_90d` | P2P spend 7д / P2P spend 90д |

### Соотношения балансов card / P2P

| Признак | Описание |
|---------|----------|
| `card_vs_p2p_ratio_1d` | Card spend 1д / P2P spend 1д |
| `card_vs_p2p_ratio_30d` | Card spend 30д / P2P spend 30д |
| `card_vs_p2p_ratio_90d` | Card spend 90д / P2P spend 90д |

---

## Секция E -- Риски устройства и сессии (`device.py`)

### Флаги состояния устройства

| Признак | Описание |
|---------|----------|
| `compromised_flag` | 1, если на устройстве есть root/jailbreak (`compromised = "true"`) |
| `web_rdp_connection_flag` | 1, если устройство управляется удалённо (RDP) |
| `developer_tools_flag` | 1, если на устройстве включены инструменты разработчика |
| `phone_voip_call_flag` | 1, если во время транзакции был активен VoIP-звонок |
| `low_battery_flag` | 1, если уровень заряда батареи устройства ниже 15% |

### Флаги новизны устройства (впервые у данного клиента)

| Признак | Описание |
|---------|----------|
| `operating_system_is_new` | 1, если этот тип OS ни разу не встречался у данного клиента |
| `os_version_is_new` | 1, если эта версия OS ни разу не встречалась у данного клиента |
| `screen_size_is_new` | 1, если это разрешение экрана ни разу не встречалось у данного клиента |
| `timezone_is_new` | 1, если этот часовой пояс ни разу не встречался у данного клиента |
| `accept_language_is_new` | 1, если этот HTTP Accept-Language заголовок ни разу не встречался у данного клиента |

### Составные признаки устройства и сессии

| Признак | Описание |
|---------|----------|
| `device_risk_score` | Взвешенная сумма: `compromised x 5 + rdp x 3 + dev_tools x 2` |
| `session_tx_count` | Число предыдущих транзакций в рамках текущей сессии |
| `session_amount_sum` | Суммарные траты предыдущих транзакций в текущей сессии |
| `session_channel_diversity` | 1, если тип канала отличается от предыдущей транзакции в этой сессии |
| `session_length_estimate` | Псевдоним `session_tx_count` |
| `session_mcc_switch` | 1, если MCC отличается от предыдущей транзакции в этой сессии |
| `session_duration_minutes` | Минут с момента первой транзакции в текущей сессии |
| `pause_ses` | Секунд с момента предыдущей транзакции в той же сессии |
| `screen_w` | Ширина экрана в пикселях (извлечена из строки `screen_size` формата "WxH") |
| `screen_h` | Высота экрана в пикселях (извлечена из строки `screen_size` формата "WxH") |
| `session_avg_amount` | Средняя сумма на транзакцию за текущую сессию |
| `rdp_x_session_depth` | `rdp_flag x session_tx_count` — глубина сессии при удалённом управлении |

---

## Секция F -- Временные признаки и глобальные частоты (`temporal.py`)

### Глобальные частоты (уровень популяции, вычислены на тренировочных данных)

| Признак | Описание |
|---------|----------|
| `global_mcc_freq` | Как часто этот MCC встречается среди всех тренировочных транзакций |
| `global_channel_freq` | Как часто этот тип канала встречается среди всех тренировочных транзакций |
| `global_device_os_freq` | Как часто этот тип OS встречается среди всех тренировочных транзакций |
| `global_timezone_freq` | Как часто этот часовой пояс встречается среди всех тренировочных транзакций |
| `global_language_freq` | Как часто этот Accept-Language заголовок встречается среди всех тренировочных транзакций |
| `global_pos_cd_freq` | Как часто этот POS код встречается среди всех тренировочных транзакций |
| `global_event_type_freq` | Как часто этот тип события встречается среди всех тренировочных транзакций |
| `global_event_desc_freq` | Как часто это описание события встречается среди всех тренировочных транзакций |

*Если `global_stats` не передан, каждый признак заменяется накопительным счётчиком по клиенту (fallback без утечки данных).*

### Временные признаки и скорость транзакций

| Признак | Описание |
|---------|----------|
| `circadian_deviation_score` | Абсолютное отклонение текущего часа от исторического среднего часа клиента |
| `channel_shift_score` | 1, если тип канала отличается от предыдущей транзакции |
| `velocity_change_flag` | 1, если сегодняшнее число транзакций превышает 2x среднедневное клиента за 30д |
| `time_gap_variance_30d` | Std межтранзакционных интервалов по последним 90 транзакциям (скользящее) |
| `time_gap_mean_30d` | Среднее межтранзакционных интервалов по последним 90 транзакциям (скользящее) |
| `time_gap_min_10tx` | Минимальный межтранзакционный интервал среди последних 10 |
| `time_gap_cv_30d` | Коэффициент вариации интервалов: `std / mean` (масштабонезависимая нерегулярность) |
| `new_device_flag` | 1, если этот тип OS новый для клиента (псевдоним `operating_system_is_new`) |
| `merchant_entropy_user` | Доля транзакций клиента в данном MCC (`mcc_freq_user_cum / tx_count_lifetime`) |
| `browser_language_mismatch` | 1, если `accept_language` != `browser_language` |

### Составные сигналы риска

| Признак | Описание |
|---------|----------|
| `rdp_and_large_amount_flag` | 1, если активен RDP и сумма > 1000 |
| `amount_zscore_given_channel` | Сумма относительно накопленного среднего клиента в данном канале |
| `amount_zscore_given_mcc` | Сумма относительно накопленного среднего клиента в данном MCC |
| `amount_zscore_given_device` | Сумма относительно накопленного среднего клиента на данном OS |
| `tx_time_zscore_given_user` | Скользящее std часов транзакций клиента (временная нерегулярность) |

---

## Секция G -- Z-score и составные флаги (`zscore.py`)

### Z-score суммы (глобальные базисы по каналу/MCC)

| Признак | Описание |
|---------|----------|
| `amount_zscore_channel` | Z-score суммы относительно глобального среднего/std для данного канала |
| `amount_zscore_mcc` | Z-score суммы относительно глобального среднего/std для данного MCC |

### Комбинация и сессия

| Признак | Описание |
|---------|----------|
| `global_combination_freq` | Частота комбинации (channel x OS) на уровне популяции |
| `session_first_tx_flag` | 1, если это первая транзакция в сессии |

### Флаги изменений (относительно предыдущей транзакции)

| Признак | Описание |
|---------|----------|
| `language_change_flag` | 1, если `accept_language` отличается от предыдущей транзакции |
| `os_change_flag` | 1, если тип OS отличается от предыдущей транзакции |
| `timezone_change_flag` | 1, если часовой пояс отличается от предыдущей транзакции |
| `rapid_sequence_flag` | 1, если с последней транзакции прошло менее 5 минут |

### Составные флаги риска

| Признак | Описание |
|---------|----------|
| `device_change_and_large_amount_flag` | 1, если обнаружен новый OS и сумма > 1000 |
| `session_first_tx_large_flag` | 1, если первая транзакция в сессии и сумма > 1000 |

### Расстояние и энтропия

| Признак | Описание |
|---------|----------|
| `geo_jump_proxy` | Абсолютная разница часовых поясов с предыдущей транзакцией (приближённый геосдвиг) |
| `device_entropy_ratio` | Уникальных типов OS за 30 дней, делённое на число транзакций за 30 дней |
| `language_mismatch` | 1, если `accept_language` != `browser_language` (null-safe версия) |
| `merchant_last_seen_days` | Дней с предыдущей транзакции клиента (любой); 999 если первая |
| `mcc_last_seen_days` | Дней с предыдущей транзакции клиента в данном MCC; 999 если первая |

---

## Секция H -- Накопительная статистика по категориям (`category_stats.py`)

Статистики по ключевым категориальным измерениям на уровне клиента, без утечки данных.
Предшествующие mean/std используют `cum_sum().shift(1) / cum_count().shift(1)` для исключения текущей строки.

### По (клиент, event_desc)

| Признак | Описание |
|---------|----------|
| `event_desc_spend_user` | Предыдущие накопленные траты для пары (клиент, event_desc) |
| `event_desc_amount_mean_user` | Предыдущее среднее суммы для данного event_desc |
| `event_desc_amount_std_user` | Предыдущее std суммы для данного event_desc |
| `event_desc_last_seen_days` | Дней с последней транзакции с данным event_desc; 999 если первая |
| `amount_zscore_given_event_desc` | Z-score суммы относительно предыдущей истории клиента для данного event_desc |
| `event_desc_spend_share_user` | Доля предыдущих трат за всё время, приходящихся на данный event_desc |
| `event_desc_spend_vs_90d` | Накопленные траты по event_desc / траты за 90 дней |

### По (клиент, event_type_nm)

| Признак | Описание |
|---------|----------|
| `event_type_spend_user` | Предыдущие накопленные траты для данного event_type |
| `event_type_amount_mean_user` | Предыдущее среднее суммы для данного event_type |
| `event_type_amount_std_user` | Предыдущее std суммы для данного event_type |
| `event_type_last_seen_days` | Дней с последней транзакции с данным event_type; 999 если первая |
| `amount_zscore_given_event_type` | Z-score суммы относительно предыдущей истории клиента для данного event_type |
| `event_type_spend_share_user` | Доля предыдущих трат за всё время для данного event_type |
| `event_type_spend_vs_90d` | Накопленные траты по event_type / траты за 90 дней |

### По (клиент, channel_indicator_sub_type)

| Признак | Описание |
|---------|----------|
| `spend_in_subchannel_lifetime` | Предыдущие накопленные траты для данного subchannel |
| `tx_count_in_subchannel_lifetime` | Предыдущее накопленное число транзакций для данного subchannel |
| `amount_mean_subchannel_user` | Предыдущее среднее суммы для данного subchannel |
| `amount_std_subchannel_user` | Предыдущее std суммы для данного subchannel |
| `subchannel_last_seen_days` | Дней с последней транзакции с данным subchannel; 999 если первая |
| `subchannel_usage_share` | Доля транзакций за всё время в данном subchannel |
| `channel_indicator_sub_type_log_count` | log1p от `tx_count_in_subchannel_lifetime` |
| `is_new_subchannel_for_user` | 1, если данный subchannel ни разу не встречался у клиента |
| `amount_zscore_given_subchannel` | Z-score суммы относительно предыдущей истории клиента для данного subchannel |
| `subchannel_spend_vs_90d` | Накопленные траты по subchannel / траты за 90 дней |

### По (клиент, channel_type_subtype)

| Признак | Описание |
|---------|----------|
| `amount_mean_channel_type_subtype_user` | Предыдущее среднее суммы для комбо type+subtype |
| `amount_std_channel_type_subtype_user` | Предыдущее std суммы для комбо type+subtype |
| `channel_type_subtype_last_seen_days` | Дней с последней транзакции с данным type+subtype; 999 если первая |
| `is_new_channel_type_subtype_for_user` | 1, если данный type+subtype новый для клиента |
| `amount_zscore_given_channel_type_subtype` | Z-score суммы относительно предыдущей истории для данного type+subtype |
| `channel_type_subtype_spend_vs_90d` | Накопленные траты по type+subtype / траты за 90 дней |

### По (клиент, evtype_channel)

| Признак | Описание |
|---------|----------|
| `amount_mean_evtype_channel_user` | Предыдущее среднее суммы для комбо evtype+channel |
| `amount_std_evtype_channel_user` | Предыдущее std суммы для комбо evtype+channel |
| `evtype_channel_last_seen_days` | Дней с последней транзакции с данным evtype+channel; 999 если первая |
| `is_new_evtype_channel_for_user` | 1, если данный evtype+channel новый для клиента |
| `amount_zscore_given_evtype_channel` | Z-score суммы относительно предыдущей истории для данного evtype+channel |
| `evtype_channel_spend_vs_90d` | Накопленные траты по evtype+channel / траты за 90 дней |

### По (клиент, evtype_subchannel)

| Признак | Описание |
|---------|----------|
| `amount_mean_evtype_subchannel_user` | Предыдущее среднее суммы для комбо evtype+subchannel |
| `amount_std_evtype_subchannel_user` | Предыдущее std суммы для комбо evtype+subchannel |
| `evtype_subchannel_last_seen_days` | Дней с последней транзакции с данным evtype+subchannel; 999 если первая |
| `is_new_evtype_subchannel_for_user` | 1, если данный evtype+subchannel новый для клиента |
| `amount_zscore_given_evtype_subchannel` | Z-score суммы относительно предыдущей истории для данного evtype+subchannel |
| `evtype_subchannel_spend_vs_90d` | Накопленные траты по evtype+subchannel / траты за 90 дней |

### По (клиент, event_type_nm, mcc_code)

| Признак | Описание |
|---------|----------|
| `amount_mean_evtype_mcc_user` | Предыдущее среднее суммы для комбо evtype+MCC |
| `amount_std_evtype_mcc_user` | Предыдущее std суммы для комбо evtype+MCC |
| `evtype_mcc_last_seen_days` | Дней с последней транзакции с данным evtype+MCC; 999 если первая |
| `is_new_evtype_mcc_for_user` | 1, если данный evtype+MCC новый для клиента |
| `amount_zscore_given_evtype_mcc` | Z-score суммы относительно предыдущей истории для данного evtype+MCC |
| `evtype_mcc_spend_vs_90d` | Накопленные траты по evtype+MCC / траты за 90 дней |

### По (клиент, pos_cd)

| Признак | Описание |
|---------|----------|
| `spend_in_pos_lifetime` | Предыдущие накопленные траты для данного POS кода |
| `amount_mean_pos_user` | Предыдущее среднее суммы для данного POS кода |
| `amount_std_pos_user` | Предыдущее std суммы для данного POS кода |
| `pos_last_seen_days` | Дней с последней транзакции с данным POS кодом; 999 если первая |
| `amount_zscore_given_pos` | Z-score суммы относительно предыдущей истории для данного POS кода |
| `pos_spend_vs_90d` | Накопленные траты по POS / траты за 90 дней |

### По (клиент, tx_type_group)

| Признак | Описание |
|---------|----------|
| `spend_in_tx_type_lifetime` | Предыдущие накопленные траты для данного tx_type_group |
| `tx_count_in_tx_type_lifetime` | Предыдущее накопленное число транзакций для данного tx_type_group |
| `amount_mean_tx_type_user` | Предыдущее среднее суммы для данного tx_type_group |
| `amount_std_tx_type_user` | Предыдущее std суммы для данного tx_type_group |
| `tx_type_usage_share` | Доля транзакций за всё время в данном tx_type_group |
| `amount_zscore_given_tx_type` | Z-score суммы относительно предыдущей истории для данного tx_type_group |
| `tx_type_spend_share` | Доля предыдущих трат за всё время для данного tx_type_group |
| `tx_type_spend_vs_90d` | Накопленные траты по tx_type / траты за 90 дней |

### Флаги новизны кросс-категориальных комбинаций

| Признак | Описание |
|---------|----------|
| `is_new_channel_desc_combo` | 1, если комбинация (channel_type, event_desc) новая для данного клиента |
| `is_new_channel_type_combo` | 1, если комбинация (channel_type, event_type_nm) новая |
| `is_new_subchannel_type_combo` | 1, если комбинация (subchannel, event_type_nm) новая |
| `is_new_type_desc_combo` | 1, если комбинация (event_type_nm, event_desc) новая |
| `is_new_txtype_channel_combo` | 1, если комбинация (tx_type_group, channel_type) новая |

### Глобальные z-score (уровень популяции)

Доступны при наличии `global_stats`; иначе константа 0.

| Признак | Описание |
|---------|----------|
| `amount_zscore_event_desc_global` | Z-score суммы относительно глобального среднего/std для данного event_desc |
| `amount_zscore_event_type_global` | Z-score суммы относительно глобального среднего/std для данного event_type |
| `amount_zscore_subchannel_global` | Z-score суммы относительно глобального среднего/std для данного subchannel |
| `amount_zscore_pos_global` | Z-score суммы относительно глобального среднего/std для данного POS кода |
| `global_subchannel_freq` | Глобальная частота данного subchannel в тренировочной популяции |
| `amount_zscore_channel_type_subtype_global` | Z-score суммы относительно глобального среднего/std для данного channel_type_subtype |
| `global_channel_type_subtype_freq` | Глобальная частота данного channel_type_subtype |
| `amount_zscore_evtype_channel_global` | Z-score суммы относительно глобального среднего/std для данного evtype_channel |
| `global_evtype_channel_freq` | Глобальная частота данного evtype_channel |
| `amount_zscore_evtype_subchannel_global` | Z-score суммы относительно глобального среднего/std для данного evtype_subchannel |
| `global_evtype_subchannel_freq` | Глобальная частота данного evtype_subchannel |
| `amount_zscore_evtype_mcc_global` | Z-score суммы относительно глобального среднего/std для пары (event_type, mcc_code) |
| `global_evtype_mcc_freq` | Глобальная частота пары (event_type, mcc_code) |

---

## Секция I -- Байесовские target-энкодинги (`category_stats.py`)

Вычислены на размеченной тренировочной выборке через `compute_global_stats()`. Каждый признак — сглаженная Байесовским методом частота мошенничества для значения категории: `smoothed = (fraud_count + alpha * global_rate) / (labeled_count + alpha)`, alpha=20. Новые значения в тесте заполняются сохранённой глобальной частотой мошенничества; 0.5 при отсутствии `global_stats`.

| Признак | Ключ соединения | Описание |
|---------|-----------------|----------|
| `event_type_nm_target_enc` | `event_type_nm` | Сглаженная частота мошенничества по типу события |
| `event_desc_target_enc` | `event_desc` | Сглаженная частота мошенничества по описанию события |
| `channel_type_target_enc` | `channel_indicator_type` | Сглаженная частота мошенничества по типу канала |
| `channel_subtype_target_enc` | `channel_indicator_sub_type` | Сглаженная частота мошенничества по subchannel |
| `mcc_target_enc` | `mcc_code` | Сглаженная частота мошенничества по MCC; null (non-card) заполняется глобальной частотой |
| `channel_type_subtype_target_enc` | `channel_type_subtype` | Сглаженная частота мошенничества по (type x subtype) |
| `evtype_channel_target_enc` | `evtype_channel` | Сглаженная частота мошенничества по (event_type x channel) |
| `evtype_subchannel_target_enc` | `evtype_subchannel` | Сглаженная частота мошенничества по (event_type x subchannel) |
| `type_desc_pair_target_enc` | `(event_type_nm, event_desc)` | Частота мошенничества на уровне пары — сильнейший одиночный сигнал |
| `channel_type_fraud_rate_within_group` | `(tx_type_group, channel_indicator_type)` | Частота мошенничества по каналу при условии tx_type_group |
| `channel_subtype_fraud_rate_within_group` | `(tx_type_group, channel_indicator_sub_type)` | Частота мошенничества по subchannel при условии tx_type_group |
| `evtype_mcc_target_enc` | `(event_type_nm, mcc_code)` | Сглаженная частота мошенничества по (event_type x MCC) |

---

## Секция J -- Бинарные флаги риска (`category_risk.py`)

Жёстко заданы по результатам эмпирического анализа частот мошенничества. Не зависят от `global_stats`.

| Признак | Условие |
|---------|---------|
| `is_very_high_risk_desc` | `event_desc` входит в {60, 41, 27, 73, 103} — все >89% мошенничества |
| `is_high_risk_desc` | `event_desc` входит в {68, 29, 31, 119, 113, 109, 37} — все >67% мошенничества |
| `is_low_risk_desc` | `event_desc` входит в {51, 5, 4, 69, 97, 32} — все <20% мошенничества |
| `is_high_risk_channel` | `channel_indicator_type == 6` ИЛИ `channel_indicator_sub_type == 11` |
| `is_p2p_danger_channel` | `tx_type_group == 2` И `channel_indicator_sub_type == 5` (90.5% мошенничества в P2P) |
| `is_near_certain_fraud_pair` | `(event_type_nm, event_desc)` входит в {(14,60), (14,41), (14,73)} — 93-100% мошенничества |

---

## Сводка по количеству признаков

| Секция | Файл | Признаков |
|--------|------|-----------|
| A -- Уровень транзакции | `transaction.py` | 25 (вкл. ключ маршрутизации model_group) |
| B -- Поведение и история | `behavioral.py` | 64 (6 счётчиков + 8 производных + 21 лаговый + 9 running-max + 18 log-частот + 2 совпадения) |
| K -- Обратная связь по меткам | `feedback.py` | 20 |
| D -- Базовые rolling-статистики | `rolling.py` | 126 (14 метрик x 9 окон) |
| D -- Производные rolling | `rolling.py` | 60 (9 субдневных + 20 день+ + 10 накопленных + 5 долей + 7 card/p2p + 6 кросс-оконных + 3 балансовых) |
| E -- Устройство и сессия | `device.py` | 22 |
| F -- Временные и глобальные частоты | `temporal.py` | 23 |
| G -- Z-score и составные | `zscore.py` | 15 |
| H -- Статистики по категориям | `category_stats.py` | 67 (7+7+10+6+6+6+6+6+8+5 накопленных) |
| H -- Глобальные z-score | `category_stats.py` | 13 |
| I -- Target-энкодинги | `category_stats.py` | 12 |
| J -- Бинарные флаги риска | `category_risk.py` | 6 |
| **Итого** | | **~453** |

*Минус 1 `model_group` (ключ маршрутизации, не признак) и 1 `amount_clean` (промежуточная, удаляется) = ~451 используемых колонок признаков.*

---

## Чёрный список признаков (нулевой gain, исключены из входа модели)

Эти признаки были удалены из кода или внесены в чёрный список в `config.py` после того, как показали нулевой split gain:

`compromised_and_high_amount_flag`, `timezone_mismatch`, `burst_flag_15m`, `voip_and_new_mcc_flag`, `suspicious_env_flag`, `mcc_rare_global_flag`, `rare_combination_flag`, `compromised_x_amount_ratio`, `pos_cd_is_new`, `mcc_is_new_for_user`, `amount_usd_normalized`, `amount_missing_flag`, `new_device_and_night_flag`, `new_mcc_flag`, `new_channel_flag`.

Примечание: `compromised_flag` и `developer_tools_flag` по-прежнему производятся в `device.py`, но находятся в чёрном списке. Все остальные признаки из чёрного списка полностью удалены из пайплайна.
