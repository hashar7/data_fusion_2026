# Data Fusion 2026 — Guardian Track

Решение хакатона [Data Fusion 2026](https://ods.ai/competitions/data-fusion2026-guardian), трек Guardian.

**Задача:** классификация неподтверждённых банковских транзакций как мошеннических или легитимных.
**Метрика:** PR-AUC (`sklearn.metrics.average_precision_score`).

---

## Описание задачи

Клиенты банка иногда не подтверждают транзакции по соображениям безопасности. Задача — построить бинарный классификатор, который для каждой такой «неподтверждённой» операции предсказывает, является ли она мошеннической.

Ключевые сложности:
- **Сильный дисбаланс классов** — мошеннические транзакции встречаются очень редко
- **Временной реализм** — модель должна работать как живой классификатор; признаки могут использовать только информацию, доступную *до* текущей транзакции
- **Масштаб** — более 200 млн операций по 100К клиентам за 1.5 года
- **Три класса меток** — red (мошенничество), yellow (подтверждённая легитимная операция), green (непомеченные транзакции)

### Периоды данных

| Период | Даты | Назначение |
|--------|------|------------|
| Pre-train | 2023-10-01 – 2024-09-30 | Непомеченная история транзакций для контекста признаков |
| Train | 2024-10-01 – 2025-05-31 | Размеченные данные (red/yellow/green) |
| Pre-test | 2025-06-01 – 2025-08-09 | Непомеченный контекст, предшествующий тестовому дню |
| Test | 2025-06-01 – 2025-08-09 | Один финальный день на клиента — требует скоринга |

---

## Структура репозитория

```
scripts/
  features/           Модули feature engineering (один файл на секцию A–K)
    _helpers.py       Общие строители выражений
    global_stats.py   Статистики по популяции + Байесовские target-энкодинги
    transaction.py    Секция A — дата/время, флаги суммы, tx_type_group, комбинированные колонки
    behavioral.py     Секция B — накопительная история клиента, лаги, log-счётчики
    feedback.py       Секция K — динамические признаки на основе истории меток
    rolling.py        Секция D — rolling-агрегации (9 окон × 14 метрик)
    device.py         Секция E — флаги риска устройства/сессии
    temporal.py       Секция F — глобальные частоты, циркадное отклонение
    zscore.py         Секция G — z-score по каналу/MCC, составные флаги
    category_stats.py Секции H + I — накопительная статистика по категориям + target-энкодинги
    category_risk.py  Секция J — жёстко заданные бинарные флаги риска
  feature_engineering.py   Публичный API — запускает A → B → K → D → E → F → G → H → I → J
  feature_calculation.py   Публичный API — партиционированный пайплайн вывода в parquet

  calculation/        Партиционированное вычисление признаков
    config.py         OUTPUT_DIR, N_PARTITIONS, ID_COLS
    partition.py      process_partition(), _downcast()
    pipeline.py       Оркестрация build_processed_dataset()

  training/           Обучение моделей
    config.py         Все настраиваемые константы (даты, параметры, seeds, пути)
    _utils.py         Вспомогательные утилиты прогресс-бара
    data.py           load_labels(), build_memmaps(), кэш memmap
    train.py          train_model() — LightGBM с negative undersampling
    train_catboost.py train_catboost_model()
    hierarchical.py   Вспомогательные функции иерархического CatBoost (только train_v2)
    evaluate.py       evaluate() — PR-AUC, порог max-F1, важность признаков
    predict.py        score_test() — генерация submission CSV
    pipeline.py       Оркестратор train_baseline()
    pipeline_v2.py    Оркестратор train_v2() (рекомендуется)
  train_eval_model.py Тонкий публичный API — реэкспортирует train_baseline и train_v2

notebooks/
  feature_example.ipynb   Основной экспериментальный ноутбук (канонический пример использования)
  models/                 Сохранённые файлы моделей

misc/
  features.md         Полный каталог признаков (~450 признаков, организован по секциям)

data/                 Исходные parquet-файлы соревнования (не в репозитории)
data_processed/       Обработанные parquet с признаками (выход build_processed_dataset)
data_splits/          Memmap-файлы + memmap_meta.json (выход build_memmaps)
```

---

## Быстрый старт

```python
# В ноутбуке (рабочая директория = notebooks/):
import sys; sys.path.insert(0, "..")

import polars as pl
from scripts.feature_engineering import generate_fraud_features, compute_global_stats
from scripts.feature_calculation import build_processed_dataset
from scripts.train_eval_model import train_v2

# 1. Вычислить глобальные статистики + target-энкодинги по train+pretrain
pretrain_lf = pl.scan_parquet("../../data/pretrain_part_*.parquet")
train_lf    = pl.scan_parquet("../../data/train_part_*.parquet")
labels_lf   = pl.scan_parquet("../../data/train_labels.parquet")
full_lf     = pl.concat([pretrain_lf, train_lf])

global_stats = compute_global_stats(full_lf, labels_lf=labels_lf)

# 2. Построить обработанные parquet с признаками
all_lf = pl.concat([
    pretrain_lf, train_lf,
    pl.scan_parquet("../../data/pretest.parquet"),
    pl.scan_parquet("../../data/test.parquet"),
])
build_processed_dataset(all_lf, global_stats=global_stats, labels_lf=labels_lf)

# 3. Обучить иерархический ансамбль + сгенерировать submission
train_v2()
```

После шага 3 файл `notebooks/submission.csv` содержит скоры для тестовых транзакций.

> **После любого изменения feature engineering:** удалить `../data_splits/memmap_meta.json`
> (и `.npy`-файлы), чтобы принудительно пересобрать memmap.

---

## Feature Engineering

Пайплайн производит ~450 признаков в 10 секциях, все строго защищены от утечки данных:
- Rolling-окна используют `closed="left"` — интервал `[t − period, t)` не включает текущую строку
- Накопительные признаки используют `cum_count() - 1` / `cum_sum() - current_value`
- Глобальные статистики (target-энкодинги, популяционные средние) заморожены только по тренировочным данным

**Полный каталог признаков:** [`misc/features.md`](misc/features.md)

### Порядок выполнения пайплайна

```
A  transaction.py   — разложение даты/времени, флаги суммы, tx_type_group, комбинированные колонки
B  behavioral.py    — накопительные счётчики, лаги (n=1..5), running-max, log-частоты
K  feedback.py      — динамическая обратная связь по меткам (история мошенничества клиента)
D  rolling.py       — 9 окон × 14 метрик + card/P2P сплиты + производные соотношения
E  device.py        — флаги риска устройства, признаки сессии, размеры экрана
F  temporal.py      — глобальные частоты, циркадное отклонение, флаги скорости транзакций
G  zscore.py        — z-score по каналу/MCC, частота комбинаций, составные флаги
H  category_stats.py— накопительная статистика по (клиент, категория) для 8 ключевых измерений
I  category_stats.py— Байесовские сглаженные target-энкодинги (12 показателей частоты мошенничества)
J  category_risk.py — жёстко заданные бинарные флаги риска из эмпирического анализа
```

### Ключевые архитектурные решения

| Решение | Обоснование |
|---------|-------------|
| `tx_type_group` (0/1/2) делит non-payment / card / P2P | Частота мошенничества и полезные сигналы кардинально различаются между типами |
| `model_group` (0/1/2/3) дополнительно делит non-payment по `event_type_nm==7` | Тип 7 содержит 70M строк при 58% мошенничества и подавляет остальные типы в единой модели |
| Комбинированные колонки `channel_type_subtype`, `evtype_channel`, `evtype_subchannel` | Кодируют декартовы произведения как целые числа; используются в rolling-статистике, log-счётчиках и статистиках по категориям |
| Признаки обратной связи секции K | Переносят сигнал из предыдущих меток вперёд во времени; `fb_cust_prev_any_red` и `fb_sec_since_prev_red` — одни из сильнейших признаков |
| Байесовские target-энкодинги (α=20) | Сглаживают частоты мошенничества для редких категорий; для новых значений в тесте подставляется глобальная частота |

---

## Пайплайны обучения

Доступны два полноценных пайплайна. **Рекомендуется `train_v2()`** — он стабильно превосходит `train_baseline()` по PR-AUC.

### Общая инфраструктура

Оба пайплайна используют один и тот же шаг построения memmap:

```
build_memmaps()
    ├── Сканирует обработанные parquet, фильтрует is_train==1
    ├── Строки с event_dttm < VAL_CUTOFF_DATE (2025-04-01)                     → X_train / y_train
    ├── Строки с VAL_CUTOFF_DATE ≤ event_dttm < TRAIN_END_DATE (2025-06-01)    → X_val / y_val
    ├── il_train.npy  — помечает labeled (red+yellow)=1 vs green=0 в train-сплите
    ├── il_val.npy    — помечает labeled строки в val-сплите
    └── tg_train.npy / tg_val.npy  — ключ маршрутизации model_group на строку
```

Результат кэшируется в `../data_splits/memmap_meta.json`. Удалить для принудительной пересборки.

**Классы меток при обучении:**

| Класс | `target` | Есть в `train_labels.parquet` | Флаг `il_train` |
|-------|----------|-------------------------------|-----------------|
| Red (мошенничество) | 1 | Да | 1 |
| Yellow (подтверждённая легитимная) | 0 | Да | 1 |
| Green (непомеченные) | 0 | Нет | 0 |

Yellow-строки получают дополнительный вес `YELLOW_WEIGHT_MULTIPLIER = 2.0` поверх стандартного веса коррекции undersampling, поскольку это редкие «сложные негативы», которые чётко очерчивают границу мошенничества.

---

### `train_baseline()` — Ансамбль LightGBM + CatBoost по группам

```
scripts/training/pipeline.py
```

Для каждой из 4 групп моделей (`np_type7`, `np_other`, `card`, `p2p`):

**Шаг 1 — Multi-seed LightGBM**
- Обучает `len(ENSEMBLE_SEEDS)` = 1 модель LightGBM (настраивается; `[42]` для скорости)
- Negative undersampling с коэффициентом `NEG_SAMPLE_RATIO_BY_GROUP` (по умолчанию 5% для всех групп)
- Yellow-строки получают дополнительный вес `YELLOW_WEIGHT_MULTIPLIER`
- Early stopping на labeled val-строках (метрика `PRAUC`, 150 раундов; 200 для `np_other`)
- Переопределения параметров по группам через `LGBM_PARAMS_BY_GROUP`:
  - `np_type7`: более высокий LR (0.01) и меньше деревьев (3000) — большая однородная популяция
  - `np_other`: более тонкие разбиения (`min_child_samples=50`, `num_leaves=63`) — маленькая неоднородная группа
  - `p2p`: больше деревьев (8000) — при стандартном лимите итераций ещё улучшается

**Шаг 2 — Multi-seed CatBoost**
- Обучает `len(CATBOOST_SEEDS)` = 1 модель CatBoost на группу
- Те же undersampling, `eval_metric=PRAUC`, терпение 100 раундов

**Шаг 3 — Поиск весов смешивания по группам**
- Перебор `w ∈ {0.00, 0.05, …, 0.50}` на labeled val-строках
- Итоговый скор = `lgbm_avg × (1 − w) + catboost_avg × w`
- `CATBOOST_BLEND_WEIGHT = 0.25` — только резервное значение по умолчанию

**Шаг 4 — Оценка качества**
- PR-AUC по каждой группе на labeled val-строках
- Суммарный PR-AUC по всем группам (прокси метрики соревнования)
- Топ-30 важностей признаков выводится для seed-0 каждой группы

**Шаг 5 — Дообучение на полных данных**
- Все модели переобучаются на `train + labeled_val` вместе
- Количество раундов = `best_iteration × RETRAIN_FULL_ITER_FACTOR (1.05)`, без early stopping
- Val-негативы подвергаются undersampling с той же долей перед добавлением, чтобы не нарушить баланс классов

**Шаг 6 — Скоринг теста**
- `score_test()` маршрутизирует каждую тестовую строку по `model_group`, смешивает LGBM + CatBoost по группам
- Записывает `submission.csv`

**Сохранённые файлы моделей (после полного дообучения):**
```
models/model_np_type7_s0.txt          (LGBM на seed)
models/model_np_type7_catboost_s0.cbm (CatBoost на seed)
models/model_np_other_s0.txt
... (4 группы × 1+ seeds каждая)
```

---

### `train_v2()` — Иерархический ансамбль (рекомендуется)

```
scripts/training/pipeline_v2.py
```

Заменяет per-group CatBoost глобальным иерархическим разложением:

```
P(fraud | tx)  ≈  P(labeled | tx)  ×  P(fraud | labeled, tx)
              =  sigmoid(susp_raw)  ×  sigmoid(rgs_raw)
```

Иерархическая пара даёт ортогональный сигнал к per-group LGBM, поскольку обучается глобально (видит кросс-групповые паттерны) и использует нативную обработку категориальных признаков CatBoost.

#### Пошаговое описание

**Шаг 1 — Построение memmap** (то же, что в `train_baseline`)

**Шаг 2 — Предзагрузка val из parquet**
- Загружает labeled val-строки из обработанных parquet в pandas DataFrame
- Также загружает 1%-выборку непомеченных val-строк (для оптимизации весов на all-rows, `BLEND_ON_ALL_ROWS=True`)
- И labeled, и выбранные unlabeled строки используются при поиске весов смешивания

**Шаг 3 — Per-group LightGBM** (то же, что в `train_baseline`, без CatBoost)
- 1 seed на группу; каждый бустер скорит предзагруженный val DataFrame перед освобождением памяти
- `lgbm_blend_sum` накапливает скоры, выровненные по индексу val DataFrame

**Шаг 4 — Загрузка данных для иерархического CatBoost**
- `load_hier_split()` читает обработанные parquet, оставляет:
  - **Все labeled строки** (red + yellow) из периода train
  - **5% выборка green-строк** (`HIER_FULL_GREEN_RATIO`) — детерминированная хеш-выборка
- Признаки: при `HIER_USE_ALL_FEATURES=True` используется полный набор признаков LGBM + категориальные колонки CatBoost (`customer_id`, `mcc_code_int`)

**Шаг 5 — Suspicious CatBoost** (цель: `P(labeled | tx)`)
- Обучение: (red ∪ yellow) как позитивный класс, green как негативный
- Веса примеров:
  - Labeled строки: `SUSPICIOUS_LABELED_WEIGHT = 6.0`
  - Green-строки начиная с `RECENT_BORDER (2025-02-01)`: `SUSPICIOUS_GREEN_RECENT_W = 1.5`
  - Более старые green-строки: `SUSPICIOUS_GREEN_OLD_W = 3.0` (надёжно не-мошеннические — больший штраф)
- Метрика валидации: AUC на labeled val
- Параметры: `SUSPICIOUS_CATBOOST_PARAMS` (3000 итераций, глубина 8, l2=6.0)

**Шаг 6 — RGS CatBoost** (цель: `P(fraud | labeled, tx)`)
- Обучение: только labeled строки — red (мошенничество) vs yellow (подтверждённые)
- Близкое к 50% соотношение классов даёт чистый, сильный сигнал
- Веса примеров: `RGS_RED_WEIGHT = 2.5`, `RGS_YELLOW_WEIGHT = 1.0`
- Метрика валидации: PRAUC на labeled val
- Параметры: `RGS_CATBOOST_PARAMS` (5000 итераций, глубина 8, l2=8.0)

**Шаг 7 — Main CatBoost** (цель: прямое предсказание мошенничества)
- Глобальная модель: red=1 vs yellow+green=0
- Веса: red=10.0, yellow=2.5, green_recent=1.5, green_old=3.0
- Даёт дополнительный ортогональный сигнал через нативную обработку категорий CatBoost
- Параметры: `MAIN_CATBOOST_PARAMS` (5000 итераций, глубина 8, l2=8.0)

**Шаг 8 — Опциональный Recent LightGBM** (`TRAIN_RECENT_LGBM=True`)
- Один глобальный LGBM, обученный на данных начиная с `RECENT_BORDER (2025-02-01)`
- Захватывает временной дрейф паттернов мошенничества по мере приближения к тестовому периоду
- `LGBM_PARAMS_RECENT`: LR=0.02, 2000 деревьев

**Шаг 9 — Оптимизация весов смешивания**
- Скорит все модели на предзагруженном val DataFrame
- Иерархический скор: `sigmoid(susp_raw) × sigmoid(rgs_raw)`
- Смешивание: `final = w_lgbm * lgbm + w_hier * hier + w_main * main + w_recent * recent`
- При `BLEND_IN_LOGIT_SPACE=True` смешивание выполняется в пространстве логитов (лучше откалибровано)
- При `BLEND_ON_ALL_ROWS=True` оптимизируется на labeled + выбранных unlabeled val-строках
- Перебор по симплексу из 4 весов; лучшие веса выводятся в лог

**Шаг 10 — Дообучение на полных данных**
- Per-group LGBM: переобучение из memmap, `best_iter × RETRAIN_FULL_ITER_FACTOR` раундов
- Иерархический CatBoost: дообучение через `refit_model()` на расширенных данных (`cutoff_hi = TRAIN_END_DATE`)
- Main CatBoost: дообучение на полных данных
- Recent LGBM: переобучение на данных вплоть до `TRAIN_END_DATE`

**Шаг 11 — Скоринг теста**
- Для каждого чанка тестового parquet:
  - LGBM скор: среднее по группе
  - Иерархический скор: `sigmoid(susp) × sigmoid(rgs)`
  - Main CatBoost скор
  - Recent LGBM скор
  - Итог: смешивание в пространстве логитов с оптимизированными весами
- Записывает `submission.csv`

**Сохранённые файлы моделей (после полного дообучения):**
```
models/model_np_type7_s0.txt    (per-group LGBM)
models/model_np_other_s0.txt
models/model_card_s0.txt
models/model_p2p_s0.txt
models/model_suspicious.cbm    (глобальный Suspicious CatBoost)
models/model_rgs.cbm           (глобальный RGS CatBoost)
models/model_main_catboost.cbm (глобальный Main CatBoost)
models/model_recent.txt        (глобальный Recent LGBM)
```

Существующие файлы моделей автоматически архивируются как `model_*_ver_N.*` перед каждым запуском обучения.

---

## Справочник конфигурации

Все настраиваемые константы находятся в `scripts/training/config.py`. Ключевые параметры:

| Константа | Значение | Описание |
|-----------|----------|----------|
| `VAL_CUTOFF_DATE` | `2025-04-01` | Граница разбиения train/val |
| `TRAIN_END_DATE` | `2025-06-01` | Граница разбиения val/test |
| `NEG_SAMPLE_RATIO_BY_GROUP` | 0.05 для всех групп | Коэффициент undersampling негативов |
| `YELLOW_WEIGHT_MULTIPLIER` | 2.0 | Дополнительный вес для labeled негативов |
| `ENSEMBLE_SEEDS` | `[42]` | Seeds для LGBM (список → усреднение по нескольким seeds) |
| `RETRAIN_FULL_ITER_FACTOR` | 1.05 | Дополнительные раунды при дообучении на полных данных |
| `RECENT_BORDER` | `2025-02-01` | Начало «недавнего» окна для Recent LGBM |
| `BLEND_IN_LOGIT_SPACE` | `True` | Смешивать выходы моделей в пространстве логитов |
| `BLEND_ON_ALL_ROWS` | `True` | Включать выборку unlabeled строк в оптимизацию весов |
| `HIER_USE_ALL_FEATURES` | `True` | Использовать полный набор признаков LGBM для иерархического CatBoost |
| `TRAIN_RECENT_LGBM` | `True` | Обучать глобальный Recent LGBM |
| `FEATURE_BLACKLIST` | 15 признаков | Признаки с нулевым gain, исключённые из всех моделей |

---

## Железо и окружение

- **Только CPU, 32 ГБ RAM** — GPU не требуется (поддержка GPU через `train_v2(gpu=True)`)
- Python 3.11, Polars 1.x, LightGBM, CatBoost, scikit-learn, NumPy, pandas
- Feature engineering обрабатывает одну партицию клиентов за раз (50 партиций)
- Обучение использует numpy memmap, чтобы не загружать все признаки в RAM одновременно
- Выходные признаки приводятся к float32/int32, что вдвое сокращает использование диска и памяти
