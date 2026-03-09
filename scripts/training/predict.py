"""Score the test set using per-group models and write a submission CSV."""
import gc
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import polars as pl

from scripts.training.config import FEATURES_DIR, TRAIN_END_DATE, TX_TYPE_GROUPS
from scripts.training._utils import _parquet_files, _progress


def score_test(
    boosters: dict,
    feature_cols: list,
    submission_path: str,
    chunk_size: int = 500_000,
) -> None:
    """
    Score test rows using per-group LightGBM models and write a submission CSV.

    boosters       : dict mapping tx_type_group (int) → lgb.Booster.
    feature_cols   : list of feature column names (same for all models).

    Test rows: is_train == 1  AND  event_dttm >= TRAIN_END_DATE.
    Each row is routed to the model whose key matches its tx_type_group value.
    """
    test_files = _parquet_files(FEATURES_DIR)
    if not test_files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")

    train_end = datetime.fromisoformat(TRAIN_END_DATE)
    group_names = {g: TX_TYPE_GROUPS.get(g, str(g)) for g in boosters}
    print(f"\nScoring test set from {FEATURES_DIR!r} "
          f"({len(test_files)} partitions) …", flush=True)
    print(f"  Models active: { {g: group_names[g] for g in sorted(boosters)} }\n",
          flush=True)

    all_event_ids: list = []
    all_scores:    list = []
    wall_times:    list = []
    total_test_rows = 0
    group_counts: dict = {g: 0 for g in boosters}

    for i, f in enumerate(test_files):
        t0 = time.perf_counter()

        chunk = pl.read_parquet(f)
        if chunk["event_dttm"].dtype == pl.Utf8:
            chunk = chunk.with_columns(
                pl.col("event_dttm").str.strptime(pl.Datetime, strict=False)
            )
        test_chunk = chunk.filter(
            (pl.col("is_train") == 1) & (pl.col("event_dttm") >= train_end)
        )
        del chunk
        gc.collect()

        if len(test_chunk) == 0:
            wall_times.append(time.perf_counter() - t0)
            _progress(i + 1, len(test_files), wall_times,
                      suffix=f"test rows {total_test_rows:,}")
            del test_chunk
            continue

        event_ids = test_chunk["event_id"].to_numpy()
        tg_arr    = test_chunk["tx_type_group"].to_numpy().astype(np.int8)
        scores    = np.full(len(test_chunk), 0.0, dtype=np.float32)

        for group_id, booster in boosters.items():
            group_mask = tg_arr == group_id
            if not group_mask.any():
                continue
            group_local_idx = np.where(group_mask)[0]

            # Score in sub-chunks to limit RAM
            chunk_scores_parts: list = []
            for cs in range(0, len(group_local_idx), chunk_size):
                ce     = min(cs + chunk_size, len(group_local_idx))
                sub_idx = group_local_idx[cs:ce]
                X_sub   = (
                    test_chunk[sub_idx.tolist()]
                    .select(feature_cols)
                    .to_numpy(allow_copy=True)
                    .astype(np.float32)
                )
                s = booster.predict(
                    X_sub, num_iteration=booster.best_iteration
                ).astype(np.float32)
                chunk_scores_parts.append(s)
                del X_sub
                gc.collect()

            group_scores = np.concatenate(chunk_scores_parts)
            scores[group_local_idx] = group_scores
            group_counts[group_id] += int(group_mask.sum())
            del group_scores, chunk_scores_parts

        all_event_ids.append(event_ids)
        all_scores.append(scores)
        total_test_rows += len(test_chunk)

        del test_chunk, scores, event_ids, tg_arr
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(test_files), wall_times,
                  suffix=f"test rows {total_test_rows:,}")

    print(flush=True)

    if total_test_rows == 0:
        raise RuntimeError(
            f"No test rows found in {FEATURES_DIR!r}. "
            "Check TRAIN_END_DATE and parquet files."
        )

    event_ids_all = np.concatenate(all_event_ids)
    scores_all    = np.concatenate(all_scores)
    del all_event_ids, all_scores
    gc.collect()

    submission = pl.DataFrame({
        "event_id": event_ids_all,
        "predict":  scores_all,
    })

    n_dupes = len(submission) - submission["event_id"].n_unique()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes:,} duplicate event_ids in submission.", flush=True)

    submission.write_csv(submission_path)

    print(f"  Test rows scored   : {total_test_rows:,}")
    for g, name in TX_TYPE_GROUPS.items():
        if g in group_counts:
            print(f"    {name:12s} : {group_counts[g]:,} rows")
    print(f"  Submission written : {submission_path}")
    print("  Score distribution :")
    for pct, lbl in [(0, "min"), (25, "p25"), (50, "median"),
                     (75, "p75"), (90, "p90"), (95, "p95"),
                     (99, "p99"), (100, "max")]:
        print(f"    {lbl:6s}: {float(np.percentile(scores_all, pct)):.6f}")
