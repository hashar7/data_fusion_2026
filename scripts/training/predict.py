"""Score the test set using per-group ensembles and write a submission CSV."""
import gc
import time
from datetime import datetime
import os

import numpy as np
import polars as pl
from catboost import CatBoostClassifier
import lightgbm as lgb


from scripts.training.config import (
    FEATURES_DIR, TRAIN_END_DATE, TX_TYPE_GROUPS, CATBOOST_BLEND_WEIGHT,
)
from scripts.training._utils import _parquet_files, _progress
from scripts.training.train_rf import sanitize_rf_features
# from scripts.training.pipeline import _score_tx_group_ensemble_chunk


def _score_tx_group_ensemble_chunk(
    X_chunk: np.ndarray,
    model_group_chunk: np.ndarray,
    lgbm_boosters: dict[int, list[lgb.Booster]],
    catboost_models: dict[int, CatBoostClassifier],
    blend_weight: float = CATBOOST_BLEND_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score one chunk with the historical tx_type_group ensemble:
        1. average LightGBM scores across seeds for the row's model_group
        2. score CatBoost for the same group
        3. blend exactly as in train_baseline / score_test
    """
    lgbm_scores = np.empty(len(X_chunk), dtype=np.float32)
    cb_scores = np.empty(len(X_chunk), dtype=np.float32)
    blend_scores = np.empty(len(X_chunk), dtype=np.float32)

    for group_id in TX_TYPE_GROUPS:
        group_mask = model_group_chunk == group_id
        if not group_mask.any():
            continue

        X_g = X_chunk[group_mask]
        boosters = lgbm_boosters[group_id]
        cb_model = catboost_models[group_id]

        lgbm_sum = np.zeros(len(X_g), dtype=np.float64)
        for booster in boosters:
            lgbm_sum += booster.predict(
                X_g,
                num_iteration=booster.best_iteration,
            ).astype(np.float64)
        lgbm_avg = (lgbm_sum / len(boosters)).astype(np.float32)

        cb_s = cb_model.predict_proba(X_g)[:, 1].astype(np.float32)
        blended = (lgbm_avg * (1.0 - blend_weight) + cb_s * blend_weight).astype(np.float32)

        lgbm_scores[group_mask] = lgbm_avg
        cb_scores[group_mask] = cb_s
        blend_scores[group_mask] = blended

        del X_g, lgbm_sum, lgbm_avg, cb_s, blended
        gc.collect()

    return lgbm_scores, cb_scores, blend_scores


def score_test(
    lgbm_boosters: dict,
    catboost_models: dict,
    feature_cols: list,
    submission_path: str,
    chunk_size: int = 500_000,
) -> None:
    """
    Score test rows using per-group ensembles and write a submission CSV.

    lgbm_boosters  : dict[group_id → list[lgb.Booster]]  (multi-seed)
    catboost_models: dict[group_id → CatBoostClassifier | None]
    feature_cols   : list of feature column names (same order for all models).

    Test rows: is_train == 1  AND  event_dttm >= TRAIN_END_DATE.
    Each row is routed to the ensemble whose key matches its model_group value.
    Scoring pipeline per row:
        1. Average LGBM predictions across seeds
        2. Blend with CatBoost (CATBOOST_BLEND_WEIGHT)
    """
    test_files = _parquet_files(FEATURES_DIR)
    if not test_files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")

    train_end   = datetime.fromisoformat(TRAIN_END_DATE)
    n_seeds     = {g: len(v) for g, v in lgbm_boosters.items()}
    group_names = {g: TX_TYPE_GROUPS.get(g, str(g)) for g in lgbm_boosters}
    print(f"\nScoring test set from {FEATURES_DIR!r} "
          f"({len(test_files)} partitions) …", flush=True)
    print(f"  LightGBM seeds  : {n_seeds}")
    print(f"  CatBoost weight : {CATBOOST_BLEND_WEIGHT}\n", flush=True)

    all_event_ids: list = []
    all_scores:    list = []
    wall_times:    list = []
    total_test_rows = 0
    group_counts: dict = {g: 0 for g in lgbm_boosters}

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
        tg_arr    = test_chunk["model_group"].to_numpy().astype(np.int8)
        scores    = np.zeros(len(test_chunk), dtype=np.float32)

        for group_id, boosters in lgbm_boosters.items():
            group_mask = tg_arr == group_id
            if not group_mask.any():
                continue
            group_local_idx = np.where(group_mask)[0]
            cb_model        = catboost_models.get(group_id)

            # Score in sub-chunks to limit RAM
            group_scores_parts: list = []
            for cs in range(0, len(group_local_idx), chunk_size):
                ce      = min(cs + chunk_size, len(group_local_idx))
                sub_idx = group_local_idx[cs:ce]
                X_sub   = (
                    test_chunk[sub_idx.tolist()]
                    .select(feature_cols)
                    .to_numpy(allow_copy=True)
                    .astype(np.float32)
                )

                # Average LGBM predictions across seeds
                lgbm_sum = np.zeros(len(sub_idx), dtype=np.float64)
                for booster in boosters:
                    lgbm_sum += booster.predict(
                        X_sub, num_iteration=booster.best_iteration
                    ).astype(np.float64)
                lgbm_avg = (lgbm_sum / len(boosters)).astype(np.float32)

                # Blend with CatBoost
                if cb_model is not None:
                    w    = CATBOOST_BLEND_WEIGHT
                    cb_s = cb_model.predict_proba(X_sub)[:, 1].astype(np.float32)
                    blended = (lgbm_avg * (1.0 - w) + cb_s * w).astype(np.float32)
                    del cb_s
                else:
                    blended = lgbm_avg

                group_scores_parts.append(blended)
                del X_sub, lgbm_sum, lgbm_avg, blended
                gc.collect()

            group_scores = np.concatenate(group_scores_parts)
            del group_scores_parts
            scores[group_local_idx] = group_scores
            group_counts[group_id] += int(group_mask.sum())
            del group_scores
            gc.collect()

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


def score_test_final_ensemble(
    final_model: CatBoostClassifier,
    tx_lgbm_models: dict[int, list[lgb.Booster]],
    tx_cb_models: dict[int, CatBoostClassifier],
    rf_bundle: dict,
    fu_cb_model: CatBoostClassifier,
    feature_cols: list[str],
    submission_path: str,
) -> None:
    """
    Score test rows using the final ensemble and write submission with the same
    schema/location pattern as train_baseline.
    """
    test_files = _parquet_files(FEATURES_DIR)
    if not test_files:
        raise FileNotFoundError(f"No parquet files found in {FEATURES_DIR!r}")

    train_end = datetime.fromisoformat(TRAIN_END_DATE)
    all_event_ids: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    wall_times: list[float] = []
    total_test_rows = 0

    print(f"\nScoring test set with final ensemble from {FEATURES_DIR!r} …", flush=True)
    print(f"  CatBoost tx blend weight : {CATBOOST_BLEND_WEIGHT}\n", flush=True)

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
            _progress(i + 1, len(test_files), wall_times, suffix=f"test rows {total_test_rows:,}")
            del test_chunk
            continue

        event_ids = test_chunk["event_id"].to_numpy()
        model_group_chunk = test_chunk["model_group"].to_numpy().astype(np.int8)
        X_chunk = test_chunk.select(feature_cols).to_numpy(allow_copy=True).astype(np.float32)

        tx_lgbm_scores, tx_cb_scores, tx_blend_scores = _score_tx_group_ensemble_chunk(
            X_chunk,
            model_group_chunk,
            tx_lgbm_models,
            tx_cb_models,
            blend_weight=CATBOOST_BLEND_WEIGHT,
        )

        X_rf = sanitize_rf_features(
            X_chunk,
            feature_cols=rf_bundle["feature_cols"],
            stage=f"test chunk {i} rf",
        )
        rf_scores = rf_bundle["model"].predict_proba(X_rf)[:, 1].astype(np.float32)
        fu_scores = fu_cb_model.predict_proba(X_chunk)[:, 1].astype(np.float32)

        X_meta = np.column_stack(
            [
                tx_lgbm_scores,
                tx_cb_scores,
                tx_blend_scores,
                rf_scores,
                fu_scores,
            ]
        ).astype(np.float32)

        final_scores = final_model.predict_proba(X_meta)[:, 1].astype(np.float32)

        all_event_ids.append(event_ids)
        all_scores.append(final_scores)
        total_test_rows += len(test_chunk)

        del (
            test_chunk,
            event_ids,
            model_group_chunk,
            X_chunk,
            tx_lgbm_scores,
            tx_cb_scores,
            tx_blend_scores,
            X_rf,
            rf_scores,
            fu_scores,
            X_meta,
            final_scores,
        )
        gc.collect()

        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, len(test_files), wall_times, suffix=f"test rows {total_test_rows:,}")

    print()

    if total_test_rows == 0:
        raise RuntimeError(
            f"No test rows found in {FEATURES_DIR!r}. "
            "Check TRAIN_END_DATE and parquet files."
        )

    event_ids_all = np.concatenate(all_event_ids)
    scores_all = np.concatenate(all_scores)
    del all_event_ids, all_scores
    gc.collect()

    submission = pl.DataFrame(
        {
            "event_id": event_ids_all,
            "predict": scores_all,
        }
    )

    n_dupes = len(submission) - submission["event_id"].n_unique()
    if n_dupes > 0:
        print(f"  WARNING: {n_dupes:,} duplicate event_ids in submission.", flush=True)

    submission.write_csv(submission_path)

    print(f"  Test rows scored   : {total_test_rows:,}")
    print(f"  Submission written : {submission_path}")
    print("  Score distribution :")
    for pct, lbl in [
        (0, "min"),
        (25, "p25"),
        (50, "median"),
        (75, "p75"),
        (90, "p90"),
        (95, "p95"),
        (99, "p99"),
        (100, "max"),
    ]:
        print(f"    {lbl:6s}: {float(np.percentile(scores_all, pct)):.6f}")

