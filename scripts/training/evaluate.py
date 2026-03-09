"""Validation-set evaluation: PR-AUC, max-F1 operating point, feature importance."""
import gc
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

from scripts.training._utils import _progress


def evaluate(
    booster: lgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    il_val: np.ndarray,
    feature_cols: list,
    row_mask: np.ndarray | None = None,
    chunk_size: int = 500_000,
    label: str = "",
) -> float:
    """
    Score the val split and compute PR-AUC on labeled rows.

    row_mask : optional boolean array (len = len(y_val)).
               When provided, only the subset of rows where row_mask is True
               is scored and evaluated. Useful for per-group evaluation.
               Processed via sorted index chunks — never loads the full group
               into RAM simultaneously.

    Returns PR-AUC on labeled rows (competition metric).
    """
    tag = f" [{label}]" if label else ""
    print(f"Evaluating on validation set{tag} …", flush=True)

    if row_mask is not None:
        indices = np.where(np.asarray(row_mask))[0]   # sorted
    else:
        indices = np.arange(len(y_val))

    n_rows      = len(indices)
    n_chunks    = max(1, (n_rows + chunk_size - 1) // chunk_size)
    all_scores: list = []
    wall_times: list = []

    for i in range(n_chunks):
        t0    = time.perf_counter()
        start = i * chunk_size
        end   = min(start + chunk_size, n_rows)
        chunk_idx = indices[start:end]
        X_chunk   = np.array(X_val[chunk_idx])
        scores    = booster.predict(
            X_chunk, num_iteration=booster.best_iteration
        ).astype(np.float32)
        all_scores.append(scores)
        del X_chunk
        gc.collect()
        wall_times.append(time.perf_counter() - t0)
        _progress(i + 1, n_chunks, wall_times)

    print(flush=True)

    scores_all = np.concatenate(all_scores)
    labels_all = np.asarray(y_val)[indices]
    il_all     = np.asarray(il_val)[indices]
    del all_scores
    gc.collect()

    labeled_mask   = il_all == 1
    n_labeled      = int(labeled_mask.sum())
    n_total        = len(labels_all)
    scores_labeled = scores_all[labeled_mask]
    labels_labeled = labels_all[labeled_mask]
    print(f"  Scoring {n_labeled:,} labeled rows out of {n_total:,} rows "
          f"({n_labeled/max(n_total,1):.2%} labeled)", flush=True)

    if n_labeled == 0:
        print("  WARNING: no labeled rows — PR-AUC undefined, returning 0.0")
        return 0.0
    if labels_labeled.sum() == 0:
        print("  WARNING: no positives in labeled rows — PR-AUC undefined, returning 0.0")
        return 0.0

    pr_auc     = average_precision_score(labels_labeled, scores_labeled)
    pr_auc_all = average_precision_score(labels_all, scores_all)
    print(f"\n  PR-AUC (labeled only, competition metric) : {pr_auc:.6f}")
    print(f"  PR-AUC (all rows incl. open-loop)         : {pr_auc_all:.6f}")

    prec_arr, rec_arr, thresholds = precision_recall_curve(labels_labeled, scores_labeled)
    f1_arr  = np.where(
        (prec_arr + rec_arr) == 0, 0.0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9),
    )
    best_idx = int(np.argmax(f1_arr))
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    print(f"\n  Max-F1 operating point (threshold = {best_thr:.4f})")
    print(f"    Precision : {prec_arr[best_idx]:.4f}")
    print(f"    Recall    : {rec_arr[best_idx]:.4f}")
    print(f"    F1        : {f1_arr[best_idx]:.4f}")

    y_pred_labeled = (scores_labeled >= best_thr).astype(np.int8)
    print("\n  Classification report at max-F1 threshold (labeled rows only):")
    print(classification_report(
        labels_labeled, y_pred_labeled,
        target_names=["confirmed (0)", "unconfirmed (1)"],
        digits=4,
    ))

    print("  Score distribution (labeled val rows):")
    for pct, lbl in [(0, "min"), (25, "p25"), (50, "median"),
                     (75, "p75"), (90, "p90"), (95, "p95"),
                     (99, "p99"), (100, "max")]:
        print(f"    {lbl:6s}: {np.percentile(scores_labeled, pct):.6f}")

    importance = booster.feature_importance(importance_type="gain")
    feat_imp = (
        pl.DataFrame({"feature": feature_cols, "gain": importance.tolist()})
        .sort("gain", descending=True)
        .head(30)
    )
    print("\n  Top-30 features by gain:")
    print(feat_imp)

    return pr_auc
