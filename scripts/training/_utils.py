"""Shared utility helpers (progress display, file discovery, column selection)."""
from pathlib import Path
import os
import joblib
import glob
import re

import numpy as np
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier

from scripts.training.config import CATBOOST_FU_MODEL_PATH_FMT, RF_MODEL_PATH_FMT, TX_TYPE_GROUPS, LGBM_MODEL_PATH_FMT
from scripts.training.config import NON_FEATURE_COLS


# ── Progress display ──────────────────────────────────────────────────────────––––––––––––––

def _bar(done: int, total: int, width: int = 35) -> str:
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _progress(done: int, total: int, wall_times: list, suffix: str = "") -> None:
    pct    = 100.0 * done / max(total, 1)
    recent = wall_times[-5:]
    eta    = _fmt((sum(recent) / len(recent)) * (total - done)) if recent else "?"
    print(
        f"\r  {_bar(done, total)} {pct:5.1f}%  "
        f"{done}/{total}  ETA {eta}  {suffix}          ",
        end="", flush=True,
    )


# ── File / column helpers ─────────────────────────────────────────────────────––––––––––––––

def _parquet_files(directory: str) -> list:
    return sorted(Path(directory).glob("*.parquet"))


def _prepare_models_dir(models_dir: str) -> None:
    """
    Create models_dir if it does not exist.
    If unversioned model files (model_*.txt / model_*.cbm) are already present,
    rename them all with a _ver_N suffix so the new run's files don't overwrite them.
    All files from the same previous run share the same version number.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Collect unversioned model files — anything that does NOT already have _ver_N
    all_files = (
        glob.glob(os.path.join(models_dir, "model_*.txt")) +
        glob.glob(os.path.join(models_dir, "model_*.cbm")) +
        glob.glob(os.path.join(models_dir, "model_*.pkl"))
    )
    unversioned = [
        f for f in all_files
        if not re.search(r"_ver_\d+\.(txt|cbm|pkl)$", f)
    ]

    if not unversioned:
        return

    # Find the highest existing version number so we don't collide
    versioned = (
        glob.glob(os.path.join(models_dir, "model_*_ver_*.txt")) +
        glob.glob(os.path.join(models_dir, "model_*_ver_*.cbm")) +
        glob.glob(os.path.join(models_dir, "model_*_ver_*.pkl"))
    )
    max_ver = 0
    for f in versioned:
        m = re.search(r"_ver_(\d+)\.(txt|cbm|pkl)$", f)
        if m:
            max_ver = max(max_ver, int(m.group(1)))

    next_ver = max_ver + 1
    print(f"  Found {len(unversioned)} existing model file(s) - "
          f"archiving as _ver_{next_ver} ...")
    for f in sorted(unversioned):
        base, ext = os.path.splitext(f)
        dest = f"{base}_ver_{next_ver}{ext}"
        os.rename(f, dest)
        print(f"    {os.path.basename(f)}  ->  {os.path.basename(dest)}")
    print()


def _get_feature_cols(df: pl.DataFrame) -> list:
    numeric = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and df[c].dtype in numeric]


def _load_group_tx_catboost_models(models_dir: str) -> dict[int, CatBoostClassifier]:
    filenames = {
        0: "model_np_type7_catboost.cbm",
        1: "model_np_other_catboost.cbm",
        2: "model_card_catboost.cbm",
        3: "model_p2p_catboost.cbm",
    }

    models: dict[int, CatBoostClassifier] = {}
    for group_id, filename in filenames.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing tx CatBoost model: {path}")
        model = CatBoostClassifier()
        model.load_model(path)
        models[group_id] = model
        print(f"Loaded tx CatBoost model → {path}", flush=True)

    print()
    return models


def _load_group_tx_lgbm_models(models_dir: str) -> dict[int, list[lgb.Booster]]:
    boosters_by_group: dict[int, list[lgb.Booster]] = {}

    for group_id, group_name in TX_TYPE_GROUPS.items():
        pattern = os.path.join(
            models_dir,
            LGBM_MODEL_PATH_FMT.format(name=group_name, seed_idx="*"),
        )
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No LightGBM tx models found for group {group_id} ({group_name}) "
                f"using pattern {pattern!r}"
            )

        boosters = [lgb.Booster(model_file=path) for path in paths]
        for path in paths:
            print(f"Loaded tx LightGBM model → {path}", flush=True)
        boosters_by_group[group_id] = boosters

    print()
    return boosters_by_group


def _load_group_rf_bundles(models_dir: str) -> dict[int, dict]:
    bundles: dict[int, dict] = {}
    for group_id, group_name in TX_TYPE_GROUPS.items():
        path = os.path.join(models_dir, RF_MODEL_PATH_FMT.format(name=group_name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing RF bundle for group {group_id}: {path}")
        bundles[group_id] = joblib.load(path)
        print(f"Loaded RF bundle → {path}", flush=True)
    print()
    return bundles


def _load_group_fu_catboost_models(models_dir: str) -> dict[int, CatBoostClassifier]:
    models: dict[int, CatBoostClassifier] = {}
    for group_id, group_name in TX_TYPE_GROUPS.items():
        path = os.path.join(models_dir, CATBOOST_FU_MODEL_PATH_FMT.format(name=group_name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing F/U CatBoost model for group {group_id}: {path}")
        model = CatBoostClassifier()
        model.load_model(path)
        models[group_id] = model
        print(f"Loaded F/U CatBoost model → {path}", flush=True)
    print()
    return models


def _submission_with_suffix(base_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(base_path)
    return f"{base}_{suffix}{ext}"


# ── System helpers ─────────────────────────────────────────────────────–––––––––––––––––––––

def _estimate_matrix_ram_gb(n_rows: int, n_features: int, dtype_bytes: int = 4) -> float:
    """
    Estimate dense matrix RAM usage in GB.
    """
    return (n_rows * n_features * dtype_bytes) / (1024 ** 3)

