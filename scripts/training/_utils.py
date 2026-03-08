"""Shared utility helpers (progress display, file discovery, column selection)."""
from pathlib import Path

import polars as pl

from scripts.training.config import NON_FEATURE_COLS


# ── Progress display ──────────────────────────────────────────────────────────

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


# ── File / column helpers ─────────────────────────────────────────────────────

def _parquet_files(directory: str) -> list:
    return sorted(Path(directory).glob("*.parquet"))


def _get_feature_cols(df: pl.DataFrame) -> list:
    numeric = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and df[c].dtype in numeric]
