"""
Thin entry point — kept for backward compatibility.
All logic lives in scripts/calculation/:
  config.py    — defaults (output dir, n_partitions, compression, ID_COLS)
  partition.py — get_customer_partitions, process_partition, _downcast
  pipeline.py  — build_processed_dataset (orchestrator)
"""
from scripts.calculation.pipeline import build_processed_dataset  # noqa: F401
