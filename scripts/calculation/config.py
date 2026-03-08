"""
Default settings for the feature calculation pipeline.
Change values here — no other file needs to be touched for config changes.
"""

# Where partitioned parquet files are written
OUTPUT_DIR   = "../data_processed/"

# Number of customer buckets / output files.
# Increase if you run out of memory (peak RAM ≈ total_rows/N_PARTITIONS * bytes_per_row).
# 50 is a reasonable default for 32 GB RAM + ~200 M rows.
N_PARTITIONS = 50

# Parquet compression codec for output files
COMPRESSION  = "snappy"

# Int64 ID columns that must NOT be downcast to Int32 (overflow risk)
ID_COLS = {"customer_id", "event_id", "session_id"}
