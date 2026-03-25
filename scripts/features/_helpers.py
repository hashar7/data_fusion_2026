import polars as pl
from polars import col


def prior_mean_expr(group_cols: list) -> pl.Expr:
    """Expanding mean of amount_clean for group_cols, excluding the current row."""
    n = (col("event_id").cum_count().over(group_cols) - 1).clip(lower_bound=1)
    s = col("amount_clean").cum_sum().over(group_cols) - col("amount_clean")
    return (s / n).fill_null(0)


def prior_std_expr(group_cols: list) -> pl.Expr:
    """Expanding sample std of amount_clean for group_cols, excluding the current row."""
    n   = col("event_id").cum_count().over(group_cols) - 1
    s   = col("amount_clean").cum_sum().over(group_cols) - col("amount_clean")
    sq  = (col("amount_clean") ** 2).cum_sum().over(group_cols) - col("amount_clean") ** 2
    var = (sq - s ** 2 / n.clip(lower_bound=1)) / (n - 1).clip(lower_bound=1)
    return var.clip(lower_bound=0).sqrt()
