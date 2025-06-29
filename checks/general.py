import polars as pl

from helpers import _write_lines


def general_checks(df: pl.DataFrame, out_file: str = "./outputs/general_checks.txt") -> None:
    """Run generic sanity checks and write a summary text report.

    Args:
        df: polars dataframe
        out_file: str name of file

    Returns:
        None

    Computes:
        * Total rows, columns and estimated in‑memory size.
        * Per‑column size in MiB.
        * Column data types.
        * Null proportion per column.
    """
    lines: list[str] = []

    # Overview
    n_rows, n_cols = df.shape
    total_bytes = df.estimated_size()
    lines.append("# ==== DATASET OVERVIEW ====")
    lines.append(f"Rows:        {n_rows:,}")
    lines.append(f"Columns:     {n_cols:,}")
    lines.append(f"Size (bytes):{total_bytes:,} ≈ {total_bytes / 1024**2:.2f} MiB")
    lines.append("")

    # Per‑column size & dtype
    lines.append("# ==== PER‑COLUMN SIZE & DTYPE ====")
    for col in df.columns:
        s = df[col]
        sz = s.estimated_size()
        dtype = s.dtype
        lines.append(f"{col:30s} {dtype!s:20s} {sz / 1024**2:10.2f} MiB")
    lines.append("")

    # Proportion of nulls
    lines.append("# ==== NULL PROPORTION ====")
    null_props = df.select(pl.all().is_null().mean()).to_dicts()[0]
    for col, prop in null_props.items():
        lines.append(f"{col:30s} {prop * 100:6.2f}% null")

    _write_lines(lines, out_file)
