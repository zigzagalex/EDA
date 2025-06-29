import os
from pathlib import Path

import polars as pl

from helpers import _write_lines  # your custom function stays

_DEF_TXT = Path("./outputs/numerical_checks.txt")
_DEF_PLOT_DIR = Path("./outputs/plots")


def numerical_checks(
    df: pl.DataFrame,
    num_cols: list[str] | None = None,
    *,
    txt_out: os.PathLike | str = _DEF_TXT,
    plot_dir: os.PathLike | str = _DEF_PLOT_DIR,
    box_rows: int | None = None,
    no_html: bool = False,
) -> None:
    if not num_cols:
        print("[!] No numeric columns to analyse.")
        return

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Computing column stats...")
    lines = [
        "# ==== NUMERICAL SUMMARY ====",
        f"Analysed {len(num_cols)} numeric columns: {', '.join(num_cols)}",
        "",
        f"{'Column':20s}  {'min':>10s}  {'median':>10s}  {'mean':>10s}  "
        f"{'max':>10s}  {'std':>10s}  {'var':>10s}  {'skew':>8s}  {'kurt':>8s}",
    ]

    agg_exprs = []
    for c in num_cols:
        agg_exprs.extend(
            [
                pl.col(c).min().alias(f"{c}__min"),
                pl.col(c).median().alias(f"{c}__median"),
                pl.col(c).mean().alias(f"{c}__mean"),
                pl.col(c).max().alias(f"{c}__max"),
                pl.col(c).std().alias(f"{c}__std"),
                pl.col(c).var().alias(f"{c}__var"),
                pl.col(c).skew().alias(f"{c}__skew"),
                pl.col(c).kurtosis().alias(f"{c}__kurt"),
            ]
        )
    stats_row = df.select(agg_exprs).to_dicts()[0]

    for c in num_cols:
        lines.append(
            f"{c:20s}  "
            f"{stats_row[f'{c}__min']:<10.3g}  "
            f"{stats_row[f'{c}__median']:<10.3g}  "
            f"{stats_row[f'{c}__mean']:<10.3g}  "
            f"{stats_row[f'{c}__max']:<10.3g}  "
            f"{stats_row[f'{c}__std']:<10.3g}  "
            f"{stats_row[f'{c}__var']:<10.3g}  "
            f"{stats_row[f'{c}__skew']:<8.3g}  "
            f"{stats_row[f'{c}__kurt']:<8.3g}"
        )

    cor_df = df.select(num_cols).drop_nulls().corr()
    lines.append("\n# ==== CORRELATION MATRIX (Pearson) ====")
    lines.append(str(cor_df))

    _write_lines(lines, txt_out)
    print(f"[+] Stats written to {txt_out}")
