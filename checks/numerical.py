import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from helpers import _write_lines

_DEF_TXT = Path("./outputs/numerical_checks.txt")
_DEF_PLOT_DIR = Path("./outputs/plots")


def _plot_one(
    col: str,
    values: np.ndarray,
    box_dir: Path,
    hist_dir: Path,
    qq_dir: Path,
    bins: int | None = 30,
) -> None:
    """Worker function: create three plots for one column."""
    # Boxplot
    fig, ax = plt.subplots()
    ax.boxplot(values, vert=False, showfliers=True)
    ax.set_title(f"Box plot – {col}")
    ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(box_dir / f"{col}.png", dpi=150)
    plt.close(fig)

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins or "auto")
    ax.set_title(f"Histogram – {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(hist_dir / f"{col}.png", dpi=150)
    plt.close(fig)

    # QQ plot
    fig, ax = plt.subplots()
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(f"QQ plot – {col}")
    fig.tight_layout()
    fig.savefig(qq_dir / f"{col}.png", dpi=150)
    plt.close(fig)


def numerical_checks(
    df: pl.DataFrame,
    num_cols: list[str] | None = None,
    *,
    txt_out: os.PathLike | str = _DEF_TXT,
    plot_dir: os.PathLike | str = _DEF_PLOT_DIR,
    box_rows: int | None = None,
    no_html: bool = False,
    bins: int | None = 30,
    max_workers: int | None = None,
) -> None:
    """
    Compute stats **and** dump box/hist/qq plots.

    Args:
        df : pl.DataFrame
            The data.
        num_cols : list[str] | None
            Numeric columns to analyse.  If None/empty -> early exit.
        txt_out : path-like
            Where to write the text summary.
        plot_dir : path-like
            Root folder for plots.  Three sub-dirs will be created.
        bins : int | None
            Histogram bins.  None = 'auto'.
        max_workers : int | None
            Override #processes.  Default = cpu_count().
    """
    if not num_cols:
        print("[!] No numeric columns to analyse.")
        return

    plot_dir = Path(plot_dir)
    box_dir = plot_dir / "box"
    hist_dir = plot_dir / "histogram"
    qq_dir = plot_dir / "qq"
    for d in (box_dir, hist_dir, qq_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Text Summary
    print("[*] Computing column stats...")
    header = (
        f"{'Column':20s}  {'min':>10s}  {'median':>10s}  {'mean':>10s}  "
        f"{'max':>10s}  {'std':>10s}  {'var':>10s}  {'skew':>8s}  {'kurt':>8s}"
    )
    lines = [
        "# ==== NUMERICAL SUMMARY ====",
        f"Analysed {len(num_cols)} numeric columns: {', '.join(num_cols)}",
        "",
        header,
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

    # Plots
    print("[*] Generating plots (this can take a moment)…")
    tasks = [
        (
            col,
            df[col].drop_nulls().to_numpy(),
            box_dir,
            hist_dir,
            qq_dir,
            bins,
        )
        for col in num_cols
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_plot_one, *t) for t in tasks]
        for f in as_completed(futures):
            f.result()

    print(f"[+] Plots saved under:\n    • {box_dir}\n    • {hist_dir}\n    • {qq_dir}")
