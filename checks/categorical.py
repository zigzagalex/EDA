import polars as pl
from textwrap import indent

from helpers import _write_lines

def categorical_checks(
    df: pl.DataFrame,
    cat_cols: list[str] | None,
    *,
    top_n: int = 20,
    out_file: str = "./outputs/categorical_checks.txt",
) -> None:
    """Profile categorical variables and save a text report.

    Args:
        df: polars dataframe
        cat_cols: list of names of categorical columns to analyse
    Return:
        None

    Computes:
        For each categorical column we record:
        * Cardinality (unique categories)
        * Ordinal vs. Nominal flag (best‑effort heuristic)
        * Top‑*N* relative frequencies (normalized)
    """
    if cat_cols is None:
        print("No categorical columns selected")
        return None

    lines: list[str] = []
    lines.append("# ==== CATEGORICAL SUMMARY ====")
    lines.append(f"Detected {len(cat_cols)} categorical columns: {', '.join(cat_cols)}")
    lines.append("")

    for col in cat_cols:
        s = df[col]
        card = s.n_unique()

        lines.append("-" * 72)
        lines.append(f"Column: {col}  |  Cardinality: {card:,}")

        # Frequency table (Top‑N)
        vc = s.value_counts(sort=True)
        counts_col = "counts" if "counts" in vc.columns else "count"
        vc = vc.with_columns((pl.col(counts_col) / s.len()).alias("rel_freq"))
        head = vc.head(top_n)

        top5_cum = (
            vc.select(pl.col("rel_freq").head(5).sum()).item() * 100
            if card > 0
            else 0.0
        )

        lines.append(indent("Top categories (rel_freq shown):", "  "))
        for row in head.iter_rows(named=True):
            val = str(
                row["null"] if "null" in row else row["" + col]
            )  # handle null label
            rel = row["rel_freq"] * 100
            lines.append(indent(f"{val[:40]:40s} : {rel:6.2f}%", "    "))
        if card > top_n:
            lines.append(indent(f"... {card - top_n:,} more categories", "    "))
        lines.append("")

        lines.append(indent(f"Cumulative rel_freq (top 5): {top5_cum:6.2f}%", "  "))
        lines.append("")

    _write_lines(lines, out_file)
