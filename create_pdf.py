import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def compile_plots_pdf(
    plot_root: os.PathLike | str = "./outputs/plots",
    pdf_path: os.PathLike | str = "./outputs/numerical_plots.pdf",
    *,
    cols: list[str] | None = None,
    rows_per_page: int = 10,
) -> None:
    """
    Bundle the tri-plots for each numeric column into a single PDF.

    Args:
        plot_root : path-like
            Directory that contains the sub-dirs 'box', 'histogram', 'qq'.
        pdf_path : path-like
            Destination PDF file.
        cols : list[str] | None
            Explicit column order.  If None, infer from filenames in 'box'.
        rows_per_page : int
            How many rows (i.e. columns) to place on each PDF page.
    """
    root = Path(plot_root)
    box_dir, hist_dir, qq_dir = root / "box", root / "histogram", root / "qq"

    if cols is None:
        cols = sorted(p.stem for p in box_dir.glob("*.png"))
    if not cols:
        raise ValueError("No plot PNGs found under the expected folders.")

    with PdfPages(pdf_path) as pdf:
        pages = math.ceil(len(cols) / rows_per_page)
        for p in range(pages):
            start, end = p * rows_per_page, min((p + 1) * rows_per_page, len(cols))
            this_page_cols = cols[start:end]
            nrows = len(this_page_cols)

            fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(9, 2.5 * nrows))
            if nrows == 1:
                axes = axes.reshape(1, -1)

            for r, col in enumerate(this_page_cols):
                imgs = [
                    plt.imread(box_dir / f"{col}.png"),
                    plt.imread(hist_dir / f"{col}.png"),
                    plt.imread(qq_dir / f"{col}.png"),
                ]
                for c, img in enumerate(imgs):
                    ax = axes[r][c]
                    ax.imshow(img)
                    ax.axis("off")
                    if c == 1:
                        ax.set_title(col, fontsize=10, pad=10)

            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"[+] PDF saved to {pdf_path}")
