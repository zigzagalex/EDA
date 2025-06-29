import os

from pathlib import Path


# IO Helpers
def _write_lines(lines: list[str], file_name: str) -> None:
    """Write a list of lines to *file_name* in the current directory."""
    with open(file_name, "w", encoding="utf-8") as fh:
        fh.writelines([f"{ln}\n" for ln in lines])
    print(f"[+] Written {file_name} ({len(lines):,} lines)")


def _save_html(content: str, path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Written {path}")

